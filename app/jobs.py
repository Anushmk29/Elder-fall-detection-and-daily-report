"""Single worker thread; job state as JSON files; uploads and results purged after RETENTION_HOURS."""
import json
import logging
import os
import queue
import re
import secrets
import shutil
import threading
import time
from pathlib import Path

ID_RE = re.compile(r"^[A-Za-z0-9_-]{20,64}$")
log = logging.getLogger("jobs")


class Jobs:
    def __init__(self, data_dir, process, retention_hours=24.0):
        self.dir = Path(data_dir)
        self.jobs_dir, self.uploads, self.results = self.dir / "jobs", self.dir / "uploads", self.dir / "results"
        for d in (self.jobs_dir, self.uploads, self.results):
            d.mkdir(parents=True, exist_ok=True)
        self.process = process
        self.retention = retention_hours * 3600
        self.q, self.waiting, self.lock = queue.Queue(), [], threading.Lock()
        self.thread = None

    def start(self):
        if self.thread:
            return
        for p in sorted(self.jobs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime):
            st = self._read(p.stem)
            if st and st["status"] in ("queued", "processing") and (self.uploads / st["file"]).exists():
                self._enqueue(st["id"])  # resume after a restart
        self.thread = threading.Thread(target=self._loop, daemon=True, name="job-worker")
        self.thread.start()

    def new_upload(self, ext):
        job_id = secrets.token_urlsafe(18)  # unguessable; doubles as the result id
        return job_id, self.uploads / f"{job_id}{ext}"

    def submit(self, job_id, path, name=""):
        self._write({"id": job_id, "status": "queued", "progress": 0, "stage": "Waiting in queue",
                     "created": time.time(), "file": Path(path).name, "name": name[:120]})
        self._enqueue(job_id)

    def get(self, job_id):
        st = self._read(job_id) if ID_RE.match(job_id or "") else None
        if st:
            with self.lock:
                st["queue_position"] = self.waiting.index(job_id) + 1 if job_id in self.waiting else 0
            st.pop("file", None)
        return st

    def all(self):
        """Every job still kept (newest first), for the local history page."""
        out = [self._read(p.stem) for p in self.jobs_dir.glob("*.json")]
        return sorted((st for st in out if st), key=lambda st: st["created"], reverse=True)

    def delete(self, job_id):
        """Remove a finished job's result, upload and state. False if unknown or still running."""
        st = self._read(job_id) if ID_RE.match(job_id or "") else None
        if not st or st["status"] in ("queued", "processing"):
            return False
        shutil.rmtree(self.results / job_id, ignore_errors=True)
        (self.uploads / st["file"]).unlink(missing_ok=True)
        (self.jobs_dir / f"{job_id}.json").unlink(missing_ok=True)
        return True

    def queued(self):
        return len(self.waiting)

    def _enqueue(self, job_id):
        with self.lock:
            self.waiting.append(job_id)
        self.q.put(job_id)

    def _loop(self):
        last_clean = 0.0
        while True:
            if time.time() - last_clean > 600:
                self.cleanup()
                last_clean = time.time()
            try:
                job_id = self.q.get(timeout=60)
            except queue.Empty:
                continue
            with self.lock:
                self.waiting.remove(job_id)
            self._run(job_id)

    def _run(self, job_id):
        st = self._read(job_id)
        if not st:
            return
        src = self.uploads / st["file"]
        st.update(status="processing", stage="Starting", progress=0)
        self._write(st)
        last = [0.0]

        def progress(pct, stage):
            if pct - last[0] >= 1 or stage != st["stage"]:
                last[0] = pct
                st.update(progress=round(pct), stage=stage)
                try:
                    self._write(st)
                except OSError:  # a missed progress update must never fail the job
                    log.warning("could not write progress for %s", job_id)

        try:
            self.process(src, self.results / job_id, progress)
            st.update(status="done", progress=100, stage="Done")
        except Exception:
            log.exception("job %s failed", job_id)
            shutil.rmtree(self.results / job_id, ignore_errors=True)
            st.update(status="error", stage="Failed", error="We couldn't process this clip. Try another one.")
        finally:
            src.unlink(missing_ok=True)  # the raw upload is never needed after processing
        self._write(st)

    def cleanup(self):
        cutoff = time.time() - self.retention
        for p in self.jobs_dir.glob("*.json"):
            st = self._read(p.stem)
            if st and st["created"] < cutoff and st["status"] not in ("queued", "processing"):
                shutil.rmtree(self.results / st["id"], ignore_errors=True)
                (self.uploads / st["file"]).unlink(missing_ok=True)
                p.unlink(missing_ok=True)
                log.info("purged job %s", st["id"])

    def _read(self, job_id):
        try:
            return json.loads((self.jobs_dir / f"{job_id}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _write(self, st):
        p = self.jobs_dir / f"{st['id']}.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(st), encoding="utf-8")
        for attempt in range(10):  # Windows refuses to replace a file another thread is reading
            try:
                os.replace(tmp, p)
                return
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.05)
