import csv
import html
import ipaddress
import json
import logging
import os
import re
import secrets
import shutil
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import pipeline
from .jobs import ID_RE, Jobs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

BASE = pipeline.BASE
STATIC, EXAMPLES = BASE / "static", BASE / "examples"
MAX_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "100")) * 1024 * 1024
MAX_SECONDS = int(os.environ.get("MAX_UPLOAD_SECONDS", "600"))  # POC: up to 10 min clips
UPLOADS_PER_HOUR = int(os.environ.get("UPLOADS_PER_HOUR", "5"))
MAX_QUEUE = 20
PASSCODE = os.environ.get("DEMO_PASSCODE", "")
FORMATS = {".mp4": ("mp4", "mov"), ".mov": ("mp4", "mov"), ".webm": ("webm",), ".mkv": ("matroska",),
           ".avi": ("avi",)}  # ext -> accepted ffprobe names
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,40}$")
BATCH = pipeline.DATA_DIR / "batch"  # written by scripts/batch_eval.py
LOCAL = {"127.0.0.1", "::1"}

EXAMPLES.mkdir(exist_ok=True)
jobs = Jobs(pipeline.DATA_DIR, pipeline.run, float(os.environ.get("RETENTION_HOURS", "24")))


engine = None  # live.Engine, created at startup when cameras.yaml exists, or by the first "play as live camera"
_engine_lock = threading.Lock()


def _engine():
    global engine
    with _engine_lock:
        if engine is None:
            from . import live  # imports torch/YOLO; keep plain upload startup light
            engine = live.Engine(live.CAMERAS_PATH if live.CAMERAS_PATH.exists() else None)
            engine.start()
    return engine


@asynccontextmanager
async def lifespan(app):
    jobs.start()
    if (BASE / "cameras.yaml").exists() or os.environ.get("CAMERAS_FILE"):
        threading.Thread(target=_engine, daemon=True, name="live-start").start()
    yield
    if engine:
        engine.stop()


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/media/examples", StaticFiles(directory=EXAMPLES), name="examples")
app.mount("/media/results", StaticFiles(directory=jobs.results), name="results")  # no listing; ids are unguessable


@app.middleware("http")
async def security_headers(request, call_next):
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "no-referrer"  # result ids live in URLs
    resp.headers["X-Robots-Tag"] = "noindex"
    return resp


_hits, _hits_lock = defaultdict(deque), threading.Lock()


def _rate_ok(ip):
    now = time.time()
    with _hits_lock:
        d = _hits[ip]
        while d and d[0] < now - 3600:
            d.popleft()
        if len(d) >= UPLOADS_PER_HOUR:
            return False
        d.append(now)
        return True


def _local_only(request):
    # batch clips are private test footage: only this machine may see them (behind Caddy the client is not loopback)
    if not request.client or request.client.host not in LOCAL:
        raise HTTPException(404)


def _batch_path(rel):
    p = (BATCH / rel).resolve()
    if not p.is_relative_to(BATCH.resolve()):
        raise HTTPException(404)
    return p


def _result(rid, request=None):
    if rid.startswith("batch/") and request and request.client and request.client.host in LOCAL:
        d = _batch_path(rid[6:])
        return (d, f"/media/{rid}/") if (d / "result.json").is_file() else (None, None)
    if SLUG_RE.match(rid) and (EXAMPLES / rid / "result.json").is_file():
        return EXAMPLES / rid, f"/media/examples/{rid}/"
    if ID_RE.match(rid) and (jobs.results / rid / "result.json").is_file():
        return jobs.results / rid, f"/media/results/{rid}/"
    return None, None


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/r/{rid:path}")
def result_page(rid: str, request: Request):
    if _result(rid, request)[0] or jobs.get(rid):
        return FileResponse(STATIC / "result.html")
    return HTMLResponse("<h1>Not found</h1><p>This result doesn't exist or has expired. <a href='/'>Back</a></p>", 404)


@app.get("/healthz")
def healthz():
    return {"ok": True, "queued": jobs.queued()}


@app.get("/api/config")
def config():
    return {"passcode": bool(PASSCODE), "max_mb": MAX_BYTES // 2**20, "max_seconds": MAX_SECONDS}


@app.get("/api/examples")
def examples():
    out = []
    for p in sorted(EXAMPLES.glob("*/result.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        out.append({"id": p.parent.name, "title": r.get("title"), "description": r.get("description", ""),
                    "duration": r["duration"], "poster": f"/media/examples/{p.parent.name}/{r['poster']}",
                    "alerts": sum(e["severity"] != "info" for e in r["events"])})
    return out


@app.get("/api/results/{rid:path}")
def get_result(rid: str, request: Request):
    d, media = _result(rid, request)
    if not d:
        raise HTTPException(404, "Result not found or expired")
    return {**json.loads((d / "result.json").read_text(encoding="utf-8")), "id": rid, "media_base": media}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    st = jobs.get(job_id)
    if not st:
        raise HTTPException(404, "Job not found")
    return st


@app.post("/api/jobs")
def create_job(request: Request, file: UploadFile = File(...), consent: str = Form(""), passcode: str = Form("")):
    if not _rate_ok(request.client.host if request.client else "?"):
        raise HTTPException(429, f"Upload limit reached ({UPLOADS_PER_HOUR} per hour). Please try later.")
    if PASSCODE and not secrets.compare_digest(passcode.encode(), PASSCODE.encode()):
        raise HTTPException(403, "Wrong demo passcode.")
    if consent.lower() not in ("yes", "true", "on", "1"):
        raise HTTPException(400, "Please confirm consent before uploading.")
    if jobs.queued() >= MAX_QUEUE:
        raise HTTPException(503, "The demo is busy right now. Please try again in a few minutes.")
    ext = Path(file.filename or "").suffix.lower()
    if ext not in FORMATS:
        raise HTTPException(400, "Please upload an MP4, MOV, WebM, MKV or AVI file.")
    if file.size is not None and file.size > MAX_BYTES:
        raise HTTPException(413, f"File is larger than {MAX_BYTES // 2**20} MB.")
    job_id, path = jobs.new_upload(ext)
    try:
        with open(path, "wb") as out:
            shutil.copyfileobj(file.file, out, 1 << 20)
        if path.stat().st_size > MAX_BYTES:
            raise HTTPException(413, f"File is larger than {MAX_BYTES // 2**20} MB.")
        try:
            info = pipeline.probe(path)
        except (ValueError, OSError):
            raise HTTPException(400, "That file doesn't look like a playable video.")
        if not any(f in info["format"] for f in FORMATS[ext]):
            raise HTTPException(400, "The file contents don't match the file's extension.")
        if not 0 < info["duration"] <= MAX_SECONDS + 0.5:
            raise HTTPException(400, f"Clips must be {MAX_SECONDS} seconds or shorter.")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    jobs.submit(job_id, path)
    return {"id": job_id, "status_url": f"/api/jobs/{job_id}", "result_url": f"/r/{job_id}"}


@app.get("/batch")
def batch_page(request: Request):
    _local_only(request)
    return FileResponse(STATIC / "batch.html")


@app.get("/media/batch/{rel:path}")
def batch_media(rel: str, request: Request):
    _local_only(request)
    p = _batch_path(rel)
    if not p.is_file():
        raise HTTPException(404)
    return FileResponse(p)


VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}  # same as scripts/batch_eval.py


def _read_json(f, default):
    return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else default


def _videos(folder):
    return sorted(p.name for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS) if folder.is_dir() else []


def _is_group(d):
    return (d / "summary.csv").is_file() or (d / "batch.json").is_file()


@app.get("/api/batch")
def batch_list(request: Request):
    """Every batch folder with each clip's status, a history log, and source folders not run yet."""
    _local_only(request)
    groups, history, sources = [], [], set()
    for d in sorted({p.parent for n in ("summary.csv", "batch.json") for p in BATCH.rglob(n)}):
        name, info, reviews = d.relative_to(BATCH).as_posix(), _read_json(d / "batch.json", {}), _read_json(d / "review.json", {})
        rows = {}
        if (d / "summary.csv").is_file():
            with open(d / "summary.csv", newline="", encoding="utf-8") as f:
                rows = {r["file"]: r for r in csv.DictReader(f)}
        src = Path(info["source"]) if info.get("source") else None
        if src:
            sources.add(src.resolve())
        clips = []
        for file in sorted(set(rows) | set(_videos(src) if src else [])):
            rel = (d / Path(file).stem).relative_to(BATCH).as_posix()
            res, row = BATCH / rel / "result.json", rows.get(file, {})
            done = res.is_file()
            at = res.stat().st_mtime if done else None
            status = ("error" if row.get("error") else "not processed" if not done
                      else "reviewed" if file in reviews else "to review")
            clips.append({**row, "file": file, "id": f"batch/{rel}", "status": status, "processed_at": at,
                          "video": f"/media/batch/{rel}/annotated.mp4" if done else "",
                          "poster": f"/media/batch/{rel}/poster.jpg" if (BATCH / rel / "poster.jpg").is_file() else "",
                          "review": reviews.get(file, {})})
            if done:
                history.append({"at": at, "group": name, "file": file, "what": "processed",
                                "detail": f"{row.get('falls', '?')} fall(s) detected"})
        groups.append({"group": name, "source": str(src or ""), "clips": clips})
        if (d / "history.jsonl").is_file():
            history += [json.loads(x) for x in (d / "history.jsonl").read_text(encoding="utf-8").splitlines() if x]
    todo = [{"name": sub.name, "folder": str(sub), "videos": len(_videos(sub))}
            for parent in sorted({s.parent for s in sources}) for sub in sorted(parent.iterdir())
            if sub.is_dir() and sub.resolve() not in sources and _videos(sub)]
    history.sort(key=lambda h: h["at"], reverse=True)
    return {"groups": groups, "todo": todo, "history": history[:300]}


@app.post("/api/batch/review")
def batch_review(request: Request, group: str = Body(...), file: str = Body(...), fall: int | None = Body(None),
                 note: str = Body("")):
    _local_only(request)
    d = _batch_path(group)
    if not _is_group(d) or fall not in (None, 0, 1):
        raise HTTPException(400, "Unknown batch or bad label")
    reviews = _read_json(d / "review.json", {})
    old, new = reviews.get(file, {}), {"fall": fall, "note": note[:1000]}
    reviews[file] = new
    (d / "review.json").write_text(json.dumps(reviews, indent=1), encoding="utf-8")
    words = {1: "real fall", 0: "no fall", None: "unsure"}
    detail = "; ".join(([f"labelled {words[fall]}"] if old.get("fall", "x") != fall else [])
                       + ([f"note: {new['note']}"] if old.get("note", "") != new["note"] else []))
    if detail:
        with open(d / "history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"at": time.time(), "group": d.relative_to(BATCH.resolve()).as_posix(), "file": file,
                                "what": "reviewed", "detail": detail}) + "\n")
    return new


# ---- live cameras (home network only: these show real footage from the home) -------------------------------
SNAP_RE = re.compile(r"^[0-9]+-[A-Za-z0-9_.-]+\.jpg$")
CAM_RE = re.compile(r"^[A-Za-z0-9_.-]{1,40}$")


def _lan_only(request):
    host = request.client.host if request.client else ""
    try:
        ok = ipaddress.ip_address(host).is_private  # loopback, 10/8, 172.16/12, 192.168/16 (phones, HA in Docker)
    except ValueError:
        ok = False
    if not ok:
        raise HTTPException(404)


@app.get("/live")
def live_page(request: Request):
    _lan_only(request)
    return FileResponse(STATIC / "live.html")


@app.get("/api/live/state")
def live_state(request: Request):
    _lan_only(request)
    if not engine:
        return {"running": False, "cameras": [], "events": []}
    return {"running": True, **engine.status(), "events": list(engine.recent)[-60:][::-1]}


@app.get("/api/live/events")
def live_events(request: Request, date: str = ""):
    _lan_only(request)
    if date and not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
        raise HTTPException(400, "date must be YYYY-MM-DD")
    from . import live
    return live.read_events(date or None)


@app.get("/live/snapshots/{name}")
def live_snapshot(name: str, request: Request):
    _lan_only(request)
    from . import live
    p = live.SNAP_DIR / name
    if not SNAP_RE.match(name) or not p.is_file():
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/live/{cam}.mjpg")
def live_mjpeg(cam: str, request: Request):
    _lan_only(request)
    c = engine.cameras.get(cam) if engine and CAM_RE.match(cam) else None
    if not c:
        raise HTTPException(404)

    def frames():
        last = None
        while not c.stop_event.is_set():
            if c.jpeg is not last and c.jpeg:
                last = c.jpeg
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + last + b"\r\n"
            time.sleep(0.08)
    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/live/play")
def live_play(request: Request, file: UploadFile = File(...), room: str = Form("")):
    """'Play as live camera': a recorded clip (e.g. a staged fall) runs through the live engine in real time, so it
    produces the same alerts, event log and phone pushes as a real camera."""
    _lan_only(request)
    ext = Path(file.filename or "").suffix.lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(400, "Please upload a video file.")
    from . import live
    d = live.LIVE_DIR / "files"
    d.mkdir(parents=True, exist_ok=True)
    name = "file-" + re.sub(r"[^A-Za-z0-9_-]", "", Path(file.filename).stem)[:24] or "file-clip"
    path = d / f"{name}{ext}"
    with open(path, "wb") as out:
        shutil.copyfileobj(file.file, out, 1 << 20)
    try:
        pipeline.probe(path)
    except (ValueError, OSError):
        path.unlink(missing_ok=True)
        raise HTTPException(400, "That file doesn't look like a playable video.")
    _engine().add({"name": name, "source": str(path), "room": room.strip()[:40] or None})
    return {"camera": name}


@app.get("/live/report")
@app.get("/live/report/{day}")
def live_report(request: Request, day: str = ""):
    _lan_only(request)
    day = day or time.strftime("%Y-%m-%d")
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", day):
        raise HTTPException(404)
    from . import daily, llm
    text, m, base, flags = daily.report(day, llm.LLM(pipeline.DATA_DIR / "llm"))
    rows = "".join(f"<tr><td>{k}</td><td>{m.get(k)}</td><td>{'' if k not in base['metrics'] else round(base['metrics'][k]['mean'], 1)}</td>"
                   f"<td>{flags.get(k, '')}</td></tr>" for k in daily.NUMERIC)
    return HTMLResponse(
        "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
        "<meta name=robots content=noindex><title>Daily report</title><link rel=stylesheet href=/static/style.css>"
        "<header class=top><a class=brand href=/live>ElderCare <span>Guardian</span></a></header><main>"
        f"<section class=card><h1>Daily report</h1><p style='white-space:pre-wrap'>{html.escape(text)}</p></section>"
        f"<section class=card><h2>Today vs 7-day average</h2><table><tr><th>Metric</th><th>Today</th><th>Average</th>"
        f"<th></th></tr>{rows}</table></section></main>")
