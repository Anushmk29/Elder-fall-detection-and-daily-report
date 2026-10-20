"""Run a day fetched by scripts/dvr.py through the live engine, so /day?date=<day> shows its report and timeline.

  python scripts/replay_day.py 2026-10-18 [--keep] [--follow] [--cams stairs,porch]

--follow starts on the clips already downloaded and keeps taking new ones until dvr.py writes data/dvr/<day>/.done.

Each camera's clips play in time order through the same rules as live (cameras.yaml: room, rules, sample_fps), all
cameras at once, as fast as the GPU allows. A clip is first checked for a person (one frame every CHECK_S, people
in ignore zones don't count): DVR motion is mostly trees, cars and IR noise at night, and skipping those clips cuts
the GPU time several-fold. The day's earlier events are moved to events-<day>.jsonl.bak unless --keep; with --cams
only those cameras' earlier events are dropped (to redo one camera after changing its zones).
"""
import argparse
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import live, perception  # noqa: E402
from dvr import OUT  # noqa: E402

CHECK_S = 2.0      # seconds between frames of the person check
CHECK_CONF = 0.5   # a real person scores well above the noise (det_conf 0.25 is for tracking, not for this)


def has_person(model, path, rules):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    tracker, ids = perception.new_tracker()
    idx, step = 0, max(int(fps * CHECK_S), 1)
    try:
        while cap.grab():
            if idx % step == 0:
                ok, frame = cap.retrieve()
                with live._predict_lock:
                    people = perception._detect(model, tracker, ids, frame, rules) if ok else []
                if any(p["conf"] >= CHECK_CONF for p in people):
                    return True
            idx += 1
    finally:
        cap.release()
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("day")
    ap.add_argument("--keep", action="store_true", help="add to the day's events instead of starting fresh")
    ap.add_argument("--follow", action="store_true", help="take clips as dvr.py downloads them")
    ap.add_argument("--cams", default="", help="only these cameras (comma-separated)")
    ap.add_argument("--until", default="", help="only clips that start before HH:MM (a pilot run)")
    a = ap.parse_args()
    only = set(filter(None, a.cams.split(",")))
    until = a.until.replace(":", "") + "00" if a.until else "999999"
    log = live.LIVE_DIR / f"events-{a.day}.jsonl"
    if log.exists() and not a.keep:
        lines = log.read_text(encoding="utf-8").splitlines()
        log.replace(log.with_suffix(".jsonl.bak"))
        if only:
            log.write_text("".join(x + "\n" for x in lines if x.strip() and json.loads(x)["camera"] not in only),
                           encoding="utf-8")
    engine = live.Engine(None, sender=lambda *x, **k: None)
    stats = {}

    def clips(cfg):
        """The camera's clips in time order; with --follow, waits for each next one until the download is done."""
        d, seen = OUT / a.day / cfg["name"], set()
        while True:
            done = not a.follow or (OUT / a.day / ".done").exists()
            new = sorted(f for f in d.glob("??????.mp4") if f not in seen and f.stem < until
                         and (done or time.time() - f.stat().st_mtime > 30))  # still being written
            if new:
                seen.add(new[0])
                yield new[0]
            elif done or (until < "999999" and any(f.stem >= until for f in d.glob("??????.mp4"))):
                return
            else:
                time.sleep(20)

    def run(cfg):
        rules = live.live_rules(cfg.get("rules"))
        st = stats[cfg["name"]] = {"with_person": 0, "done": 0}
        for f in clips(cfg):
            if has_person(engine.model, f, rules):
                st["with_person"] += 1
                at = datetime.strptime(f"{a.day} {f.stem}", "%Y-%m-%d %H%M%S").timestamp()
                cam = engine.add({**cfg, "source": str(f), "start_at": at, "realtime": False, "notify": False})
                cam.thread.join()
            st["done"] += 1
            st["at"] = f.stem

    threads = [threading.Thread(target=run, args=(c,)) for c in live.load_cameras() if not only or c["name"] in only]
    for t in threads:
        t.start()
    t0 = time.time()
    while any(t.is_alive() for t in threads):
        time.sleep(30)
        print(f"{(time.time() - t0) / 60:5.1f} min  " + "  ".join(
            f"{k} {v['done']} clips, {v['with_person']} with a person, at {v.get('at', '-')}" for k, v in stats.items()),
            flush=True)
    if engine.ai:
        print("waiting for the AI reviews still running...", flush=True)
        engine.pool.shutdown(wait=True)
        print(f"AI spend today ${engine.ai.spent_today():.3f}", flush=True)
    print(f"done in {(time.time() - t0) / 60:.0f} min; open http://localhost:8000/day?date={a.day}")


if __name__ == "__main__":
    main()
