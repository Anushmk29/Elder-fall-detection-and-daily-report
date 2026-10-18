"""Replay labelled clips through the real live engine (live.Camera, rolling window, 1 s evaluations) as fast as possible,
and measure what the *family* would receive. Uses the cached YOLO output, so it doesn't need the GPU.

Usage: python scripts/eval_live.py "C:/.../archive (4)" D:/datasets/le2i [--hold 70]

1. Stayed down (Le2i, last frame held --hold s, i.e. the person never gets up): does the FALL ALERT fire, how long
   after the labelled fall start, and does the separate "on the floor" alert fire as a safety net?
   Held frames are frozen, so this can't tell "moving" from "not moving"; part 3 measures that.
2. False pushes per hour (raw clips, MCFD + Le2i): every pushed event (alert/warn) that isn't within a labelled fall
   (2 s before its start to 60 s after its end).
3. Motion of really fallen people (labelled fall end + 4 s): what floor_moving_bl_s has to stay above so a fallen
   person isn't called "moving".
"""
import argparse
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "scripts"))
import eval_mcfd as ev  # noqa: E402
import eval_staged as es  # noqa: E402
from app import events, live, perception, pipeline, tracks  # noqa: E402

NOON = datetime(2026, 10, 18, 12, 0).timestamp()
PUSH = live.PUSH


class Recorder:
    def __init__(self):
        self.events = []

    def emit(self, cam, typ, severity, title, detail, t=None, **kw):
        self.events.append({"type": typ, "severity": severity, "t": t, "title": title})
        return {"type": typ, "severity": severity, "detail": detail, "snapshot": None}

    def saw(self, cam, places, t):
        pass

    def second_opinion(self, cam, ev_):
        pass


def replay(frames, meta, rules_path=None):
    """Feed cached frames to a live.Camera exactly like Camera._read does; returns the emitted events."""
    rec = Recorder()
    cam = live.Camera({"name": "replay", "source": "replay.mp4", "rules": rules_path}, rec, model=None)
    cam.clock_start, cam.t0_wall = 12 * 3600, NOON
    frame = _Frame(meta["width"], meta["height"])
    last_eval = -1.0
    for f in frames:
        cam.window.append({"i": f["i"], "t": f["t"], "people": [dict(p) for p in f["people"]]})
        while cam.window and cam.window[0]["t"] < f["t"] - cam.window_s:
            cam.window.popleft()
        if f["t"] - last_eval >= 1.0:
            last_eval = f["t"]
            cam._evaluate(f["t"], frame)
    return rec.events


class _Frame:
    """Camera._evaluate only reads frame.shape (and passes the frame on for snapshots, which need cam.jpeg)."""
    def __init__(self, w, h):
        self.shape = (h, w, 3)


def le2i_clips(root, base):
    for room in ("Coffee_room_01", "Coffee_room_02", "Home_01", "Home_02"):
        folder = Path(root) / room / room / "Videos"
        cache = (pipeline.DATA_DIR / "staged_cache" / f"{Path(perception.MODEL_PATH).stem}-c{base.get('det_conf', 0.25)}"
                 / f"{room}_Videos")
        for name, rows in es.read_labels(folder / "labels.csv").items():
            frames, meta = ev.load_frames(folder / name, base, cache)
            yield f"{room}/{name}", frames, meta, [r for r in rows if r["kind"] != "none"], folder / f"{rows[0]['camera']}.yaml"


def mcfd_clips(root, base):
    labels = ev.fall_windows(Path(root) / "data_tuple3.csv")
    rules = dict(base, source_fps=ev.FPS)
    for v in sorted((Path(root) / "dataset" / "dataset").glob("chute*/cam*.avi")):
        chute, cam = int(v.parent.name[5:]), int(v.stem[3:])
        frames, meta = ev.load_frames(v, rules)
        rows = [{"kind": "fall", "start_s": a, "end_s": b} for a, b in labels.get((chute, cam), [])]
        yield f"{v.parent.name}/{v.name}", frames, meta, rows, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mcfd")
    ap.add_argument("le2i")
    ap.add_argument("--hold", type=float, default=70)
    a = ap.parse_args()
    base = pipeline.load_rules()
    live_cfg = live.live_rules()
    no_rec = live_cfg["live"]["no_recovery_s"]

    # 1. stayed down
    n = Counter()
    latency, floor_sev = [], Counter()
    for name, frames, meta, rows, zones in le2i_clips(a.le2i, base):
        if not rows:
            continue
        frames = [dict(f) for f in frames]
        es.hold(frames, meta, a.hold, base["sample_fps"])
        got = replay(frames, meta, zones if zones.exists() else None)
        start = rows[0]["start_s"]
        alert = next((e for e in got if e["type"] == "fall_alert" and e["t"] >= start - 2), None)
        floor = next((e for e in got if e["type"] == "lying_outside_rest" and e["t"] >= start - 2), None)
        n["falls"] += 1
        n["fall_alert"] += alert is not None
        n["floor"] += floor is not None
        n["either"] += bool(alert or (floor and floor["severity"] in PUSH))
        if alert:
            latency.append(alert["t"] - start)
        if floor:
            floor_sev[floor["severity"]] += 1
    latency.sort()
    print(f"1. STAYED DOWN (Le2i, last frame held {a.hold:.0f} s), {n['falls']} falls")
    print(f"   FALL ALERT fired           {n['fall_alert']}/{n['falls']} = {n['fall_alert'] / n['falls']:.0%}"
          + (f", median {latency[len(latency) // 2]:.0f} s after the fall started (rule: detect + {no_rec:.0f} s)"
             if latency else ""))
    print(f"   'on the floor' fired       {n['floor']}/{n['falls']} = {n['floor'] / n['falls']:.0%}  severities {dict(floor_sev)}"
          f"  (info = already covered by the fall alert)")
    print(f"   family told by either      {n['either']}/{n['falls']} = {n['either'] / n['falls']:.0%}")

    # 2. false pushes
    secs, false, by_type = 0.0, 0, Counter()
    for gen in (mcfd_clips(a.mcfd, base), le2i_clips(a.le2i, base)):
        for name, frames, meta, rows, zones in gen:
            secs += meta["duration"]
            for e in replay(frames, meta, zones if zones and zones.exists() else None):
                if e["severity"] in PUSH and not any(r["start_s"] - 2 <= e["t"] <= r["end_s"] + 60 for r in rows):
                    false += 1
                    by_type[e["type"]] += 1
    print(f"\n2. FALSE PUSHES (raw clips, MCFD + Le2i, {secs / 3600:.1f} h of video): {false} = {false / (secs / 3600):.1f}/h"
          f"  by type {dict(by_type)}")

    # 3. how much do fallen people move?
    vals = []
    for gen in (mcfd_clips(a.mcfd, base), le2i_clips(a.le2i, base)):
        for name, frames, meta, rows, zones in gen:
            if not rows:
                continue
            for tr in tracks.build(frames, meta, base):
                for r in rows:
                    down = [s for s in tr["samples"] if r["end_s"] <= s["t"] <= r["end_s"] + 4 and s["posture"] == "lying"]
                    if len(down) >= 6:
                        vals.append(events.motion(tr["samples"], r["end_s"], r["end_s"] + 4))
    vals.sort()
    q = lambda p: vals[min(int(p * len(vals)), len(vals) - 1)]  # noqa: E731
    print(f"\n3. MOTION of fallen people on the floor (first 4 s), {len(vals)} falls: median {q(0.5):.3f}, "
          f"75% {q(0.75):.3f}, 90% {q(0.9):.3f} body lengths/s; floor_moving_bl_s = {base['floor_moving_bl_s']} "
          f"calls {sum(v >= base['floor_moving_bl_s'] for v in vals)}/{len(vals)} of them 'moving'")


if __name__ == "__main__":
    main()
