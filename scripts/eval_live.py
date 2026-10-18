"""Replay labelled clips through the real live engine (live.Camera, rolling window, 1 s evaluations) as fast as possible,
and measure what the *family* would receive. Uses the cached YOLO output, so it doesn't need the GPU.

Usage: python scripts/eval_live.py "C:/.../archive (4)" D:/datasets/le2i [--urfall D:/datasets/urfall] [--hold 70]

1. Stayed down (Le2i, last frame held --hold s, i.e. the person never gets up): does the FALL ALERT fire, how long
   after the labelled fall start, and does the separate "on the floor" alert fire as a safety net?
   Held frames are frozen, so this can't tell "moving" from "not moving"; part 3 measures that.
2. False pushes per hour (raw clips, MCFD + Le2i): every pushed event (alert/warn) that isn't within a labelled fall
   (2 s before its start to 60 s after its end).
3. Motion of really fallen people (labelled fall end + 4 s): what floor_moving_bl_s has to stay above so a fallen
   person isn't called "moving". With --urfall, also people on the floor *on purpose* (UR Fall ADL look-alikes
   outside bed/sofa): what it has to stay below so they are.
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


def staged_clips(folder, base, prefix=""):
    """A labels.csv folder (eval_staged format), with eval_staged's YOLO cache."""
    cache = (pipeline.DATA_DIR / "staged_cache" / f"{Path(perception.MODEL_PATH).stem}-c{base.get('det_conf', 0.25)}"
             / f"{folder.parent.name}_{folder.name}")
    for name, rows in es.read_labels(folder / "labels.csv").items():
        frames, meta = ev.load_frames(folder / name, base, cache)
        yield f"{prefix}{name}", frames, meta, [r for r in rows if r["kind"] != "none"], folder / f"{rows[0]['camera']}.yaml"


def le2i_clips(root, base):
    for room in ("Coffee_room_01", "Coffee_room_02", "Home_01", "Home_02"):
        yield from staged_clips(Path(root) / room / room / "Videos", base, f"{room}/")


def urfall_clips(root, base):
    yield from staged_clips(Path(root) / "Videos", base, "urfall/") if root else ()


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
    ap.add_argument("--urfall", help="UR Fall folder prepared by scripts/urfall_labels.py")
    ap.add_argument("--hold", type=float, default=70)
    a = ap.parse_args()
    base = pipeline.load_rules()
    live_cfg = live.live_rules()
    no_rec = live_cfg["live"]["no_recovery_s"]

    # 1. stayed down
    for title, clips in (("Le2i", le2i_clips(a.le2i, base)), ("UR Fall", urfall_clips(a.urfall, base))):
        stayed_down(title, clips, a.hold, base, no_rec)

    # 2. false pushes
    secs, false, by_type = 0.0, 0, Counter()
    for gen in (mcfd_clips(a.mcfd, base), le2i_clips(a.le2i, base), urfall_clips(a.urfall, base)):
        for name, frames, meta, rows, zones in gen:
            secs += meta["duration"]
            for e in replay(frames, meta, zones if zones and zones.exists() else None):
                if e["severity"] in PUSH and not any(r["start_s"] - 2 <= e["t"] <= r["end_s"] + 60
                                                     for r in rows if r["kind"] in es.FALLS):
                    false += 1
                    by_type[e["type"]] += 1
    print(f"\n2. FALSE PUSHES (raw clips, MCFD + Le2i{' + UR Fall' if a.urfall else ''}, {secs / 3600:.1f} h of video): "
          f"{false} = {false / (secs / 3600):.1f}/h  by type {dict(by_type)}")

    # 3. how much do people on the floor move? fallen (should be below the threshold) vs on purpose (above)
    fallen, purpose = [], []
    for gen in (mcfd_clips(a.mcfd, base), le2i_clips(a.le2i, base), urfall_clips(a.urfall, base)):
        for name, frames, meta, rows, zones in gen:
            if not rows:
                continue
            rest = {z["name"] for z in pipeline.load_rules(zones)["zones"] if z.get("kind") == "rest"} if zones else set()
            for tr in tracks.build(frames, meta, base):
                events.analyze([tr], meta, pipeline.load_rules(zones) if zones else base)  # sets s["zones"]
                for r in rows:
                    a_, b_ = (r["end_s"], r["end_s"] + 4) if r["kind"] in es.FALLS else (r["start_s"], r["end_s"])
                    down = [s for s in tr["samples"] if a_ <= s["t"] <= b_ and s["posture"] == "lying"
                            and not set(s.get("zones", [])) & rest]
                    if len(down) >= 6:
                        (fallen if r["kind"] in es.FALLS else purpose).append(events.motion(down, down[0]["t"], down[-1]["t"]))
    thr = base["floor_moving_bl_s"]
    for label, vals in (("fallen (first 4 s)", fallen), ("on the floor on purpose (UR Fall ADL)", purpose)):
        if not vals:
            continue
        vals.sort()
        q = lambda p: vals[min(int(p * len(vals)), len(vals) - 1)]  # noqa: E731
        print(f"\n3. MOTION of people {label}, {len(vals)}: 10% {q(0.1):.3f}, median {q(0.5):.3f}, 90% {q(0.9):.3f} "
              f"body lengths/s; floor_moving_bl_s = {thr} calls {sum(v >= thr for v in vals)}/{len(vals)} 'moving'")


def stayed_down(title, clips, hold, base, no_rec):
    n = Counter()
    latency, floor_sev = [], Counter()
    for name, frames, meta, rows, zones in clips:
        rows = [r for r in rows if r["kind"] in es.FALLS]
        if not rows:
            continue
        frames = [dict(f) for f in frames]
        es.hold(frames, meta, hold, base["sample_fps"])
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
    print(f"1. STAYED DOWN ({title}, last frame held {hold:.0f} s), {n['falls']} falls")
    print(f"   FALL ALERT fired           {n['fall_alert']}/{n['falls']} = {n['fall_alert'] / n['falls']:.0%}"
          + (f", median {latency[len(latency) // 2]:.0f} s after the fall started (rule: detect + {no_rec:.0f} s)"
             if latency else ""))
    print(f"   'on the floor' fired       {n['floor']}/{n['falls']} = {n['floor'] / n['falls']:.0%}  severities {dict(floor_sev)}"
          f"  (info = already covered by the fall alert)")
    print(f"   family told by either      {n['either']}/{n['falls']} = {n['either'] / n['falls']:.0%}")


if __name__ == "__main__":
    main()
