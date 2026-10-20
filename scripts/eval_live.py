"""Replay labelled clips through the real live engine (live.Camera, rolling window, 1 s evaluations) as fast as possible,
and measure what the *family* would receive. Uses the cached YOLO output, so it doesn't need the GPU.

Usage: python scripts/eval_live.py "C:/.../archive (4)" D:/datasets/le2i [--extra D:/datasets/urfall/Videos D:/datasets/cauca/Videos] [--hold 70]

1. Stayed down (Le2i, last frame held --hold s, i.e. the person never gets up): does the FALL ALERT fire, how long
   after the labelled fall start, and does the separate "on the floor" alert fire as a safety net?
   Held frames are frozen, so this can't tell "moving" from "not moving"; part 3 measures that.
2. False pushes per hour (raw clips, MCFD + Le2i): every pushed event (alert/warn) that isn't within a labelled fall
   (2 s before its start to 60 s after its end).
3. Motion of really fallen people (labelled fall end + 4 s): what floor_moving_bl_s has to stay above so a fallen
   person isn't called "moving". With --extra, also people on the floor *on purpose* (look-alike rows, e.g. UR Fall ADL
   outside bed/sofa): what it has to stay below so they are.

--vlm: every pushed fall (FALL ALERT, fall-got-up) in 1 and 2 is also shown to the vision model as live does it (frames
~1 s before the fall, 2 s after, and at the push; Camera._ai_check, here unannotated, from the video). Prints how many
real falls the AI would keep and how many false pushes it would stop (only a clear NO stops one). Costs ~$0.0005 a
check, cached in data/llm.
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
import cv2  # noqa: E402

from app import events, live, llm, perception, pipeline, tracks  # noqa: E402

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
        meta = dict(meta, path=str(folder / name))
        yield f"{prefix}{name}", frames, meta, [r for r in rows if r["kind"] != "none"], folder / f"{rows[0]['camera']}.yaml"


def le2i_clips(root, base):
    for room in ("Coffee_room_01", "Coffee_room_02", "Home_01", "Home_02"):
        yield from staged_clips(Path(root) / room / room / "Videos", base, f"{room}/")


def extra_clips(folders, base):
    for f in folders:
        yield from staged_clips(Path(f), base, f"{Path(f).parent.name}/")


def mcfd_clips(root, base):
    labels = ev.fall_windows(Path(root) / "data_tuple3.csv")
    rules = dict(base, source_fps=ev.FPS)
    for v in sorted((Path(root) / "dataset" / "dataset").glob("chute*/cam*.avi")):
        chute, cam = int(v.parent.name[5:]), int(v.stem[3:])
        frames, meta = ev.load_frames(v, rules)
        meta = dict(meta, path=str(v))
        rows = [{"kind": "fall", "start_s": a, "end_s": b} for a, b in labels.get((chute, cam), [])]
        yield f"{v.parent.name}/{v.name}", frames, meta, rows, None


def ai_verdict(ai, meta, got, e):
    """The AI's verdict on a pushed fall event, from the video: 'FALLEN' | 'NO' | 'UNSURE' | None."""
    onset = max((x["t"] for x in got if x["type"] == "fall_suspected" and x["t"] <= e["t"]), default=e["t"] - 5)
    cap = cv2.VideoCapture(meta["path"])
    fps, last = cap.get(cv2.CAP_PROP_FPS) or 25, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1
    imgs = []
    for x in (onset - 1, onset + 2, e["t"]):  # past the end = the held last frame, as in part 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(max(int(x * fps), 0), last))
        ok, f = cap.read()
        if ok:
            imgs.append(cv2.resize(f, (640, round(f.shape[0] * 640 / f.shape[1]))))
    cap.release()
    if len(imgs) < 3:
        return None
    path = pipeline.DATA_DIR / "llm" / "eval_strip.jpg"
    cv2.imwrite(str(path), cv2.hconcat(imgs))
    return ai.fall_check(path)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mcfd")
    ap.add_argument("le2i")
    ap.add_argument("--extra", nargs="*", default=[], help="more labels.csv folders (urfall_labels.py, cauca_labels.py)")
    ap.add_argument("--hold", type=float, default=70)
    ap.add_argument("--vlm", action="store_true", help="also ask the vision model about each pushed fall (cloud, ~$0.0005 each)")
    a = ap.parse_args()
    ai = llm.LLM(pipeline.DATA_DIR / "llm") if a.vlm else None
    base = pipeline.load_rules()
    live_cfg = live.live_rules()
    no_rec = live_cfg["live"]["no_recovery_s"]

    # 1. stayed down
    for title, clips in [("Le2i", le2i_clips(a.le2i, base))] + [(Path(f).parent.name, staged_clips(Path(f), base))
                                                                for f in a.extra]:
        stayed_down(title, clips, a.hold, base, no_rec, ai)

    # 2. false pushes
    secs, false, by_type, ai_n = 0.0, 0, Counter(), Counter()
    for gen in (mcfd_clips(a.mcfd, base), le2i_clips(a.le2i, base), extra_clips(a.extra, base)):
        for name, frames, meta, rows, zones in gen:
            secs += meta["duration"]
            got = replay(frames, meta, zones if zones and zones.exists() else None)
            for e in got:
                if e["severity"] in PUSH and not any(r["start_s"] - 2 <= e["t"] <= r["end_s"] + 60
                                                     for r in rows if r["kind"] in es.FALLS):
                    false += 1
                    by_type[e["type"]] += 1
                    if ai and e["type"] in ("fall_alert", "fall_recovered"):
                        ai_n["asked"] += 1
                        ai_n["stopped"] += ai_verdict(ai, meta, got, e) == "NO"
    print(f"\n2. FALSE PUSHES (raw clips, MCFD + Le2i{''.join(' + ' + Path(f).parent.name for f in a.extra)}, {secs / 3600:.1f} h of video): "
          f"{false} = {false / (secs / 3600):.1f}/h  by type {dict(by_type)}")
    if ai:
        print(f"   AI check stops             {ai_n['stopped']}/{ai_n['asked']} false fall pushes"
              f" -> {(false - ai_n['stopped']) / (secs / 3600):.1f}/h left; spent ${ai.spent_today():.2f} today")

    # 3. how much do people on the floor move? fallen (should be below the threshold) vs on purpose (above)
    fallen, purpose = [], []
    for gen in (mcfd_clips(a.mcfd, base), le2i_clips(a.le2i, base), extra_clips(a.extra, base)):
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
    for label, vals in (("fallen (first 4 s)", fallen), ("on the floor on purpose (look-alikes)", purpose)):
        if not vals:
            continue
        vals.sort()
        q = lambda p: vals[min(int(p * len(vals)), len(vals) - 1)]  # noqa: E731
        print(f"\n3. MOTION of people {label}, {len(vals)}: 10% {q(0.1):.3f}, median {q(0.5):.3f}, 90% {q(0.9):.3f} "
              f"body lengths/s; floor_moving_bl_s = {thr} calls {sum(v >= thr for v in vals)}/{len(vals)} 'moving'")


def stayed_down(title, clips, hold, base, no_rec, ai=None):
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
            if ai:
                n["ai_kept"] += ai_verdict(ai, meta, got, alert) != "NO"
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
    if ai and n["fall_alert"]:
        print(f"   AI check keeps             {n['ai_kept']}/{n['fall_alert']} = {n['ai_kept'] / n['fall_alert']:.0%}"
              f" of the FALL ALERTs (rule: keep the veto only if >= 95%)")


if __name__ == "__main__":
    main()
