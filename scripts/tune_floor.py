"""Tune floor_low_ratio, the posture-free "on the floor" state (falls that end sitting or slumped, GUIDE §7 run 13).

Usage: python scripts/tune_floor.py "C:/.../archive (4)" D:/datasets/le2i [0,0.4,0.5,0.55,0.6]
tune = MCFD chute01-16 + Le2i Coffee rooms;  test = MCFD chute17-24 + Le2i Home rooms (reported once).
- caught: Le2i falls, last frame held 70 s (never gets up): an "on the floor" run >= 60 s starts after the fall.
- false: raw clips (MCFD + Le2i), runs of the floor state >= PROXY_S outside every labelled fall (2 s before to
  60 s after). Dataset clips are too short for a 60 s run, so a 10 s run is the proxy for "would have alerted".
score = recall - 0.5 * false_per_hour / 10 (the same trade-off as every other tuning run).
"""
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "scripts"))
import eval_live as el  # noqa: E402
import eval_staged as es  # noqa: E402
from app import events, pipeline, tracks  # noqa: E402

PROXY_S, HOLD_S, FLOOR_S = 10, 70, 60


def floor_runs(frames, meta, rules, min_s):
    found = events.analyze(tracks.build(frames, meta, rules), meta, {**rules, "lying_outside_rest_s": min_s})["events"]
    return [e for e in found if e["type"] == "lying_outside_rest"]


def load(mcfd, le2i, base):
    data = {"tune": [], "test": []}
    for name, frames, meta, rows, _ in el.mcfd_clips(mcfd, base):
        data["tune" if int(name[5:7]) <= 16 else "test"].append((name, frames, meta, rows, []))
    for name, frames, meta, rows, zf in el.le2i_clips(le2i, base):
        zones = pipeline.load_rules(zf)["zones"] if zf.exists() else []
        data["tune" if name.startswith("Coffee") else "test"].append((name, frames, meta, rows, zones))
    return data


def measure(clips, rules, show=False):
    caught = falls = false = 0
    secs = 0.0
    for name, frames, meta, rows, zones in clips:
        r = {**rules, "zones": zones}
        secs += meta["duration"]
        for e in floor_runs(frames, meta, r, PROXY_S):
            if not any(x["start_s"] - 2 <= e["t"] <= x["end_s"] + 60 for x in rows):
                false += 1
                if show:
                    print(f"    false: {name} t={e['t']:.0f}s {e['t_end'] - e['t']:.0f}s {e['how']}")
        if rows and name[0] in "CH":  # Le2i: never gets up
            held, hmeta = [dict(f) for f in frames], dict(meta)
            es.hold(held, hmeta, HOLD_S, rules["sample_fps"])
            falls += 1
            caught += any(e["t"] >= rows[0]["start_s"] - 2 for e in floor_runs(held, hmeta, r, FLOOR_S))
    return caught, falls, false, false / (secs / 3600)


def main():
    grid = json.loads(sys.argv[3]) if len(sys.argv) > 3 else [0, 0.4, 0.5, 0.55, 0.6, 0.65]
    base = pipeline.load_rules()
    data = load(sys.argv[1], sys.argv[2], base)
    scores = {}
    for ratio in grid:
        c, n, f, fh = measure(data["tune"], {**base, "floor_low_ratio": ratio})
        score = c / n - 0.5 * fh / 10
        print(f"  floor_low_ratio {ratio}: on-the-floor alert {c}/{n} = {c / n:.0%}, false {f} = {fh:.1f}/h, "
              f"score {score:.3f}", flush=True)
        scores[ratio] = score
    best = min(grid, key=lambda r: (-scores[r], r))  # ties: the lower (more cautious) ratio
    print("best on TUNE:", best)
    for label, ratio in (("off", 0), ("best", best)):
        c, n, f, fh = measure(data["test"], {**base, "floor_low_ratio": ratio}, show=ratio != 0)
        print(f"TEST (MCFD chute17-24 + Le2i Home, never tuned on) {label} ({ratio}): {c}/{n} = {c / n:.0%}, "
              f"false {f} = {fh:.1f}/h")


if __name__ == "__main__":
    main()
