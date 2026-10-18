"""Grid-search fall thresholds on the MCFD *tune* split (chute01-16), then report the best setting once on the
held-out *test* split (chute17-24). This is the "training" step of a rules-based detector: the pose model is fixed,
only a few readable thresholds are fitted to labelled data. See learning/GUIDE.md.

Usage: python scripts/tune_falls.py "C:/.../archive (4)" [--fa-weight 0.5]
score = recall - fa_weight * false_alarms_per_hour / 10   (one false alarm per 6 min of video costs as much as 5% recall)
"""
import argparse
import itertools
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "scripts"))
import eval_mcfd as ev  # noqa: E402
from app import pipeline  # noqa: E402

GRID = {"min_down_s": [2.0, 3.0, 4.0, 5.0], "vanish_min_aspect": [0, 0.9, 1.0, 1.2],
        "floor_height_ratio": [1.0, 0.8], "height_ratio": [0.6, 0.7]}


def score(clips, labels, rules):
    hits = total = fa = 0
    secs = 0.0
    for chute, cam, frames, meta in clips:
        wins = labels.get((chute, cam), [])
        total += len(wins)
        secs += meta["duration"]
        matched = set()
        for e in ev.detect(frames, meta, rules):
            k = next((n for n, (s, t) in enumerate(wins) if s - ev.WINDOW_PAD[0] <= e["t"] <= t + ev.WINDOW_PAD[1]), None)
            if k is None:
                fa += 1
            else:
                matched.add(k)
        hits += len(matched)
    return hits / max(total, 1), fa / (secs / 3600), hits, total, fa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--fa-weight", type=float, default=0.5)
    a = ap.parse_args()
    root = Path(a.root)
    labels = ev.fall_windows(root / "data_tuple3.csv")
    base = pipeline.load_rules()
    base["zones"], base["source_fps"] = [], ev.FPS
    split = {"tune": [], "test": []}
    for v in sorted((root / "dataset" / "dataset").glob("chute*/cam*.avi")):
        chute, cam = int(v.parent.name[5:]), int(v.stem[3:])
        frames, meta = ev.load_frames(v, base)  # cached YOLO output; posture recomputed from keypoints
        meta["clock_start"] = 0
        split["tune" if chute <= ev.TUNE_LAST else "test"].append((chute, cam, frames, meta))

    results = []
    for values in itertools.product(*GRID.values()):
        rules = {**base, "fall": {**base["fall"], **dict(zip(GRID, values))}}
        rec, fa_h, *_ = score(split["tune"], labels, rules)
        results.append((rec - a.fa_weight * fa_h / 10, rec, fa_h, dict(zip(GRID, values))))
    results.sort(key=lambda r: -r[0])
    print("best on TUNE split (chute01-16):")
    for s, rec, fa_h, cfg in results[:8]:
        print(f"  score {s:+.3f}  recall {rec:.0%}  false alarms {fa_h:5.1f}/h  {cfg}")
    best = results[0][3]
    for name, cfg in (("current rules.yaml", {}), ("best", best)):
        rules = {**base, "fall": {**base["fall"], **cfg}}
        rec, fa_h, hits, total, fa = score(split["test"], labels, rules)
        print(f"TEST split (chute17-24, never tuned on) with {name}: recall {hits}/{total} = {rec:.0%}, "
              f"false alarms {fa} = {fa_h:.1f}/h")


if __name__ == "__main__":
    main()
