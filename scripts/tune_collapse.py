"""Tune the posture-free "collapse" fall rule on two datasets at once, then report the winner once on held-out data.

Usage: python scripts/tune_collapse.py "C:/.../archive (4)" D:/datasets/le2i ['{"settle_s": [0, 0.5, 1.0]}']
The optional JSON grid replaces GRID, so any fall setting can be tuned the same way.
tune = MCFD chute01-16 + Le2i Coffee rooms;   test = MCFD chute17-24 + Le2i Home rooms (never looked at while tuning)
Le2i clips use each camera view's rest zones (learning/le2i_zones, --no-zones to skip); MCFD has none.
Le2i clips stop ~4 s after the fall, so their last frame is held for 10 s (eval_staged.py --hold).
score = recall - 0.5 * false_alarms_per_hour / 10, over all falls and hours of both datasets pooled.
"""
import itertools
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "scripts"))
import eval_live as el  # noqa: E402
import eval_mcfd as ev  # noqa: E402
import eval_staged as es  # noqa: E402
from app import pipeline  # noqa: E402

GRID = {"collapse_frac": [0, 0.35, 0.4, 0.45, 0.5, 0.55], "collapse_window_s": [1.0, 1.5]}
TUNE_ROOMS, TEST_ROOMS = ("Coffee_room_01", "Coffee_room_02"), ("Home_01", "Home_02")


def load(mcfd_root, le2i_root, base, zones_on=True):
    labels = ev.fall_windows(Path(mcfd_root) / "data_tuple3.csv")
    mcfd = dict(base, source_fps=ev.FPS)
    data = {"tune": [], "test": []}
    for v in sorted((Path(mcfd_root) / "dataset" / "dataset").glob("chute*/cam*.avi")):
        chute, cam = int(v.parent.name[5:]), int(v.stem[3:])
        frames, meta = ev.load_frames(v, mcfd)
        rows = [{"kind": "fall", "start_s": a, "end_s": b, "note": ""} for a, b in labels.get((chute, cam), [])]
        data["tune" if chute <= ev.TUNE_LAST else "test"].append((frames, meta, rows, []))  # no zones: furniture moves
    for room in TUNE_ROOMS + TEST_ROOMS:
        for _, frames, meta, rows, zf in el.staged_clips(Path(le2i_root) / room / room / "Videos", base):
            es.hold(frames, meta, 10, base["sample_fps"])
            zones = pipeline.load_rules(zf)["zones"] if zones_on else []
            data["tune" if room in TUNE_ROOMS else "test"].append((frames, meta, rows, zones))
    for clips in data.values():
        for _, meta, _, _ in clips:
            meta["clock_start"] = 0
    return data


def measure(clips, rules):
    hits = falls = fa = 0
    secs = 0.0
    for frames, meta, rows, zones in clips:
        r = es.score(ev.detect(frames, meta, {**rules, "zones": zones}), rows, meta["duration"], 30)
        falls += len(rows)
        hits += len(r["hits"])
        fa += len(r["false"])
        secs += meta["duration"]
    return hits, falls, fa, fa / (secs / 3600)


def main():
    grid = json.loads(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].startswith("{") else GRID
    base = pipeline.load_rules()
    base["zones"] = []
    data = load(sys.argv[1], sys.argv[2], base, zones_on="--no-zones" not in sys.argv)
    results = []
    for values in itertools.product(*grid.values()):
        cfg = dict(zip(grid, values))
        if cfg.get("collapse_frac") == 0 and cfg.get("collapse_window_s", 0) != grid.get("collapse_window_s", [0])[0]:
            continue  # rule off: the window doesn't matter
        hits, falls, fa, fa_h = measure(data["tune"], {**base, "fall": {**base["fall"], **cfg}})
        results.append((hits / falls - 0.5 * fa_h / 10, hits, falls, fa, fa_h, cfg))
        print(f"  {cfg}: recall {hits}/{falls} = {hits / falls:.0%}, false alarms {fa} = {fa_h:.1f}/h", flush=True)
    results.sort(key=lambda r: -r[0])
    best = results[0][5]
    print("best on TUNE:", best)
    for name, cfg in (("current rules.yaml", {}), ("best", best)):
        hits, falls, fa, fa_h = measure(data["test"], {**base, "fall": {**base["fall"], **cfg}})
        print(f"TEST (MCFD chute17-24 + Le2i Home, never tuned on) {name}: recall {hits}/{falls} = {hits / falls:.0%}, "
              f"false alarms {fa} = {fa_h:.1f}/h")


if __name__ == "__main__":
    main()
