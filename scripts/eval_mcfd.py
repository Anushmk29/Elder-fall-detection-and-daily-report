"""Score fall detection on the Multiple Cameras Fall Dataset (Univ. Montreal; chute01-24 x cam1-8).

Usage: python scripts/eval_mcfd.py "C:/.../archive (4)" [--rules cam.yaml] [--only chute20] [--list]

The folder must hold data_tuple3.csv (frame windows labelled 1 = fall) and dataset/dataset/chuteNN/camN.avi.
YOLO output is cached per clip in data/mcfd_cache/, and posture is recomputed from the cached keypoints on load,
so changes to posture/fall/track rules re-score all 192 clips in seconds. Delete the cache after changing the
model, imgsz or source_fps.

A detected fall is a hit if it starts inside a labelled fall window (widened by WINDOW_PAD); anything else is a
false alarm. The files claim 120 fps but are 30 fps (frame counts match the dataset's 30 s-4 min scenarios).
"""
import argparse
import csv
import gzip
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from app import events, perception, pipeline, tracks  # noqa: E402
from app.posture import classify  # noqa: E402

FPS = 30.0
WINDOW_PAD = (2.0, 4.0)  # seconds before / after a labelled window that still count as that fall
CACHE_ROOT = pipeline.DATA_DIR / "mcfd_cache"  # one folder per pose model + detection confidence


def fall_windows(csv_path):
    """{(chute, cam): [(start_s, end_s)]}, adjacent label-1 windows merged into one fall."""
    raw = defaultdict(list)
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if float(r["label"]) == 1:
                raw[(int(float(r["chute"])), int(float(r["cam"])))].append((float(r["start"]), float(r["end"])))
    out = {}
    for key, wins in raw.items():
        merged = []
        for a, b in sorted(wins):
            if merged and a - merged[-1][1] <= 1:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        out[key] = [(a / FPS, b / FPS) for a, b in merged]
    return out


def cache_dir(rules):
    return CACHE_ROOT / f"{Path(perception.MODEL_PATH).stem}-c{rules.get('det_conf', 0.25)}"


def load_frames(video, rules):
    cache = cache_dir(rules) / f"{video.parent.name}_{video.stem}.json.gz"
    if cache.exists():
        data = json.loads(gzip.decompress(cache.read_bytes()))
    else:
        t0 = time.time()
        frames, meta = perception.analyze(video, rules)
        data = {"frames": frames, "meta": meta}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(gzip.compress(json.dumps(data).encode()))
        print(f"  {video.parent.name}/{video.name}: {meta['duration']:.0f} s of video in {time.time() - t0:.0f} s")
    for f in data["frames"]:
        for p in f["people"]:
            p["posture"] = classify(p["box"], p["kp"], rules.get("posture"))
    return data["frames"], data["meta"]


def detect(frames, meta, rules):
    trk = tracks.build(frames, meta, rules)
    return [e for e in events.analyze(trk, meta, rules)["events"] if e["type"] == "fall"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="folder with data_tuple3.csv and dataset/dataset/chuteNN/")
    ap.add_argument("--rules", help="yaml overriding rules.yaml")
    ap.add_argument("--only", help="e.g. chute20")
    ap.add_argument("--list", action="store_true", help="print every clip, not just misses and false alarms")
    ap.add_argument("--cached", action="store_true", help="only clips already in the cache (no YOLO)")
    a = ap.parse_args()

    root = Path(a.root)
    labels = fall_windows(root / "data_tuple3.csv")
    rules = pipeline.load_rules(a.rules)
    rules["zones"], rules["source_fps"] = [], FPS
    meta_extra = {"clock_start": 0}

    falls_total = hits = false_alarms = 0
    seconds = 0.0
    delays, rows = [], []
    for video in sorted((root / "dataset" / "dataset").glob("chute*/cam*.avi")):
        if a.only and video.parent.name != a.only:
            continue
        chute, cam = int(video.parent.name[5:]), int(video.stem[3:])
        if a.cached and not (cache_dir(rules) / f"{video.parent.name}_{video.stem}.json.gz").exists():
            continue
        frames, meta = load_frames(video, rules)
        meta.update(meta_extra)
        seconds += meta["duration"]
        found = detect(frames, meta, rules)
        wins = labels.get((chute, cam), [])
        matched = set()
        fa = []
        for e in found:
            k = next((n for n, (s, t) in enumerate(wins) if s - WINDOW_PAD[0] <= e["t"] <= t + WINDOW_PAD[1]), None)
            if k is None:
                fa.append(e["t"])
            elif k not in matched:
                matched.add(k)
                delays.append(e["t"] - wins[k][0])
        falls_total += len(wins)
        hits += len(matched)
        false_alarms += len(fa)
        missed = [w for n, w in enumerate(wins) if n not in matched]
        line = (f"{video.parent.name}/cam{cam}: falls {len(matched)}/{len(wins)}"
                + (f"  missed at {', '.join(f'{s:.0f}s' for s, _ in missed)}" if missed else "")
                + (f"  FALSE ALARM at {', '.join(f'{t:.0f}s' for t in fa)}" if fa else ""))
        rows.append(line)
        if a.list or missed or fa:
            print(line)

    print(f"\nclips {len(rows)}, video {seconds / 3600:.2f} h")
    print(f"recall      {hits}/{falls_total} = {hits / max(falls_total, 1):.0%}")
    print(f"false alarms {false_alarms} ({false_alarms / max(seconds / 3600, 1e-9):.1f} per hour of video)")
    if delays:
        delays.sort()
        print(f"detected at {delays[len(delays) // 2]:+.1f} s (median) from the labelled fall start")


if __name__ == "__main__":
    main()
