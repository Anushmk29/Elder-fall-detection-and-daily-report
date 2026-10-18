"""Dump one row per fall *candidate* on the MCFD tune split with its measurements and whether it was a real fall,
so thresholds can be chosen from data instead of guessed (see learning/GUIDE.md, "Tuning").

Usage: python scripts/fall_features.py "C:/.../archive (4)" [--split tune|test] [--out data/fall_features.csv]
"""
import argparse
import csv
import statistics as st
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "scripts"))
import eval_mcfd as ev  # noqa: E402
from app import events, pipeline, tracks  # noqa: E402
from app.posture import features  # noqa: E402


def after(samples, t, secs=3.0):
    return [s for s in samples if t <= s["t"] <= t + secs]


def row_for(f, S):
    down = after(S, f["t"])
    feats = [features(s["box"], s["kp"]) for s in down]
    torso = [x["torso"] for x in feats if x["torso"] is not None]
    return {"how": f["how"], "drop_s": f.get("drop_s"), "h_ratio": f.get("h_ratio"), "hip_drop": f.get("hip_drop"),
            "down_s": round(f["t_end"] - f["t"], 1), "got_up": f["got_up_s"] is not None,
            "aspect_down": round(st.median(x["aspect"] for x in feats), 2) if feats else "",
            "torso_down": round(st.median(torso)) if torso else "",
            "lying_frac": round(sum(s["posture"] == "lying" for s in down) / len(down), 2) if down else ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--split", default="tune", choices=["tune", "test", "all"])
    ap.add_argument("--out", default=str(pipeline.DATA_DIR / "fall_features.csv"))
    a = ap.parse_args()
    root = Path(a.root)
    labels = ev.fall_windows(root / "data_tuple3.csv")
    rules = pipeline.load_rules()
    rules["zones"], rules["source_fps"] = [], ev.FPS
    rows = []
    for v in sorted((root / "dataset" / "dataset").glob("chute*/cam*.avi")):
        chute, cam = int(v.parent.name[5:]), int(v.stem[3:])
        if (a.split == "tune" and chute > ev.TUNE_LAST) or (a.split == "test" and chute <= ev.TUNE_LAST):
            continue
        frames, meta = ev.load_frames(v, rules)
        meta["clock_start"] = 0
        trk = tracks.build(frames, meta, rules)
        events.analyze(trk, meta, rules)  # adds zones to samples
        wins = labels.get((chute, cam), [])
        for tr in trk:
            S = tr["samples"]
            cands = events.detect_falls(S, rules["fall"])
            vz = events.detect_vanish(S, [o for o in trk if o is not tr], meta, rules["fall"])
            cands += [vz] if vz else []
            for f in cands:
                real = any(s - ev.WINDOW_PAD[0] <= f["t"] <= t + ev.WINDOW_PAD[1] for s, t in wins)
                rows.append({"clip": f"chute{chute:02d}/cam{cam}", "t": round(f["t"], 1), "real_fall": int(real),
                             **row_for(f, S)})
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} candidates ({sum(r['real_fall'] for r in rows)} real) -> {a.out}")


if __name__ == "__main__":
    main()
