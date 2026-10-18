"""Run the pipeline on every video in a folder and score fall detection against your labels.

Usage: python scripts/batch_eval.py D:/test-videos [--labels labels.csv] [--rules cam.yaml] [--out data/batch]

labels.csv (optional): two columns, `file,fall`, with fall = 1 if the clip really contains a fall, else 0.
Writes <out>/<video-name>/ (annotated.mp4 + result.json, open the mp4 to eyeball it) and <out>/summary.csv.
Already-processed videos are skipped, so you can stop and re-run. Delete a folder to redo it.
No zones are used unless --rules gives some (the default rules.yaml zones only fit the synthetic scene).
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from app import pipeline  # noqa: E402

EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}


def verdict(expected, detected):
    if expected is None:
        return ""
    return {(1, True): "hit", (1, False): "MISSED", (0, True): "FALSE ALARM", (0, False): "ok"}[(expected, detected)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--labels")
    ap.add_argument("--rules", help="per-camera yaml (zones, thresholds); overrides rules.yaml")
    ap.add_argument("--out", default=str(BASE / "data" / "batch"))
    a = ap.parse_args()

    labels = {}
    if a.labels:
        with open(a.labels, newline="", encoding="utf-8-sig") as f:
            labels = {r["file"].strip(): int(r["fall"]) for r in csv.DictReader(f)}
    rules = pipeline.load_rules(a.rules)
    if not a.rules:
        rules["zones"] = []
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # lets the /batch page list clips not processed yet and sibling folders not run yet
    (out / "batch.json").write_text(json.dumps({"source": str(Path(a.folder).resolve()), "labels": a.labels,
                                                "rules": a.rules}, indent=1), encoding="utf-8")
    fields = ["file", "duration_s", "people", "falls", "fall_at_s", "other_events", "expected_fall", "verdict",
              "secs", "video", "error"]

    def write_summary():  # after every clip, so progress shows while the batch runs
        with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fields, restval="")
            w.writeheader()
            w.writerows(rows)

    rows = []
    for video in sorted(p for p in Path(a.folder).iterdir() if p.suffix.lower() in EXTS):
        res_dir = out / video.stem
        res_file = res_dir / "result.json"
        t0 = time.time()
        try:
            if res_file.exists():
                result = json.loads(res_file.read_text(encoding="utf-8"))
            else:
                print(f"{video.name}: ", end="", flush=True)
                result = pipeline.run(video, res_dir, lambda p, s: print(f"\r{video.name}: {p:5.1f}% {s:<30}",
                                                                          end="", flush=True),
                                      rules=rules, title=video.stem)
                print()
        except Exception as e:  # one bad file must not stop the batch
            print(f"\n{video.name}: FAILED {e}")
            rows.append({"file": video.name, "error": str(e)})
            write_summary()
            continue
        ev = result["events"]
        falls = [e for e in ev if e["type"] == "fall"]
        expected = labels.get(video.name)
        rows.append({"file": video.name, "duration_s": round(result["duration"], 1), "people": len(result["people"]),
                     "falls": len(falls), "fall_at_s": " ".join(f"{e['t']:.1f}" for e in falls),
                     "other_events": " ".join(e["type"] for e in ev if e["type"] != "fall"),
                     "expected_fall": "" if expected is None else expected,
                     "verdict": verdict(expected, bool(falls)), "secs": round(time.time() - t0),
                     "video": str(res_dir / "annotated.mp4"), "error": ""})
        write_summary()

    v = [r.get("verdict") for r in rows]
    print(f"\n{len(rows)} videos -> {out / 'summary.csv'}")
    if labels:
        hit, miss, fa, ok = (v.count(k) for k in ("hit", "MISSED", "FALSE ALARM", "ok"))
        print(f"falls caught {hit}/{hit + miss}, missed {miss}, false alarms {fa}/{fa + ok} no-fall clips")


if __name__ == "__main__":
    main()
