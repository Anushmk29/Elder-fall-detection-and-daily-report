"""Score fall detection on your own recordings (staged falls + look-alikes), see learning/STAGED_TEST.md.

Usage: python scripts/eval_staged.py data/staged/session1 [--list]

The folder holds the videos and a labels.csv, one row per labelled event:
    video,camera,take,kind,start_s,end_s,note
    phoneA_take01.mp4,phoneA,1,fall_stay,12.0,14.5,forward onto mattress
kind: fall       fell and got up again within 30 s  -> expect "possible fall", then "recovered", no FALL ALERT
      fall_stay  fell and stayed down >= 40 s       -> expect a FALL ALERT about 30 s after the fall
      lookalike  sat/lay/bent/knelt on purpose       -> expect nothing (anything here is a false alarm)
      none       whole video, nothing to detect      -> start_s/end_s empty
      fall_any   a fall from a public dataset, unknown if they stayed down -> only "seen or missed" is scored
take: recordings of the same moment from different phones share a take number ("caught by any camera").
Optional <folder>/<camera>.yaml = that camera's rules (its sofa/bed rest zones), like a live camera's rules file.
YOLO output is cached in data/staged_cache/, so re-scoring after a rule change takes seconds.
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "scripts"))
import eval_mcfd as ev  # noqa: E402
from app import perception, pipeline  # noqa: E402

FALLS = {"fall", "fall_stay", "fall_any"}
KINDS = FALLS | {"lookalike", "none"}


def read_labels(path):
    rows = defaultdict(list)
    with open(path, newline="", encoding="utf-8-sig") as f:
        for n, r in enumerate(csv.DictReader(f), start=2):
            if r["kind"] not in KINDS:
                sys.exit(f"{path} line {n}: kind must be one of {sorted(KINDS)}, got {r['kind']!r}")
            if r["kind"] != "none" and not (r["start_s"] and r["end_s"]):
                sys.exit(f"{path} line {n}: {r['kind']} needs start_s and end_s")
            r["start_s"] = float(r["start_s"]) if r["start_s"] else None
            r["end_s"] = float(r["end_s"]) if r["end_s"] else None
            rows[r["video"]].append(r)
    return rows


def score(found, rows, duration, no_rec, pad=ev.WINDOW_PAD):
    """found: fall events from events.analyze; rows: this video's labels.
    Returns {"hits": {row index: event}, "false": [(event, cause)]}. A detection inside a labelled fall window
    (widened by pad) is that fall's hit; any other detection is a false alarm, blamed on the look-alike it
    overlaps if there is one."""
    near = lambda e, r: r["start_s"] - pad[0] <= e["t"] <= r["end_s"] + pad[1]  # noqa: E731
    hits, false = {}, []
    for e in found:
        e["alerts"] = (e["got_up_s"] or duration - e["t"]) >= no_rec  # what live.py would push after no_rec s
        k = next((k for k, r in enumerate(rows) if r["kind"] in FALLS and near(e, r)), None)
        if k is None:
            cause = next((r["note"] or "lookalike" for r in rows if r["kind"] == "lookalike" and near(e, r)), "other")
            false.append((e, cause))
        elif k not in hits:
            hits[k] = e
    return {"hits": hits, "false": false}


def hold(frames, meta, secs, fps):
    """The person stays exactly as they were when the recording stopped (standing stays standing)."""
    if not secs or not frames:
        return
    last = frames[-1]
    for k in range(1, int(secs * fps) + 1):
        frames.append({"i": len(frames), "t": round(last["t"] + k / fps, 3), "people": [dict(p) for p in last["people"]]})
    meta["duration"] += secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="a session folder with videos + labels.csv")
    ap.add_argument("--list", action="store_true", help="print every video, not just problems")
    ap.add_argument("--hold", type=float, default=0, help="repeat each clip's last frame for this many seconds: "
                    "for datasets cut a few seconds after the fall (Le2i), which the stay-down rules never see")
    a = ap.parse_args()
    folder = Path(a.folder)
    labels = read_labels(folder / "labels.csv")
    unlabelled = sorted(p.name for p in folder.iterdir()
                        if p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv") and p.name not in labels)
    if unlabelled:
        print("not in labels.csv, skipped:", ", ".join(unlabelled))

    base = pipeline.load_rules()
    no_rec = base.get("live", {}).get("no_recovery_s", 30)
    seconds = 0.0
    n = defaultdict(int)  # counters for the summary
    takes, latencies, causes = {}, [], defaultdict(int)
    for name, rows in sorted(labels.items()):
        video = folder / name
        if not video.exists():
            sys.exit(f"labels.csv names {name}, which is not in {folder}")
        cam_rules = folder / f"{rows[0]['camera']}.yaml"
        rules = pipeline.load_rules(cam_rules if cam_rules.exists() else None)
        cache = (pipeline.DATA_DIR / "staged_cache" / f"{Path(perception.MODEL_PATH).stem}-c{rules.get('det_conf', 0.25)}"
                 / f"{folder.resolve().parent.name}_{folder.name}")  # Le2i rooms all name their folder "Videos"
        frames, meta = ev.load_frames(video, rules, cache)
        meta["clock_start"] = 0
        seconds += meta["duration"]
        hold(frames, meta, a.hold, rules["sample_fps"])
        res = score(ev.detect(frames, meta, rules), rows, meta["duration"], no_rec)

        notes = []
        for k, r in enumerate(rows):
            if r["kind"] not in FALLS:
                continue
            e = res["hits"].get(k)
            n[r["kind"]] += 1
            n[r["kind"] + "_hit"] += e is not None
            key = r["take"] or (name, r["start_s"])  # one fall per take, so a take = one fall seen from N phones
            takes[key] = takes.get(key, False) or e is not None
            if e is None:
                notes.append(f"MISSED {r['kind']} at {r['start_s']:.0f}s ({r['note']})")
            elif r["kind"] == "fall_stay":
                n["alert_hit"] += e["alerts"]
                if e["alerts"]:
                    latencies.append(e["t"] + no_rec - r["start_s"])
                else:
                    notes.append(f"seen at {e['t']:.0f}s but NO ALERT (looked like they got up after {e['got_up_s']}s)")
            elif r["kind"] == "fall" and e["alerts"]:
                n["alert_wrong"] += 1
                notes.append(f"ALERTED at {r['start_s']:.0f}s although they got up")
        for e, cause in res["false"]:
            n["fa"] += 1
            n["fa_alert"] += e["alerts"]
            causes[cause] += 1
            notes.append(f"FALSE {'ALERT' if e['alerts'] else 'possible fall'} at {e['t']:.0f}s "
                         f"(rule {e.get('how', '?')}, during: {cause})")
        if a.list or notes:
            print(f"{name}: " + ("; ".join(notes) or "ok"))

    hours = max(seconds / 3600, 1e-9)
    falls = n["fall"] + n["fall_stay"] + n["fall_any"]
    hit = n["fall_hit"] + n["fall_stay_hit"] + n["fall_any_hit"]
    print(f"\nvideos {len(labels)}, {seconds / 60:.1f} min of video")
    print(f"falls seen (possible fall)   {hit}/{falls} = {hit / max(falls, 1):.0%}")
    print(f"  per take, any camera       {sum(takes.values())}/{len(takes)}")
    print(f"FALL ALERT on stay-down falls {n['alert_hit']}/{n['fall_stay']}"
          + (f", median {sorted(latencies)[len(latencies) // 2]:.0f} s after the fall" if latencies else ""))
    print(f"wrong ALERT on got-up falls  {n['alert_wrong']}/{n['fall']}")
    print(f"false possible falls         {n['fa']} ({n['fa'] / hours:.1f}/h)")
    print(f"false FALL ALERTs            {n['fa_alert']} ({n['fa_alert'] / hours:.1f}/h)")
    if causes:
        print("false alarms during:", dict(causes))


if __name__ == "__main__":
    main()
