"""Download CAUCAFall (Mendeley Data 7w7fccy7ky, CC BY 4.0) and write labels.csv for eval_staged.py / eval_live.py.

Usage: python scripts/cauca_labels.py D:/datasets/cauca
10 subjects x 10 activities, one .avi each, in a real home (night scenes, occlusions). Each frame also has a .png and
a YOLO-format .txt whose class is 1 = fall, 0 = no fall. Only the .avi files are downloaded (~200 MB, not the ~10 GB
of PNGs). The fall label is 0...0 then 1 to the end of the clip (checked on a whole clip), so the first fall frame is
found by binary search over the label files: ~8 small downloads per clip instead of ~250.
- Fall backwards/forward/left/right/sitting -> fall_any. "fall" marks the person *already down* (checked on a
  contact sheet: standing 1 s before the first fall frame, on the floor at it), so the fall is LEAD_S before it to
  LEAD_S after it. One clip (S9 Fall forward) has no fall frames: left out (as "none" a correct detection would count as false).
- Hop, Kneel, Pick up object, Sit down, Walk -> none: anything detected there is a false alarm.
Writes <dir>/Videos/S<n>_<activity>.avi, labels.csv, and an empty zones file per subject (cameras: S1..S10).
"""
import csv
import json
import sys
import urllib.request
from pathlib import Path

import cv2

API = "https://data.mendeley.com/public-api/datasets/7w7fccy7ky"
VERSION = 5
LEAD_S = 1.0


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})  # the default Python agent gets a 403
    for attempt in range(4):
        try:
            data = urllib.request.urlopen(req, timeout=60).read()
            if data.startswith(b'{"url"') and b'"error"' in data[:500]:  # a gateway error sent as 200 OK
                raise OSError(data[:200])
            return data
        except OSError:
            if attempt == 3:
                raise


def files(folder_id):
    return json.loads(get(f"{API}/files?folder_id={folder_id}&version={VERSION}"))


def first_fall(labels):
    """Index of the first label file with class 1 (binary search; assumes 0...0 1...1)."""
    cls = lambda k: get(labels[k]["content_details"]["download_url"]).split()[0] == b"1"  # noqa: E731
    lo, hi = 0, len(labels) - 1
    if not cls(hi):
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if cls(mid) else (mid + 1, hi)
    return lo


def main():
    videos = Path(sys.argv[1]) / "Videos"
    videos.mkdir(parents=True, exist_ok=True)
    folders = json.loads(get(f"{API}/folders/{VERSION}"))
    subject = {f["id"]: f["name"].split(".")[1] for f in folders if f["name"].startswith("Subject.")}
    rows = []
    for f in sorted((f for f in folders if f.get("parent_id") in subject),
                    key=lambda f: (int(subject[f["parent_id"]]), f["name"])):
        cam, act = f"S{subject[f['parent_id']]}", f["name"]
        listing = files(f["id"])
        avi = next(x for x in listing if x["filename"].lower().endswith(".avi"))
        dst = videos / f"{cam}_{act.replace(' ', '_')}.avi"
        if not dst.exists():
            dst.write_bytes(get(avi["content_details"]["download_url"]))
        fps = cv2.VideoCapture(str(dst)).get(cv2.CAP_PROP_FPS) or 30.0
        k = None
        if act.startswith("Fall"):
            labels = sorted((x for x in listing if x["filename"].endswith(".txt") and x["filename"] != "classes.txt"),
                            key=lambda x: x["filename"])  # cas900001.txt ... (the prefix differs per folder)
            k = first_fall(labels)
        if k is None and act.startswith("Fall"):
            print("no fall frames, left out:", avi["filename"])
            continue
        if k is None:
            rows.append([dst.name, cam, "", "none", "", "", avi["filename"]])
        else:
            rows.append([dst.name, cam, "", "fall_any", round(max(k / fps - LEAD_S, 0), 2), round(k / fps + LEAD_S, 2), avi["filename"]])
        print(rows[-1], flush=True)
    with open(videos / "labels.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["video", "camera", "take", "kind", "start_s", "end_s", "note"])
        w.writerows(rows)
    for cam in {r[1] for r in rows}:
        zones = videos / f"{cam}.yaml"
        if not zones.exists():
            zones.write_text("zones: []  # CAUCAFall: draw this subject's sofa/bed here (kind: rest)\n")
    print(f"{videos}: {sum(r[3] == 'fall_any' for r in rows)} falls, {sum(r[3] == 'none' for r in rows)} without")


if __name__ == "__main__":
    main()
