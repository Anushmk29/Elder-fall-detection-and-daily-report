"""Write labels.csv (the eval_staged.py format) next to the videos of each Le2i fall-dataset room.

Usage: python scripts/le2i_labels.py D:/datasets/le2i
Le2i annotation files start with two lines, the first and last frame of the fall (0 and 0 = no fall), then one
bounding box per frame. A few files have no such header; they are treated as no-fall videos. Frames are converted
to seconds with each video's own fps (Home_01 is 24, the rest 25).
"""
import csv
import sys
from pathlib import Path

import cv2

for videos in sorted(Path(sys.argv[1]).glob("*/*/Videos")):
    ann = next(videos.parent.glob("Annotation*_files"))
    rows = []
    for v in sorted(videos.glob("*.avi"), key=lambda p: int(p.stem.split("(")[1][:-1])):
        a = ann / f"{v.stem}.txt"
        lines = a.read_text().split() if a.exists() else []
        if len(lines) >= 2 and "," not in lines[0] and int(lines[0]):
            fps = cv2.VideoCapture(str(v)).get(cv2.CAP_PROP_FPS) or 25.0
            start, end = int(lines[0]) / fps, int(lines[1]) / fps
            rows.append([v.name, videos.parent.name, "", "fall_any", round(start, 2), round(end, 2), ""])
        else:
            rows.append([v.name, videos.parent.name, "", "none", "", "", "" if a.exists() else "no annotation file"])
    with open(videos / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "camera", "take", "kind", "start_s", "end_s", "note"])
        w.writerows(rows)
    print(f"{videos}: {sum(r[3] == 'fall_any' for r in rows)} falls, {sum(r[3] == 'none' for r in rows)} without")
