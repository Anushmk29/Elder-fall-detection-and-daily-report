"""Prepare the UR Fall Detection dataset (fenix.ur.edu.pl/~mkepski/ds/uf.html) for eval_staged.py / eval_live.py:
the RGB half of each cam0 video goes to <dir>/Videos/<name>.mp4, with labels.csv and each room's zones next to it.

Usage: python scripts/urfall_labels.py D:/datasets/urfall
<dir> holds the downloads: fall-NN-cam0.mp4 (30), adl-NN-cam0.mp4 (40), urfall-cam0-falls.csv, urfall-cam0-adls.csv.
The videos are depth|RGB side by side (640x240, 30 fps); we keep the right half. cam1 (ceiling) has no per-frame
labels and isn't synchronised with cam0, so it isn't used.
The csv labels every frame: -1 not lying, 0 falling (or lying down), 1 lying (1-based frame numbers).
- fall-NN  -> fall_any from the first 0 frame to the first 1 frame.
- adl-NN with lying frames -> lookalike over the 0/1 stretch: lying down on purpose (bed, sofa, or the floor in
  ADL 30-35). That's the look-alike data the rules and the GRU were short of.
- other adl -> none (walking, sitting, picking things up, crouching).
"""
import csv
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

ZONES = Path(__file__).resolve().parent.parent / "learning" / "urfall_zones"
FPS = 30.0


def room_of(name):
    """The 5 rooms, assigned by eye from a contact sheet of the videos."""
    n = int(name[-2:])
    if name.startswith("fall") or 31 <= n <= 35:
        return "ur_office"
    return ("ur_bedroom1" if n <= 11 else "ur_dark" if n <= 23 else "ur_empty" if n <= 29 else "ur_bedroom2")


def crop(src, dst):
    cap = cv2.VideoCapture(str(src))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w // 2, h))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.write(frame[:, w // 2:])
    out.release()


def main():
    root = Path(sys.argv[1])
    videos = root / "Videos"
    videos.mkdir(exist_ok=True)
    frames = defaultdict(dict)
    for f in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv"):
        for r in csv.reader(open(root / f, newline="")):
            frames[r[0]][int(r[1])] = int(r[2])
    rows = []
    for src in sorted(root.glob("*-cam0.mp4")):
        name = src.name[:-9]  # fall-01
        dst = videos / f"{name}.mp4"
        if not dst.exists():
            crop(src, dst)
        lab = frames[name]
        down = sorted(k for k, v in lab.items() if v >= 0)
        t = lambda k: round((k - 1) / FPS, 2)  # noqa: E731
        if name.startswith("fall"):
            lying = [k for k in down if lab[k] == 1]
            rows.append([dst.name, room_of(name), "", "fall_any", t(down[0]), t(lying[0] if lying else down[-1]), ""])
        elif down:
            rows.append([dst.name, room_of(name), "", "lookalike", t(down[0]), t(down[-1]), "lying down on purpose"])
        else:
            rows.append([dst.name, room_of(name), "", "none", "", "", ""])
    with open(videos / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "camera", "take", "kind", "start_s", "end_s", "note"])
        w.writerows(rows)
    for room in {r[1] for r in rows}:
        shutil.copy(ZONES / f"{room}.yaml", videos / f"{room}.yaml")
    print(f"{videos}: {dict(Counter(r[3] for r in rows))}, rooms {sorted({r[1] for r in rows})}")


if __name__ == "__main__":
    main()
