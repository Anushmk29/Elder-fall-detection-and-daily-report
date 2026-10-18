"""Synthetic 13 s test clip: a person cut from ultralytics' bus.jpg walks across a drawn room, falls
(rotates to lying), stays down ~5 s and gets up; a second bus.jpg person stands by the sofa for 7 s.

Usage: python scripts/make_test_clip.py [out.mp4]
"""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import ultralytics
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from app.perception import MODEL_PATH  # noqa: E402

ASSETS = Path(ultralytics.__file__).parent / "assets"
W, H, FPS, SECONDS = 1280, 720, 25, 13


def people(name):
    """Full-body person crops from a bundled ultralytics image, largest first."""
    img = cv2.imread(str(ASSETS / name))
    r = YOLO(MODEL_PATH)(img, verbose=False)[0]
    boxes = sorted(r.boxes.xyxy.int().tolist(), key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    return [img[y1:y2, x1:x2] for x1, y1, x2, y2 in boxes if y2 - y1 > 1.8 * (x2 - x1)]


def scene():
    f = np.zeros((H, W, 3), np.uint8)
    f[:430] = (214, 222, 230)                                         # wall
    f[430:] = np.linspace(150, 110, H - 430)[:, None, None] * [1, 1.15, 1.35] / 1.35  # floor
    cv2.rectangle(f, (40, 300), (380, 470), (150, 150, 145), -1)       # kitchen counter
    cv2.rectangle(f, (40, 290), (380, 305), (90, 90, 90), -1)
    cv2.rectangle(f, (900, 380), (1240, 560), (120, 80, 60), -1)       # sofa
    cv2.rectangle(f, (900, 340), (1240, 400), (100, 65, 50), -1)
    cv2.line(f, (640, 0), (640, 430), (190, 198, 206), 3)              # room divider
    return f


def paste(frame, crop, foot, height, angle=0.0):
    """Scale crop to `height`, rotate it by `angle` deg (CCW) about its bottom-centre, place that point at `foot`."""
    h, w = crop.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h), angle, height / h)
    M[:, 2] += np.array(foot, float) - (w / 2, h)
    warped = cv2.warpAffine(crop, M, (W, H))
    mask = cv2.warpAffine(np.full((h, w), 255, np.uint8), M, (W, H)) > 127
    frame[mask] = warped[mask]


def ease(a, b, t0, t1, t):
    x = min(max((t - t0) / (t1 - t0), 0), 1)
    return a + (b - a) * x * x * (3 - 2 * x)


def main(out):
    (resident, visitor, *_), bg = people("bus.jpg"), scene()
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    for n in range(SECONDS * FPS):
        t, f = n / FPS, bg.copy()
        if t < 7:
            paste(f, visitor, (1070 + 6 * np.sin(t * 2), 640), 360)
        x = ease(150, 800, 0, 3.5, t)                       # walk
        angle = ease(0, 88, 4.5, 5.3, t) if t < 10 else ease(88, 0, 10, 11.5, t)  # fall ... get up
        paste(f, resident, (x, 620), 400, angle)
        ff.stdin.write(f.tobytes())
    ff.stdin.close()
    ff.wait()
    print(out)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else BASE / "videos" / "synthetic_fall.mp4"))
