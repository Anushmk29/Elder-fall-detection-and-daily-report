"""Save the first frame of a video with a 10% grid, to read zone polygon coordinates for rules.yaml.

Usage: python scripts/zone_helper.py videos/kitchen.mp4 [out.jpg]
"""
import sys
from pathlib import Path

import cv2

src = sys.argv[1]  # a file, a webcam index (0) or a stream URL from cameras.yaml
video = Path(src) if "://" not in src and not src.isdigit() else Path("camera")
out = Path(sys.argv[2]) if len(sys.argv) > 2 else video.with_suffix(".grid.jpg")
ok, frame = cv2.VideoCapture(int(src) if src.isdigit() else src).read()
if not ok:
    sys.exit(f"Could not read a frame from {video}")
h, w = frame.shape[:2]
for k in range(1, 10):
    x, y = int(w * k / 10), int(h * k / 10)
    cv2.line(frame, (x, 0), (x, h), (0, 255, 255), 1)
    cv2.line(frame, (0, y), (w, y), (0, 255, 255), 1)
    cv2.putText(frame, f"{k / 10:.1f}", (x + 3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{k / 10:.1f}", (3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
cv2.imwrite(str(out), frame)
print(f"{out}  ({w}x{h}). Zone points are [x, y] fractions, e.g. [[0.1, 0.5], [0.4, 0.5], [0.4, 0.9], [0.1, 0.9]]")
