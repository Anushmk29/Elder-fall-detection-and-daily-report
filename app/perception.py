"""YOLO11n-pose + ByteTrack on CPU, sampled at a fixed rate. Coordinates stay in source pixels."""
import itertools
import math
import os
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

from .posture import classify

BASE = Path(__file__).resolve().parent.parent
MODEL_PATH = os.environ.get("POSE_MODEL", str(BASE / "models" / "yolo11n-pose.pt"))
DEVICE = os.environ.get("POSE_DEVICE") or (0 if torch.cuda.is_available() else "cpu")


def load_model():
    return YOLO(MODEL_PATH)


def new_tracker():
    """(tracker, fragment_ids): one per clip or per live camera, never shared."""
    return BYTETracker(IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml")))), itertools.count(-1, -1)


def analyze(video, rules, progress=lambda f: None):
    """Returns (frames, meta). frames: [{"i", "t", "people": [{"id", "box", "kp", "conf", "posture"}]}]"""
    model = load_model()
    tracker, fragment_ids = new_tracker()
    cap = cv2.VideoCapture(str(video))
    src_fps = rules.get("source_fps") or cap.get(cv2.CAP_PROP_FPS)  # override: some files lie about their fps
    if not src_fps or math.isnan(src_fps) or src_fps > 240:
        src_fps = 25.0
    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    step = 1.0 / rules["sample_fps"]
    frames, idx, next_t, size = [], 0, 0.0, None
    try:
        while cap.grab():
            t = idx / src_fps
            if t + 1e-6 >= next_t:
                ok, frame = cap.retrieve()
                if ok:
                    size = size or (frame.shape[1], frame.shape[0])  # after OpenCV's auto-rotation
                    people = _detect(model, tracker, fragment_ids, frame, rules)
                    frames.append({"i": len(frames), "t": round(t, 3), "people": people})
                next_t += step
                progress(min(idx / total, 1.0))
            idx += 1
    finally:
        cap.release()
    if not size:
        raise ValueError("Could not decode any frames")
    meta = {"width": size[0], "height": size[1], "fps": rules["sample_fps"], "src_fps": src_fps,
            "duration": round(idx / src_fps, 2)}
    return frames, meta


def _detect(model, tracker, fragment_ids, frame, rules):
    """ByteTrack driven directly (not model.track) so detections it hasn't confirmed are kept too: at 6 fps a
    running person's box often has zero IoU with the last one, and ByteTrack would hide them. Those get a unique
    negative id and tracks.rejoin() stitches them back together."""
    r = model.predict(frame, imgsz=rules["imgsz"], device=DEVICE, conf=rules.get("det_conf", 0.25), verbose=False)[0]
    det = r.boxes.cpu().numpy()
    tracked = tracker.update(det, frame)  # rows: x1, y1, x2, y2, id, score, cls, det_index
    id_of = {int(row[-1]): int(row[4]) for row in tracked}
    people = []
    for k, (box, conf, kp) in enumerate(zip(det.xyxy.tolist(), det.conf.tolist(), r.keypoints.data.tolist())):
        tid = id_of.get(k) or next(fragment_ids)
        box = [round(v, 1) for v in box]
        kp = [[round(x, 1), round(y, 1), round(c, 2)] for x, y, c in kp]
        people.append({"id": tid, "box": box, "kp": kp, "conf": round(conf, 2),
                       "posture": classify(box, kp, rules.get("posture"))})
    return people
