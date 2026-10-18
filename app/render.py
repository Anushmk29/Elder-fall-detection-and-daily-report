"""Annotated H.264 MP4 (<=720p, faststart) piped to ffmpeg, plus thumbnails and 3-frame strips for the LLM."""
import bisect
import subprocess

import cv2
import numpy as np

SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]
ROLE_COLORS = {"resident": (96, 160, 47), "visitor": (40, 150, 230)}  # BGR
ZONE_COLORS = {"room": (200, 140, 60), "rest": (120, 170, 90)}
KP_CONF = 0.3
OUT_FPS = 15
STRIP_W = 768  # each strip frame is 768 px wide


def render(video, out_dir, frames, tracks, analysis, rules, meta, thumbs, strips, progress=lambda f: None):
    """thumbs: [(t, filename)] annotated stills. strips: [(t, filename)] raw frames at t-1, t, t+1 side by side."""
    cap = cv2.VideoCapture(str(video))
    src_fps = meta["src_fps"]
    W, H = meta["width"], meta["height"]
    s = min(1.0, 1280 / W, 720 / H)
    ow, oh = int(W * s) // 2 * 2, int(H * s) // 2 * 2
    sx, sy = ow / W, oh / H
    out_fps = min(src_fps, OUT_FPS)

    draw = {f["i"]: [] for f in frames}
    for tr in tracks:
        for smp in tr["samples"]:
            draw[smp["i"]].append((tr, smp))
    times = [f["t"] for f in frames]
    banners = [(e["t"], max(e["t_end"], e["t"] + 3), e["banner"]) for e in analysis["events"] if e.get("banner")]
    zone_layer, zone_mask = _zone_layer(rules.get("zones", []), ow, oh)

    want_raw = sorted((max(0.0, t + d), name, j) for t, name in strips for j, d in enumerate((-1, 0, 1)))
    want_ann = sorted(thumbs)
    raw_got, total = {}, max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)

    out_path = out_dir / "annotated.mp4"
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ow}x{oh}",
         "-r", f"{out_fps:.3f}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)],
        stdin=subprocess.PIPE)
    idx, next_t, last = 0, 0.0, None
    try:
        while cap.grab():
            ft = idx / src_fps
            idx += 1
            if ft + 1e-6 < next_t:
                continue
            next_t += 1 / out_fps
            ok, frame = cap.retrieve()
            if not ok:
                continue
            frame = cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_AREA)
            k = bisect.bisect_right(times, ft + 0.5 / meta["fps"]) - 1
            items = draw[frames[k]["i"]] if k >= 0 and ft - times[k] < 2 / meta["fps"] else []
            if rules.get("face_blur"):
                for tr, smp in items:
                    _blur_head(frame, smp, sx, sy)
            while want_raw and want_raw[0][0] <= ft:
                _, name, j = want_raw.pop(0)
                raw_got[(name, j)] = frame.copy()
            last = frame.copy()
            _annotate(frame, items, sx, sy, zone_layer, zone_mask, rules.get("zones", []), banners, ft, meta)
            while want_ann and want_ann[0][0] <= ft:
                cv2.imwrite(str(out_dir / want_ann.pop(0)[1]), frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            ff.stdin.write(frame.tobytes())
            progress(min(idx / total, 1.0))
    finally:
        cap.release()
        ff.stdin.close()
        ff.wait()
    if ff.returncode:
        raise RuntimeError("ffmpeg encode failed")
    for _, name in want_ann:  # requested past the last frame
        if last is not None:
            cv2.imwrite(str(out_dir / name), last, [cv2.IMWRITE_JPEG_QUALITY, 82])
    for _, name in strips:
        parts = [raw_got.get((name, j), last) for j in range(3)]
        parts = [cv2.resize(p, (STRIP_W, int(oh * STRIP_W / ow))) for p in parts if p is not None]
        if parts:
            cv2.imwrite(str(out_dir / name), np.hstack(parts), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return out_path


def _zone_layer(zones, ow, oh):
    layer = np.zeros((oh, ow, 3), np.uint8)
    for z in zones:
        pts = (np.array(z["polygon"]) * [ow, oh]).astype(np.int32)
        cv2.fillPoly(layer, [pts], ZONE_COLORS.get(z.get("kind"), (180, 180, 180)))
    return layer, layer.any(axis=2).astype(np.uint8)


def _annotate(frame, items, sx, sy, zone_layer, zone_mask, zones, banners, ft, meta):
    cv2.copyTo(cv2.addWeighted(frame, 0.85, zone_layer, 0.15, 0), zone_mask, frame)
    oh, ow = frame.shape[:2]
    for z in zones:
        pts = (np.array(z["polygon"]) * [ow, oh]).astype(np.int32)
        cv2.polylines(frame, [pts], True, ZONE_COLORS.get(z.get("kind"), (180, 180, 180)), 1, cv2.LINE_AA)
        x, y = pts.min(axis=0)
        _label(frame, z["name"], (x + 6, y + 18), (90, 90, 90), 0.45)
    for tr, smp in items:
        color = ROLE_COLORS.get(tr.get("role"), (200, 200, 200))
        kp = [(int(x * sx), int(y * sy), c) for x, y, c in smp["kp"]]
        for a, b in SKELETON:
            if kp[a][2] >= KP_CONF and kp[b][2] >= KP_CONF:
                cv2.line(frame, kp[a][:2], kp[b][:2], color, 2, cv2.LINE_AA)
        for x, y, c in kp:
            if c >= KP_CONF:
                cv2.circle(frame, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)
        x1, y1, x2, y2 = smp["box"]
        p1, p2 = (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy))
        cv2.rectangle(frame, p1, p2, color, 1, cv2.LINE_AA)
        _label(frame, f"{tr.get('label', tr['id'])} #{tr['id']} | {smp['activity']}", (p1[0], max(p1[1] - 6, 16)), color, 0.5)
    for a, b, text in banners:
        if a <= ft <= b:
            cv2.rectangle(frame, (0, 0), (ow, 46), (40, 40, 210), -1)
            cv2.putText(frame, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            break
    s = int(meta.get("clock_start", 0) + ft) % 86400
    _label(frame, f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}", (12, oh - 12), (40, 40, 40), 0.55)


def _label(frame, text, org, color, scale):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(frame, (x - 3, y - h - 5), (x + w + 3, y + 4), color, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def _blur_head(frame, smp, sx, sy):
    """Pixelate the head area from nose/eye/ear keypoints (top of the box if none are visible)."""
    x1, y1, x2, y2 = smp["box"]
    bw = (x2 - x1) * sx
    head = [(x * sx, y * sy) for x, y, c in smp["kp"][:5] if c >= KP_CONF]
    if head:
        cx, cy = sum(p[0] for p in head) / len(head), sum(p[1] for p in head) / len(head)
        spread = max(max(p[0] for p in head) - min(p[0] for p in head), max(p[1] for p in head) - min(p[1] for p in head))
        r = max(spread * 1.2, bw * 0.22, 10)
    else:
        cx, cy, r = (x1 + x2) / 2 * sx, y1 * sy + bw * 0.25, max(bw * 0.3, 10)
    h, w = frame.shape[:2]
    a, b = max(int(cx - r), 0), max(int(cy - r), 0)
    c, d = min(int(cx + r), w), min(int(cy + r), h)
    if c - a > 2 and d - b > 2:
        roi = frame[b:d, a:c]
        frame[b:d, a:c] = cv2.resize(cv2.resize(roi, (6, 6), interpolation=cv2.INTER_AREA), (c - a, d - b),
                                     interpolation=cv2.INTER_NEAREST)
