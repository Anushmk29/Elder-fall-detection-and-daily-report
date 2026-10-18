"""Turn per-frame detections into tracks: re-join broken IDs, smooth posture, add walking/running."""
import math
from collections import Counter, defaultdict
from statistics import median

KP_CONF = 0.3


def foot(s):
    ank = [s["kp"][i] for i in (15, 16) if s["kp"][i][2] >= KP_CONF]
    if ank:
        return (sum(a[0] for a in ank) / len(ank), sum(a[1] for a in ank) / len(ank))
    x1, _, x2, y2 = s["box"]
    return ((x1 + x2) / 2, y2)


def center(s):
    x1, y1, x2, y2 = s["box"]
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def body_len(s):
    x1, y1, x2, y2 = s["box"]
    return max(x2 - x1, y2 - y1, 1.0)


def build(frames, meta, rules):
    cfg = rules["tracks"]
    raw = defaultdict(list)
    for f in frames:
        for p in f["people"]:
            raw[p["id"]].append({"i": f["i"], "t": f["t"], **p})
    tracks = sorted(({"id": k, "samples": v} for k, v in raw.items()), key=lambda tr: tr["samples"][0]["t"])
    tracks = rejoin(tracks, cfg["rejoin_gap_s"], cfg["rejoin_dist_bl"], cfg["run_bh_per_s"])
    tracks = [tr for tr in tracks if len(tr["samples"]) >= cfg["min_samples"]]
    for n, tr in enumerate(tracks, 1):
        tr["id"] = n
        smooth_posture(tr["samples"], cfg["smooth_window"])
        add_motion(tr["samples"], cfg)
    return tracks


def rejoin(tracks, max_gap, max_dist, max_speed):
    """Greedy: a track that starts shortly after another ended, close to where it ended (allowing for running
    speed over the gap), is the same person."""
    out = []  # ponytail: O(tracks^2); fine for <=60 s clips, index by time if clips get long
    for tr in tracks:  # sorted by start time
        s0, best, best_d = tr["samples"][0], None, None
        for cand in out:
            last = cand["samples"][-1]
            if not 0 < s0["t"] - last["t"] <= max_gap:
                continue
            d = math.dist(center(last), center(s0)) / body_len(last)
            if d <= max_dist + max_speed * (s0["t"] - last["t"]) and (best is None or d < best_d):
                best, best_d = cand, d
        if best:
            best["samples"].extend(tr["samples"])
        else:
            out.append(tr)
    return out


def smooth_posture(samples, window):
    raw = [s["posture"] for s in samples]
    k = window // 2
    for j, s in enumerate(samples):
        s["posture_raw"] = raw[j]
        s["posture"] = Counter(raw[max(0, j - k): j + k + 1]).most_common(1)[0][0]


def add_motion(samples, cfg):
    """speed = foot-point displacement over ~1 s, in body heights per second."""
    for s in samples:
        s["pt"] = center(s) if s["posture"] == "lying" else foot(s)
    j, speeds = 0, []
    for s in samples:
        while samples[j]["t"] < s["t"] - cfg["speed_window_s"]:
            j += 1
        a = samples[j]
        dt, h = s["t"] - a["t"], s["box"][3] - s["box"][1]
        s["speed"] = round(math.dist(a["pt"], s["pt"]) / dt / h, 3) if dt > 0 and h > 0 else 0.0
        speeds.append(s["speed"])
        if s["posture"] == "standing":
            sp = median(speeds[-3:])
            s["activity"] = ("running" if sp >= cfg["run_bh_per_s"] else
                             "walking" if sp >= cfg["walk_bh_per_s"] else "standing")
        else:
            s["activity"] = s["posture"]
