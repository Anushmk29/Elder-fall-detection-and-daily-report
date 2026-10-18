"""Rule-based ElderCare analysis: roles, zones, falls, lying outside rest zones, stillness, minutes."""
import math
from collections import defaultdict

from .tracks import body_len

KP_CONF = 0.3
ACTIVE = {"standing", "walking", "running", "bending"}


def in_poly(x, y, poly):
    inside = False
    for k in range(len(poly)):
        (x1, y1), (x2, y2) = poly[k], poly[(k + 1) % len(poly)]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def zones_at(pt_norm, zones):
    return [z for z in zones if in_poly(pt_norm[0], pt_norm[1], z["polygon"])]


def clock(meta, t):
    s = int(meta.get("clock_start", 0) + t) % 86400
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def sample_dt(samples, k, fps):
    nxt = samples[k + 1]["t"] - samples[k]["t"] if k + 1 < len(samples) else 1 / fps
    return min(nxt, 2 / fps)  # don't credit time across detection gaps


def hip_y(s):
    hips = [s["kp"][i][1] for i in (11, 12) if s["kp"][i][2] >= KP_CONF]
    return sum(hips) / len(hips) if hips else (s["box"][1] + s["box"][3]) / 2


def detect_falls(samples, cfg):
    """upright -> lying within window_s, hips drop >= hip_drop_frac body heights, stays down >= min_down_s."""
    found, k, n = [], 1, len(samples)
    while k < n:
        s = samples[k]
        if s["posture"] != "lying" or samples[k - 1]["posture"] == "lying":
            k += 1
            continue
        upright = [p for p in samples[:k] if p["posture"] == "standing" and 0 < s["t"] - p["t"] <= cfg["window_s"]]
        if not upright:
            k += 1
            continue
        u = upright[0]
        h = u["box"][3] - u["box"][1]
        if hip_y(s) - hip_y(u) < cfg["hip_drop_frac"] * h:
            k += 1
            continue
        end = next((m for m in range(k + 1, n) if samples[m]["posture"] in ("standing", "sitting")), None)
        down_until = samples[end]["t"] if end is not None else samples[-1]["t"]
        if down_until - s["t"] < cfg["min_down_s"]:
            k += 1
            continue
        found.append({"t": s["t"], "t_end": down_until, "drop_s": round(s["t"] - upright[-1]["t"], 1),
                      "got_up_s": round(down_until - s["t"], 1) if end is not None else None})
        k = end if end is not None else n
    return found


def runs(samples, pred, fps, min_s):
    """Contiguous runs (allowing small detection gaps) where pred(sample) holds for >= min_s."""
    out, start, prev = [], None, None
    for s in samples:
        if start is not None and pred(s) and s["t"] - prev["t"] <= 3 / fps:
            prev = s
            continue
        if start is not None and prev["t"] - start["t"] >= min_s:
            out.append((start["t"], prev["t"]))
        start, prev = (s if pred(s) else None), s
    if start is not None and prev["t"] - start["t"] >= min_s:
        out.append((start["t"], prev["t"]))
    return out


def stillness(samples, min_s, move_bl):
    out, a = [], 0
    for k in range(1, len(samples) + 1):
        if k < len(samples) and math.dist(samples[k]["pt"], samples[a]["pt"]) <= move_bl * body_len(samples[a]):
            continue
        if samples[k - 1]["t"] - samples[a]["t"] >= min_s:
            out.append((samples[a]["t"], samples[k - 1]["t"]))
        a = k
    return out


def analyze(tracks, meta, rules):
    zones, fps = rules.get("zones", []), meta["fps"]
    W, H = meta["width"], meta["height"]
    for tr in tracks:
        tr["visible_s"] = sum(sample_dt(tr["samples"], k, fps) for k in range(len(tr["samples"])))
        for s in tr["samples"]:
            s["zones"] = [z["name"] for z in zones_at((s["pt"][0] / W, s["pt"][1] / H), zones)]
    # The longest-visible person is the resident; everyone else is a visitor/caregiver.
    order = sorted(tracks, key=lambda tr: -tr["visible_s"])
    for n, tr in enumerate(order):
        tr["role"] = "resident" if n == 0 else "visitor"
        tr["label"] = "Resident" if n == 0 else ("Visitor/Caregiver" if len(order) == 2 else f"Visitor/Caregiver {n}")

    rooms = {z["name"] for z in zones if z.get("kind") == "room"}
    rest = {z["name"] for z in zones if z.get("kind") == "rest"}
    events, people, timeline = [], [], []
    for tr in order:
        S, who = tr["samples"], tr["label"]
        minutes, dwell = defaultdict(float), defaultdict(float)
        for k, s in enumerate(S):
            dt = sample_dt(S, k, fps)
            cat = "lying" if s["posture"] == "lying" else "sitting" if s["posture"] == "sitting" else "active"
            minutes[cat] += dt
            for z in s["zones"]:
                dwell[z] += dt
        # room timeline
        seg = None
        for s in S:
            room = next((z for z in s["zones"] if z in rooms), None)
            if seg and seg["zone"] == room and s["t"] - seg["end"] <= 2.0:
                seg["end"] = s["t"]
            else:
                if seg and seg["zone"]:
                    timeline.append(seg)
                seg = {"who": who, "zone": room, "start": s["t"], "end": s["t"]}
        if seg and seg["zone"]:
            timeline.append(seg)

        for f in detect_falls(S, rules["fall"]):
            got_up = f"got up in {f['got_up_s']:.0f} s" if f["got_up_s"] is not None else "had not got up when the clip ended"
            events.append({"type": "fall", "severity": "alert", "t": f["t"], "t_end": f["t_end"], "who": who,
                           "got_up_s": f["got_up_s"],
                           "banner": f"FALL DETECTED · {who}", "title": "Fall detected",
                           "detail": f"{who} went from upright to lying in under {rules['fall']['window_s']:.0f} s "
                                     f"and {got_up}."})
        for a, b in runs(S, lambda s: s["posture"] == "lying" and not set(s["zones"]) & rest,
                         fps, rules["lying_outside_rest_s"]):
            events.append({"type": "lying_outside_rest", "severity": "warn", "t": a, "t_end": b, "who": who,
                           "title": "Lying outside bed/sofa",
                           "detail": f"{who} was lying outside a rest area for {b - a:.0f} s."})
        for a, b in stillness(S, rules["stillness"]["min_s"], rules["stillness"]["move_bl"]):
            events.append({"type": "stillness", "severity": "info", "t": a, "t_end": b, "who": who,
                           "title": "Long stillness", "detail": f"{who} barely moved for {(b - a) / 60:.1f} min."})
        people.append({"id": tr["id"], "label": who, "role": tr["role"], "visible_s": round(tr["visible_s"], 1),
                       "minutes": {k: round(v / 60, 2) for k, v in minutes.items()},
                       "zones": {k: round(v, 1) for k, v in dwell.items()}})

    events.sort(key=lambda e: e["t"])
    for n, e in enumerate(events, 1):
        e.update(id=n, clock=clock(meta, e["t"]), thumb=f"ev{n}.jpg")

    res = people[0] if people else {"minutes": {}, "zones": {}, "visible_s": 0}
    room_time = {z: v for z, v in res["zones"].items() if z in rooms}
    glance = [
        {"label": "Observed", "value": _dur(meta["duration"]), "note": f"from {clock(meta, 0)}"},
        {"label": "Active", "value": _dur(res["minutes"].get("active", 0) * 60), "note": "standing or walking"},
        {"label": "Sitting", "value": _dur(res["minutes"].get("sitting", 0) * 60)},
        {"label": "Lying", "value": _dur(res["minutes"].get("lying", 0) * 60)},
        {"label": "Rooms", "value": str(len(room_time)), "note": ", ".join(room_time) or "no zones matched"},
        {"label": "Alerts", "value": str(sum(e["severity"] != "info" for e in events)),
         "note": f"{sum(e['type'] == 'fall' for e in events)} fall(s)", "tone": "alert" if events else "ok"},
        {"label": "Visitors", "value": str(max(len(people) - 1, 0))},
    ]
    return {"people": people, "events": events, "timeline": timeline, "glance": glance,
            "zones": [{"name": z["name"], "kind": z.get("kind", "room")} for z in zones]}


def describe_at(tracks, t, max_gap=0.6):
    """'Resident: walking in Kitchen; ...' for the samples nearest t (used as LLM context and fallback captions)."""
    parts = []
    for tr in tracks:
        s = min(tr["samples"], key=lambda s: abs(s["t"] - t))
        if abs(s["t"] - t) <= max_gap:
            parts.append(f"{tr['label']}: {s['activity']}" + (f" in {s['zones'][0]}" if s["zones"] else ""))
    return "; ".join(parts) or "nobody detected"


def _dur(sec):
    sec = int(round(sec))
    return f"{sec} s" if sec < 60 else f"{sec // 60} min {sec % 60:02d} s" if sec < 3600 else f"{sec // 3600} h {sec % 3600 // 60} min"
