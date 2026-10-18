"""Rule-based ElderCare analysis: roles, zones, falls, lying outside rest zones, stillness, minutes."""
import math
from collections import defaultdict

from .tracks import body_len

KP_CONF = 0.3
ACTIVE = {"standing", "walking", "running", "bending"}
UPRIGHT = {"standing", "bending"}  # a fall can start bent over (picking something up); sitting -> lying is lying down


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


def _height(s):
    return s["box"][3] - s["box"][1]


def detect_falls(samples, cfg):
    """A fall is one of
    - standing/bending -> lying within window_s, with a real drop (hips down >= hip_drop_frac body heights, or the
      box shrinks to <= height_ratio of its upright height: what a high camera sees), down >= min_down_s;
    - standing/bending -> hips down >= floor_drop_frac body heights within floor_window_s and not standing again
      for min_down_s (falling towards the camera keeps the torso upright in the image, so it reads as sitting)."""
    found, k, n = [], 1, len(samples)
    floor_frac, floor_win = cfg.get("floor_drop_frac", 0), cfg.get("floor_window_s", 0)
    while k < n:
        s = samples[k]
        upright = [p for p in samples[:k] if p["posture"] in UPRIGHT and 0 < s["t"] - p["t"] <= cfg["window_s"]]
        if s["posture"] in UPRIGHT or not upright:
            k += 1
            continue
        u = upright[0]
        h = _height(u)
        drop = hip_y(s) - hip_y(u)
        lying = (s["posture"] == "lying" and samples[k - 1]["posture"] != "lying"
                 and (drop >= cfg["hip_drop_frac"] * h
                      or _height(s) <= cfg.get("height_ratio", 0) * max(_height(p) for p in upright)))
        fast = [p for p in upright if s["t"] - p["t"] <= floor_win]
        floor = floor_frac and fast and max(hip_y(s) - hip_y(p) - floor_frac * _height(p) for p in fast) >= 0
        if not (lying or floor):
            k += 1
            continue
        up = ("standing", "sitting") if lying else ("standing",)  # got up: off the floor
        end = next((m for m in range(k + 1, n) if samples[m]["posture"] in up), None)
        down_until = samples[end]["t"] if end is not None else samples[-1]["t"]
        if down_until - s["t"] < cfg["min_down_s"]:
            k += 1
            continue
        found.append({"t": s["t"], "t_end": down_until, "drop_s": round(s["t"] - upright[-1]["t"], 1),
                      "zones": s.get("zones", []), "how": "lying" if lying else "floor_drop",
                      "got_up_s": round(down_until - s["t"], 1) if end is not None else None})
        k = end if end is not None else n
    return found


def detect_vanish(samples, others, meta, cfg):
    """An upright person drops and then nobody can be seen: COCO pose models often miss a body lying on the floor,
    so losing someone right after their body dropped is itself a fall signature. Ignored when they were at the edge
    of the view (walked out). Returns a fall dict like detect_falls, or None."""
    gap = cfg.get("vanish_s", 0)
    if not gap or not samples:
        return None
    last = samples[-1]
    later = [s["t"] for o in others for s in o["samples"] if s["t"] > last["t"]]
    seen_again = min(later) if later else None
    if (seen_again if seen_again is not None else meta["duration"]) - last["t"] < gap:
        return None
    ref = [s for s in samples if last["t"] - 4 <= s["t"] < last["t"] - 1 and s["posture"] in UPRIGHT]
    recent = [s for s in samples if s["t"] >= last["t"] - 1.5]
    if not ref:
        return None
    h = sorted(_height(s) for s in ref)[len(ref) // 2]
    hip0 = sorted(hip_y(s) for s in ref)[len(ref) // 2]
    if not (min(_height(s) for s in recent) <= cfg.get("vanish_height_ratio", 0.8) * h
            or max(hip_y(s) for s in recent) - hip0 >= cfg["hip_drop_frac"] * h):
        return None
    m = cfg.get("vanish_edge", 0.03)
    x1, y1, x2, y2 = last["box"]
    if x1 < m * meta["width"] or y1 < m * meta["height"] or x2 > (1 - m) * meta["width"] or y2 > (1 - m) * meta["height"]:
        return None
    end = seen_again if seen_again is not None else meta["duration"]
    return {"t": last["t"], "t_end": end, "zones": last.get("zones", []), "how": "vanished",
            "got_up_s": round(seen_again - last["t"], 1) if seen_again is not None else None}


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

        falls = detect_falls(S, rules["fall"])
        v = detect_vanish(S, [o for o in order if o is not tr], meta, rules["fall"])
        if v and not any(abs(f["t"] - v["t"]) < 5 or f["got_up_s"] is None for f in falls):
            falls.append(v)
        for f in falls:
            if set(f["zones"]) & rest:
                continue  # lying down on the bed/sofa, not a fall
            got_up = f"got up in {f['got_up_s']:.0f} s" if f["got_up_s"] is not None else "had not got up when the clip ended"
            how = ("dropped suddenly and could no longer be seen (possibly on the floor)" if f["how"] == "vanished"
                   else f"went from upright to the floor in under {rules['fall']['window_s']:.0f} s")
            events.append({"type": "fall", "severity": "alert", "t": f["t"], "t_end": f["t_end"], "who": who,
                           "got_up_s": f["got_up_s"], "how": f["how"],
                           "banner": f"FALL DETECTED · {who}", "title": "Fall detected",
                           "detail": f"{who} {how}, and {got_up}."})
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
