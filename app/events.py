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
        if s["posture"] in UPRIGHT or not upright or s.get("cut"):
            k += 1
            continue
        u = upright[0]
        h = _height(u)
        # judge the drop over the first settle_s of lying: at 6 fps the first "lying" frame is often mid-fall
        settle = [p for p in samples[k:k + 12] if p["t"] - s["t"] <= cfg.get("settle_s", 0) and p["posture"] == "lying"] or [s]
        drop = max(hip_y(p) for p in settle) - hip_y(u)
        lying = (s["posture"] == "lying" and samples[k - 1]["posture"] != "lying"
                 and (drop >= cfg["hip_drop_frac"] * h
                      or min(_height(p) for p in settle) <= cfg.get("height_ratio", 0) * max(_height(p) for p in upright)))
        fast = [p for p in upright if s["t"] - p["t"] <= floor_win]
        floor = floor_frac and fast and max(hip_y(s) - hip_y(p) - floor_frac * _height(p) for p in fast) >= 0             and _height(s) <= cfg.get("floor_height_ratio", 1.0) * max(_height(p) for p in upright)
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
                      "h_ratio": round(_height(s) / max(_height(p) for p in upright), 2), "hip_drop": round(drop / h, 2),
                      "got_up_s": round(down_until - s["t"], 1) if end is not None else None})
        k = end if end is not None else n
    return found


def detect_collapse(samples, cfg):
    """Posture-free fall: the top of an upright person's box drops >= collapse_frac of their height within
    collapse_window_s while the feet don't rise (walking away from the camera shrinks the box too, but the feet move
    up the image), and they aren't that tall again for min_down_s. Catches falls that end sitting or slumped on the
    floor, which a low-resolution pose often still reads as 'standing' (see learning/GUIDE.md, Le2i)."""
    frac = cfg.get("collapse_frac", 0)
    found, k, n = [], 1, len(samples)
    while frac and k < n:
        s = samples[k]
        ups = [p for p in samples[:k] if p["posture"] in UPRIGHT and 0 < s["t"] - p["t"] <= cfg.get("collapse_window_s", 1.5)]
        p = None if s.get("cut") else next((p for p in ups if s["box"][1] - p["box"][1] >= frac * _height(p)
                  and s["box"][3] >= p["box"][3] - cfg.get("collapse_feet", 0.15) * _height(p)), None)
        if not p:
            k += 1
            continue
        end = next((m for m in range(k + 1, n) if _height(samples[m]) > cfg.get("collapse_up_ratio", 0.85) * _height(p)), None)
        down_until = samples[end]["t"] if end is not None else samples[-1]["t"]
        if down_until - s["t"] < cfg["min_down_s"]:
            k += 1
            continue
        found.append({"t": s["t"], "t_end": down_until, "zones": s.get("zones", []), "how": "collapse",
                      "h_ratio": round(_height(s) / _height(p), 2),
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
    aspects = sorted((s["box"][2] - s["box"][0]) / max(_height(s), 1e-6) for s in recent)
    if aspects[len(aspects) // 2] < cfg.get("vanish_min_aspect", 0):
        return None  # still an upright shape when last seen: walked behind something, not on the floor
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


def motion(samples, a, b, min_conf=KP_CONF):
    """How much the body moves while down: median over [a, b] of the mean keypoint displacement per second, in
    body lengths. Someone stretching or reading on the floor moves; someone hurt or unconscious mostly doesn't."""
    S = [s for s in samples if a <= s["t"] <= b]
    per_s = []
    for p, q in zip(S, S[1:]):
        dt = q["t"] - p["t"]
        pts = [math.dist(u[:2], v[:2]) for u, v in zip(p["kp"], q["kp"]) if u[2] >= min_conf and v[2] >= min_conf]
        if 0 < dt <= 0.5 and pts:
            per_s.append(sum(pts) / len(pts) / dt / body_len(p))
    return round(sorted(per_s)[len(per_s) // 2], 3) if per_s else 0.0


def mark_low(samples, ratio, k=6):
    """Posture-free "down": the box is at most `ratio` of this person's own upright height *at the same foot
    position* (median of the k walking samples nearest in box-bottom y, so walking away from the camera, which
    shrinks the box, doesn't count). Walking, not standing: a slumped person is often read as "standing", and would
    become their own reference. Catches falls that end sitting or slumped on the floor, which the pose model doesn't
    call lying. Sets s["low"]; needs >= k walking samples."""
    ref = [(s["box"][3], _height(s)) for s in samples if s.get("activity") == "walking"]
    for s in samples:
        near = sorted(ref, key=lambda r: abs(r[0] - s["box"][3]))[:k]  # ponytail: O(n^2), fine for a 330 s window
        s["low"] = bool(ratio and len(near) >= k and s["posture"] != "lying"
                        and _height(s) <= ratio * sorted(h for _, h in near)[k // 2])


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
            s["cut"] = s["box"][1] <= rules["fall"].get("cut_top", -1) * H  # cut off by the top of the frame
    # The longest-visible person is the resident; everyone else is a visitor/caregiver.
    order = sorted(tracks, key=lambda tr: -tr["visible_s"])
    for n, tr in enumerate(order):
        tr["role"] = "resident" if n == 0 else "visitor"
        tr["label"] = "Resident" if n == 0 else ("Visitor/Caregiver" if len(order) == 2 else f"Visitor/Caregiver {n}")

    rooms = {z["name"] for z in zones if z.get("kind") == "room"}
    rest = {z["name"] for z in zones if z.get("kind") == "rest"}
    seat = rest | {z["name"] for z in zones if z.get("kind") == "seat"}  # chairs: sitting there is low, not fallen
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
        falls += [c for c in detect_collapse(S, rules["fall"]) if not any(abs(f["t"] - c["t"]) < 5 for f in falls)]
        v = detect_vanish(S, [o for o in order if o is not tr], meta, rules["fall"])
        if v and not any(abs(f["t"] - v["t"]) < 5 or f["got_up_s"] is None for f in falls):
            falls.append(v)
        for f in falls:
            # where they ended up, 1-3 s later: the trigger frame is often mid-fall, swinging past the sofa
            settled = [s for s in S if f["t"] + 1 <= s["t"] <= f["t"] + 3]
            where = settled[len(settled) // 2]["zones"] if settled else f["zones"]
            if set(where) & rest:
                continue  # lying down on the bed/sofa, not a fall
            got_up = f"got up in {f['got_up_s']:.0f} s" if f["got_up_s"] is not None else "had not got up when the clip ended"
            how = ("dropped suddenly and could no longer be seen (possibly on the floor)" if f["how"] == "vanished"
                   else f"went from upright to the floor in under {rules['fall']['window_s']:.0f} s")
            events.append({"type": "fall", "severity": "alert", "t": f["t"], "t_end": f["t_end"], "who": who,
                           "got_up_s": f["got_up_s"], "how": f["how"],
                           "banner": f"FALL DETECTED · {who}", "title": "Fall detected",
                           "detail": f"{who} {how}, and {got_up}."})
        mark_low(S, rules.get("floor_low_ratio", 0))
        down = lambda s: ((s["posture"] == "lying" and not set(s["zones"]) & rest)  # noqa: E731
                          or (s["low"] and not set(s["zones"]) & seat))
        for a, b in runs(S, down, fps, rules["lying_outside_rest_s"]):
            m = motion(S, a, b)
            moving = m >= rules.get("floor_moving_bl_s", 0.1)
            span = [s for s in S if a <= s["t"] <= b]
            how = "lying" if sum(s["posture"] == "lying" for s in span) * 2 >= len(span) else "low (sitting or slumped)"
            events.append({"type": "lying_outside_rest", "severity": "warn", "t": a, "t_end": b, "who": who,
                           "motion": m, "moving": moving, "how": how,
                           "title": "On the floor, moving" if moving else "On the floor, not moving",
                           "detail": f"{who} has been {how} on the floor, outside bed/sofa/chairs, for {b - a:.0f} s, "
                                     f"{'moving around' if moving else 'barely moving'}."})
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
