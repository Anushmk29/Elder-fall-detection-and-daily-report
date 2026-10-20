"""Daily routine from the live event log, compared with the previous 7 days.

metrics(day)   one day's events-YYYY-MM-DD.jsonl -> wake/bed time, kitchen visits, bathroom, active time, longest
               rest, night activity, falls; saved as data/live/daily/YYYY-MM-DD.json
baseline(day)  mean/min/max of up to 7 earlier daily files
report(day)    the family message, e.g. "Mrs. Rao — Thu 16 Oct / Woke 6:52 (usual 6:30–7:15) ✅ ..."

Rooms are recognised by name (the camera's `room:` in cameras.yaml): kitchen*, bath*/toilet*/wc, bed*, and outdoor
places (porch, garden, outside stairs, path...). A line about a kind of room is only written if a camera saw one.

One person lives here, so all cameras' visits in time order are her path through the house (journey). Two cameras
seeing someone at the same time means somebody else was around (or the two views overlap).
"""
import json
import os
from datetime import datetime, timedelta

from . import live

DAILY_DIR = live.LIVE_DIR / "daily"
MERGE_GAP_S = 300        # visits to the same room this close together are one visit (occlusion, brief exits)
MEAL_MIN_S = 180         # a kitchen visit this long counts as a meal
NIGHT = (23, 5)          # night-time hours [23:00, 05:00)
OVERLAP_S = 30           # two cameras seeing someone this long at the same time
OUTDOOR = ("outside", "porch", "garden", "yard", "path", "gate", "terrace", "balcony", "drive", "lawn")
ACTIVE = {"standing", "bending"}
NUMERIC = ["wake_min", "bed_min", "kitchen_visits", "bathroom_count", "bathroom_longest_min", "active_min",
           "rest_longest_min", "night_episodes"]


def _kind(room):
    r = (room or "").lower()
    return ("kitchen" if "kitchen" in r else "bathroom" if any(w in r for w in ("bath", "toilet", "wc"))
            else "bedroom" if "bed" in r else "outdoor" if any(w in r for w in OUTDOOR) else "other")


def _hm(ts):
    return datetime.fromtimestamp(ts).strftime("%H:%M")


def _minute_of_day(ts):
    d = datetime.fromtimestamp(ts)
    return d.hour * 60 + d.minute


def visits(evts, merge=True):
    """[(room, start_ts, end_ts)] from zone_enter/zone_exit per camera, same-room visits merged within MERGE_GAP_S."""
    open_, raw = {}, []
    end_of_log = max((e["ts"] for e in evts), default=0)
    for e in sorted(evts, key=lambda e: e["ts"]):
        if e["type"] == "zone_enter":
            open_[e["camera"]] = (e["room"], e["ts"])
        elif e["type"] in ("zone_exit", "camera_offline") and e["camera"] in open_:
            room, start = open_.pop(e["camera"])
            raw.append((room, start, e["ts"]))
    raw += [(room, start, end_of_log) for room, start in open_.values()]
    if not merge:
        return sorted(raw, key=lambda v: v[1])
    merged = []
    for room, a, b in sorted(raw, key=lambda v: v[1]):
        prev = next((m for m in reversed(merged) if m[0] == room), None)
        if prev and a - prev[2] <= MERGE_GAP_S:
            prev[2] = max(prev[2], b)
        else:
            merged.append([room, a, b])
    return [tuple(m) for m in merged]


def posture_spans(evts):
    """[(posture, room, start_ts, end_ts)] per camera: a posture lasts until the next change or exit."""
    spans, cur = [], {}
    for e in sorted(evts, key=lambda e: e["ts"]):
        c = e["camera"]
        if c in cur and e["type"] in ("posture_change", "zone_exit", "camera_offline"):
            p, room, a = cur.pop(c)
            spans.append((p, room, a, e["ts"]))
        if e["type"] == "posture_change" and e.get("posture"):
            cur[c] = (e["posture"], e.get("room"), e["ts"])
    end = max((e["ts"] for e in evts), default=0)
    spans += [(p, room, a, end) for p, room, a in cur.values()]
    return spans


def _meal(ts):
    h = datetime.fromtimestamp(ts).hour + datetime.fromtimestamp(ts).minute / 60
    return "breakfast" if h < 10.5 else "lunch" if h < 15 else "tea" if h < 18 else "dinner"


def _night(ts):
    h = datetime.fromtimestamp(ts).hour
    return h >= NIGHT[0] or h < NIGHT[1]


def places(v, spans):
    """[{"room", "min", "visits", "postures": {posture: min}}], most time first."""
    out = {}
    for room, a, b in v:
        p = out.setdefault(room, {"room": room, "min": 0, "visits": 0, "postures": {}})
        p["min"] += (b - a) / 60
        p["visits"] += 1
    for posture, room, a, b in spans:
        if room in out:
            out[room]["postures"][posture] = out[room]["postures"].get(posture, 0) + (b - a) / 60
    for p in out.values():
        p["min"] = round(p["min"])
        p["postures"] = {k: round(m) for k, m in sorted(p["postures"].items(), key=lambda x: -x[1]) if round(m)}
    return sorted(out.values(), key=lambda p: -p["min"])


def outings(v):
    """Outdoor visits less than MERGE_GAP_S apart joined into trips: [{"from", "to", "min", "rooms"}]."""
    trips = []
    for room, a, b in v:
        if _kind(room) != "outdoor":
            continue
        if trips and a - trips[-1]["b"] <= MERGE_GAP_S:
            trips[-1]["b"] = max(trips[-1]["b"], b)
            trips[-1]["rooms"] += [] if room in trips[-1]["rooms"] else [room]
        else:
            trips.append({"a": a, "b": b, "rooms": [room]})
    return [{"from": _hm(t["a"]), "to": _hm(t["b"]), "min": round((t["b"] - t["a"]) / 60), "rooms": t["rooms"]}
            for t in trips]


def together(raw):
    """[{"at", "rooms"}]: two cameras seeing someone at once for >= OVERLAP_S, one entry per minute."""
    out = []
    for i, (r1, a1, b1) in enumerate(raw):
        for r2, a2, b2 in raw[i + 1:]:
            if a2 >= b1:
                break  # raw is sorted by start
            if r1 != r2 and min(b1, b2) - max(a1, a2) >= OVERLAP_S:
                at = _hm(max(a1, a2))
                if not out or out[-1]["at"] != at:
                    out.append({"at": at, "rooms": sorted({r1, r2})})
    return out


def ai_postures(reviews):
    """Minutes per posture from the AI reviews (each counted frame stands for AI_STEP_S of footage)."""
    out = {}
    for e in reviews:
        for k, n in e["ai"]["postures"].items():
            out[k] = out.get(k, 0) + n * live.AI_STEP_S / 60
    return out


def activities(reviews, top=6):
    """[{"activity", "min", "rooms": {room: first HH:MM}}] from the AI reviews, most time first."""
    out = {}
    for e in sorted(reviews, key=lambda e: e["ts"]):
        a = e["ai"].get("activity", "").strip().lower().rstrip(".")
        if a:
            x = out.setdefault(a, {"activity": a, "min": 0, "rooms": {}})
            x["min"] += e["ai"].get("span_s", 0) / 60
            x["rooms"].setdefault(e.get("room") or "?", _hm(e["ts"]))
    return [{**x, "min": round(x["min"], 1)} for x in sorted(out.values(), key=lambda x: -x["min"])[:top]]


def metrics(day):
    evts = live.read_events(day)
    v = visits(evts)
    spans = posture_spans(evts)
    day0 = datetime.strptime(day, "%Y-%m-%d").timestamp()
    morning = [s for s in spans if s[0] in ACTIVE and s[2] >= day0 + 4 * 3600]
    evening = [s for s in spans if s[0] == "lying" and _kind(s[1]) == "bedroom" and s[2] >= day0 + 19 * 3600]
    kitchen = [(a, b) for room, a, b in v if _kind(room) == "kitchen" and b - a >= MEAL_MIN_S]
    bath = [(a, b) for room, a, b in v if _kind(room) == "bathroom"]
    rest = max((s for s in spans if s[0] in ("lying", "sitting") and _kind(s[1]) != "bedroom"),
               key=lambda s: s[3] - s[2], default=None)
    night = [(room, a, b) for room, a, b in v if _night(a)]
    falls = [e for e in evts if e["type"] in ("fall_alert", "fall_recovered", "floor_ai_alert")]
    reviews = [e for e in evts if e["type"] == "ai_review" and e.get("ai")]
    ai = ai_postures(reviews)
    kinds = {_kind(e.get("room")) for e in evts}  # kinds of room a camera saw anything in this day
    m = {
        "date": day,
        "observed": bool(evts),
        "wake_min": _minute_of_day(morning[0][2]) if morning else None,
        "bed_min": _minute_of_day(evening[-1][2]) if evening else None,
        "kitchen_visits": len(kitchen) if "kitchen" in kinds else None,
        "meals": [{"label": _meal(a), "at": _hm(a), "min": round((b - a) / 60)} for a, b in kitchen],
        "bathroom_count": len(bath) if "bathroom" in kinds else None,
        "bathroom_longest_min": round(max((b - a for a, b in bath), default=0) / 60) if "bathroom" in kinds else None,
        # with AI reviews, postures come from the vision model (the pose model calls sweeping or bending "lying")
        "active_min": round(sum(ai.get(k, 0) for k in ("standing", "walking", "bending")) if reviews else
                            sum(b - a for p, _, a, b in spans if p in ACTIVE) / 60),
        "rest_longest_min": round((rest[3] - rest[2]) / 60) if rest else 0,
        "rest_longest": {"room": rest[1], "from": _hm(rest[2]), "to": _hm(rest[3]),
                         "posture": rest[0]} if rest else None,
        "night_episodes": len(night),
        "night": [{"at": _hm(a), "min": round((b - a) / 60), "room": room} for room, a, b in night],
        "falls": [{"at": e["clock"], "type": e["type"], "detail": e["detail"]} for e in falls],
        "alerts": [{"at": e["clock"], "title": e["title"]} for e in evts if e["severity"] == "alert"],
        "first_seen": _hm(v[0][1]) if v else None,
        "last_seen": _hm(max(b for _, _, b in v)) if v else None,
        "sitting_min": round(ai.get("sitting", 0) if reviews else
                             sum(b - a for p, _, a, b in spans if p == "sitting") / 60),
        "lying_min": round(ai.get("lying", 0) if reviews else sum(b - a for p, _, a, b in spans if p == "lying") / 60),
        "ai_reviewed": {"segments": len(reviews), "min": round(sum(e["ai"].get("span_s", 0) for e in reviews) / 60)},
        "activities": activities(reviews),
        "ai_dismissed": [{"at": e["clock"], "detail": e["detail"]} for e in evts if e["type"] == "fall_dismissed"],
        "places": places(v, spans),
        "outings": outings(v),
        "journey": [{"room": room, "from": _hm(a), "to": _hm(b), "min": round((b - a) / 60)} for room, a, b in v],
        "together": together(visits(evts, merge=False)),
        "bedroom": "bedroom" in kinds,
    }
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    (DAILY_DIR / f"{day}.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    return m


def baseline(day, days=7):
    """{metric: {"mean", "lo", "hi", "n"}} over up to `days` earlier daily files that saw anything."""
    d0, hist = datetime.strptime(day, "%Y-%m-%d"), []
    synthetic = False
    for k in range(1, days + 1):
        p = DAILY_DIR / f"{(d0 - timedelta(days=k)):%Y-%m-%d}.json"
        if p.exists():
            m = json.loads(p.read_text(encoding="utf-8"))
            if m.get("observed", True):
                hist.append(m)
                synthetic |= bool(m.get("synthetic"))
    out = {}
    for key in NUMERIC:
        vals = [m[key] for m in hist if m.get(key) is not None]
        if vals:
            out[key] = {"mean": sum(vals) / len(vals), "lo": min(vals), "hi": max(vals), "n": len(vals)}
    return {"days": len(hist), "synthetic": synthetic, "metrics": out}


def compare(m, base):
    """{metric: 'ok' | 'high' | 'low'}: outside the 7-day range AND >25 % (times: >45 min) from the mean."""
    flags = {}
    for key, b in base["metrics"].items():
        x = m.get(key)
        if x is None:
            continue
        far = abs(x - b["mean"]) > (45 if key in ("wake_min", "bed_min") else max(0.25 * b["mean"], 1))
        flags[key] = "high" if x > b["hi"] and far else "low" if x < b["lo"] and far else "ok"
    return flags


def _clock(minutes):
    return f"{int(minutes) // 60}:{int(minutes) % 60:02d}"


def _dur(minutes):
    return f"{minutes // 60}h{minutes % 60:02d}m" if minutes >= 60 else f"{minutes} min"


def report(day, ai=None):
    """Plain-text family message. Returns (text, metrics, baseline, flags)."""
    m, base = metrics(day), baseline(day)
    f, bm = compare(m, base), base["metrics"]
    mark = lambda k: " ⚠️" if f.get(k) in ("high", "low") else " ✅" if k in f else ""  # noqa: E731
    name = os.environ.get("RESIDENT_NAME", "Resident")
    lines = [f"{name} — {datetime.strptime(day, '%Y-%m-%d'):%a %d %b}"]
    if not m["observed"]:
        return "\n".join(lines + ["No camera data for this day."]), m, base, f

    first = []
    if m["wake_min"] is not None:
        usual = f" (usual {_clock(bm['wake_min']['lo'])}–{_clock(bm['wake_min']['hi'])})" if "wake_min" in bm else ""
        first.append(f"{'Woke' if m.get('bedroom', True) else 'First up'} {_clock(m['wake_min'])}{usual}"
                     f"{mark('wake_min')}")
    if m["kitchen_visits"] is not None:
        meals = ", ".join(f"{x['label']} {x['at']}" for x in m["meals"])
        first.append(f"{m['kitchen_visits']} kitchen visit{'s' if m['kitchen_visits'] != 1 else ''}"
                     + (f" ({meals})" if meals else "") + mark("kitchen_visits"))
    if m.get("first_seen"):
        first.append(f"Seen {m['first_seen']}–{m['last_seen']}")
    lines.append(" · ".join(first))
    if m.get("places"):
        lines.append("Where: " + ", ".join(
            f"{p['room']} {_dur(p['min'])}" + (f" (mostly {next(iter(p['postures']))})" if p["postures"] else "")
            for p in m["places"][:4]) + ".")
    if m.get("outings"):
        lines.append(f"Went outside {len(m['outings'])}×: " + "; ".join(
            f"{o['from']}–{o['to']} ({', '.join(r.lower() for r in o['rooms'])})" for o in m["outings"][:6])
            + (" …" if len(m["outings"]) > 6 else "") + ".")

    if m.get("activities"):
        lines.append("Seen doing (AI): " + "; ".join(
            f"{a['activity']} ({', '.join(f'{r} {at}' for r, at in list(a['rooms'].items())[:3])})"
            for a in m["activities"]) + ".")

    active = f"Active time {_dur(m['active_min'])}"
    if "active_min" in bm and bm["active_min"]["mean"]:
        pct = round(100 * (m["active_min"] - bm["active_min"]["mean"]) / bm["active_min"]["mean"])
        active += f" ({'↓' if pct < 0 else '↑'}{abs(pct)}% vs. 7-day average)"
    if m.get("sitting_min") is not None:
        active += f" · sitting {_dur(m['sitting_min'])} · lying {_dur(m['lying_min'])}"
    if m.get("ai_reviewed", {}).get("segments"):
        active += " (AI-checked)"
    bath = (f"Bathroom {m['bathroom_count']}×, longest {m['bathroom_longest_min']} min{mark('bathroom_count')} · "
            if m["bathroom_count"] is not None else "")
    lines.append(f"{bath}{active}{mark('active_min')}")

    third = ["No falls." if not m["falls"] else
             "; ".join(f"Fall at {x['at']}: {x['detail'].rstrip('.')}" for x in m["falls"]) + ". ⚠️"]
    if m.get("ai_dismissed"):
        n = len(m["ai_dismissed"])
        third.append(f"AI check: {n} possible fall{'s' if n != 1 else ''} looked at, "
                     f"{'none were falls' if n != 1 else 'not a fall'}.")
    r = m["rest_longest"]
    if r and m["rest_longest_min"] >= 30:
        longer = " which is longer than usual" if f.get("rest_longest_min") == "high" else ""
        third.append(f"{'Lay' if r['posture'] == 'lying' else 'Sat'} in the {(r['room'] or 'living room').lower()}"
                     f" {r['from']}–{r['to']}{',' if longer else ''}{longer}.")
    lines.append(" ".join(third))

    if m.get("together"):
        t = m["together"]
        lines.append(f"Someone else was around (two cameras saw a person at once) {len(t)}×, first at {t[0]['at']} "
                     f"({' + '.join(t[0]['rooms'])}).")
    if m["night"]:
        n = m["night"][0]
        night = f"Night: up at {n['at']} for {n['min']} min ({n['room'].lower()})"
        if len(m["night"]) > 1:
            night += f" and {len(m['night']) - 1} more time(s)"
        lines.append(night + (". More than usual ⚠️" if f.get("night_episodes") == "high" else ". Normal."))
    else:
        lines.append("Night: no activity seen.")
    if m["alerts"]:
        lines.append("Alerts: " + "; ".join(f"{a['at']} {a['title']}" for a in m["alerts"]))
    if base["days"] < 7 or base["synthetic"]:
        lines.append(f"(Baseline: {base['days']} day(s){', includes demo data' if base['synthetic'] else ''}.)")
    text = "\n".join(lines)
    note = ai.daily_note(text) if ai else None
    return (text + (f"\n\n{note}" if note else "")), m, base, f


def send(day):
    """Build the report, send it (Telegram + HA) and mark the day as sent. Returns the text."""
    from . import llm, notify, pipeline
    text = report(day, llm.LLM(pipeline.DATA_DIR / "llm"))[0]
    notify.send_text(text)
    (DAILY_DIR / f"{day}.sent").write_text(text, encoding="utf-8")
    return text


def due(now=None):
    """The day whose report should go out now (REPORT_AT, default 21:00, not sent yet), else None."""
    now = now or datetime.now()
    day = now.strftime("%Y-%m-%d")
    if now.strftime("%H:%M") >= os.environ.get("REPORT_AT", "21:00") and not (DAILY_DIR / f"{day}.sent").exists():
        return day
    return None
