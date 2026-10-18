"""Daily routine from the live event log, compared with the previous 7 days.

metrics(day)   one day's events-YYYY-MM-DD.jsonl -> wake/bed time, kitchen visits, bathroom, active time, longest
               rest, night activity, falls; saved as data/live/daily/YYYY-MM-DD.json
baseline(day)  mean/min/max of up to 7 earlier daily files
report(day)    the family message, e.g. "Mrs. Rao — Thu 16 Oct / Woke 6:52 (usual 6:30–7:15) ✅ ..."

Rooms are recognised by name (the camera's `room:` in cameras.yaml): kitchen*, bath*/toilet*/wc, bed*.
"""
import json
import os
from datetime import datetime, timedelta

from . import live

DAILY_DIR = live.LIVE_DIR / "daily"
MERGE_GAP_S = 300        # visits to the same room this close together are one visit (occlusion, brief exits)
MEAL_MIN_S = 180         # a kitchen visit this long counts as a meal
NIGHT = (23, 5)          # night-time hours [23:00, 05:00)
ACTIVE = {"standing", "bending"}
NUMERIC = ["wake_min", "bed_min", "kitchen_visits", "bathroom_count", "bathroom_longest_min", "active_min",
           "rest_longest_min", "night_episodes"]


def _kind(room):
    r = (room or "").lower()
    return ("kitchen" if "kitchen" in r else "bathroom" if any(w in r for w in ("bath", "toilet", "wc"))
            else "bedroom" if "bed" in r else "other")


def _hm(ts):
    return datetime.fromtimestamp(ts).strftime("%H:%M")


def _minute_of_day(ts):
    d = datetime.fromtimestamp(ts)
    return d.hour * 60 + d.minute


def visits(evts):
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
    falls = [e for e in evts if e["type"] in ("fall_alert", "fall_recovered")]
    m = {
        "date": day,
        "observed": bool(evts),
        "wake_min": _minute_of_day(morning[0][2]) if morning else None,
        "bed_min": _minute_of_day(evening[-1][2]) if evening else None,
        "kitchen_visits": len(kitchen),
        "meals": [{"label": _meal(a), "at": _hm(a), "min": round((b - a) / 60)} for a, b in kitchen],
        "bathroom_count": len(bath),
        "bathroom_longest_min": round(max((b - a for a, b in bath), default=0) / 60),
        "active_min": round(sum(b - a for p, _, a, b in spans if p in ACTIVE) / 60),
        "rest_longest_min": round((rest[3] - rest[2]) / 60) if rest else 0,
        "rest_longest": {"room": rest[1], "from": _hm(rest[2]), "to": _hm(rest[3]),
                         "posture": rest[0]} if rest else None,
        "night_episodes": len(night),
        "night": [{"at": _hm(a), "min": round((b - a) / 60), "room": room} for room, a, b in night],
        "falls": [{"at": e["clock"], "type": e["type"], "detail": e["detail"]} for e in falls],
        "alerts": [{"at": e["clock"], "title": e["title"]} for e in evts if e["severity"] == "alert"],
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
        first.append(f"Woke {_clock(m['wake_min'])}{usual}{mark('wake_min')}")
    meals = ", ".join(f"{x['label']} {x['at']}" for x in m["meals"])
    first.append(f"{m['kitchen_visits']} kitchen visit{'s' if m['kitchen_visits'] != 1 else ''}"
                 + (f" ({meals})" if meals else "") + mark("kitchen_visits"))
    lines.append(" · ".join(first))

    active = f"Active time {_dur(m['active_min'])}"
    if "active_min" in bm and bm["active_min"]["mean"]:
        pct = round(100 * (m["active_min"] - bm["active_min"]["mean"]) / bm["active_min"]["mean"])
        active += f" ({'↓' if pct < 0 else '↑'}{abs(pct)}% vs. 7-day average)"
    lines.append(f"Bathroom {m['bathroom_count']}×, longest {m['bathroom_longest_min']} min{mark('bathroom_count')}"
                 f" · {active}{mark('active_min')}")

    third = ["No falls." if not m["falls"] else
             "; ".join(f"Fall at {x['at']}: {x['detail'].rstrip('.')}" for x in m["falls"]) + ". ⚠️"]
    r = m["rest_longest"]
    if r and m["rest_longest_min"] >= 30:
        longer = " which is longer than usual" if f.get("rest_longest_min") == "high" else ""
        third.append(f"{'Lay' if r['posture'] == 'lying' else 'Sat'} in the {(r['room'] or 'living room').lower()}"
                     f" {r['from']}–{r['to']}{',' if longer else ''}{longer}.")
    lines.append(" ".join(third))

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
