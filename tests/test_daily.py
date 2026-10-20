import json
from datetime import datetime

from app import daily, live

DAY = "2026-10-16"


def ts(hm):
    return datetime.strptime(f"{DAY} {hm}", "%Y-%m-%d %H:%M").timestamp()


def ev(hm, typ, cam, room=None, posture=None, severity="info", **kw):
    return {"ts": ts(hm), "clock": f"{hm}:00", "camera": cam, "room": room, "type": typ, "severity": severity,
            "title": typ, "detail": kw.get("detail", ""), "posture": posture, "snapshot": None}


def write_day(evts):
    live.LIVE_DIR.mkdir(parents=True, exist_ok=True)
    (live.LIVE_DIR / f"events-{DAY}.jsonl").write_text("\n".join(json.dumps(e) for e in evts), encoding="utf-8")


def test_routine_metrics_and_report():
    write_day([
        ev("02:40", "zone_enter", "bath", "Bathroom", "standing"), ev("02:48", "zone_exit", "bath", "Bathroom"),
        ev("06:52", "zone_enter", "bed", "Bedroom"), ev("06:52", "posture_change", "bed", "Bedroom", "standing"),
        ev("06:55", "zone_exit", "bed", "Bedroom"),
        ev("07:20", "zone_enter", "kit", "Kitchen"), ev("07:21", "zone_exit", "kit", "Kitchen"),
        ev("07:23", "zone_enter", "kit", "Kitchen"), ev("07:40", "zone_exit", "kit", "Kitchen"),  # merged: 1 meal
        ev("09:00", "zone_enter", "bath", "Bathroom"), ev("09:12", "zone_exit", "bath", "Bathroom"),
        ev("12:45", "zone_enter", "kit", "Kitchen"), ev("13:10", "zone_exit", "kit", "Kitchen"),
        ev("14:00", "zone_enter", "liv", "Living room"), ev("14:00", "posture_change", "liv", "Living room", "lying"),
        ev("16:05", "posture_change", "liv", "Living room", "standing"), ev("16:10", "zone_exit", "liv", "Living room"),
        ev("21:30", "zone_enter", "bed", "Bedroom"), ev("21:40", "posture_change", "bed", "Bedroom", "lying"),
    ])
    m = daily.metrics(DAY)
    assert m["wake_min"] == 6 * 60 + 52 and m["bed_min"] == 21 * 60 + 40
    assert m["kitchen_visits"] == 2 and [x["label"] for x in m["meals"]] == ["breakfast", "lunch"]
    assert m["bathroom_count"] == 2 and m["bathroom_longest_min"] == 12
    assert m["rest_longest"] == {"room": "Living room", "from": "14:00", "to": "16:05", "posture": "lying"}
    assert m["night_episodes"] == 1 and m["night"][0]["at"] == "02:40"

    for k, active in enumerate([400, 420, 380], 1):  # 3 baseline days, all much more active than today
        (daily.DAILY_DIR / f"2026-10-1{6 - k}.json").write_text(json.dumps(
            {"observed": True, "wake_min": 410, "kitchen_visits": 2, "active_min": active, "bathroom_count": 2,
             "rest_longest_min": 40, "night_episodes": 1}), encoding="utf-8")
    text, m, base, flags = daily.report(DAY)
    assert base["days"] == 3 and flags["active_min"] == "low" and flags["kitchen_visits"] == "ok"
    assert flags["rest_longest_min"] == "high"
    assert "Woke 6:52 (usual 6:50–6:50) ✅" in text and "Lay in the living room 14:00–16:05" in text
    assert "longer than usual" in text and "Night: up at 02:40 for 8 min (bathroom)" in text


def test_a_house_without_kitchen_or_bathroom_cameras_gets_a_path_not_zeros():
    write_day([
        ev("07:10", "zone_enter", "liv", "Living room"), ev("07:10", "posture_change", "liv", "Living room", "sitting"),
        ev("08:50", "posture_change", "liv", "Living room", "standing"), ev("09:00", "zone_exit", "liv", "Living room"),
        ev("09:01", "zone_enter", "st", "Outside stairs"), ev("09:02", "zone_exit", "st", "Outside stairs"),
        ev("09:03", "zone_enter", "gar", "Garden"), ev("09:40", "zone_exit", "gar", "Garden"),  # one trip out
        ev("09:42", "zone_enter", "liv", "Living room"), ev("12:00", "zone_exit", "liv", "Living room"),
        ev("11:00", "zone_enter", "por", "Porch"), ev("11:05", "zone_exit", "por", "Porch"),  # while she's inside
    ])
    m = daily.metrics(DAY)
    assert m["kitchen_visits"] is None and m["bathroom_count"] is None
    assert [p["room"] for p in m["places"]][:2] == ["Living room", "Garden"]
    assert m["places"][0]["postures"]["sitting"] == 100
    assert m["outings"][0] == {"from": "09:01", "to": "09:40", "min": 39, "rooms": ["Outside stairs", "Garden"]}
    assert m["together"] == [{"at": "11:00", "rooms": ["Living room", "Porch"]}]
    text = daily.report(DAY)[0]
    assert "kitchen" not in text and "Bathroom" not in text and "First up 8:50" in text
    assert "Went outside 2×: 09:01–09:40 (outside stairs, garden)" in text and "Someone else was around" in text


def test_ai_reviews_give_the_postures_and_activities():
    def review(hm, cam, room, postures, activity):
        return {**ev(hm, "ai_review", cam, room), "title": activity,
                "ai": {"postures": postures, "activity": activity, "fall": "NO", "span_s": 30}}
    write_day([
        ev("11:20", "zone_enter", "side", "Side path"), ev("11:20", "posture_change", "side", "Side path", "lying"),
        review("11:20", "side", "Side path", {"bending": 12, "standing": 3}, "sweeping the path"),
        review("11:27", "gar", "Garden", {"bending": 15}, "Sweeping the path"),
        {**ev("11:26", "fall_dismissed", "side", "Side path"), "detail": "AI: sweeping"},
        ev("11:40", "zone_exit", "side", "Side path"),
    ])
    m = daily.metrics(DAY)
    assert m["active_min"] == 1 and m["lying_min"] == 0  # 30 frames x 2 s; the pose model's "lying" is ignored
    assert m["activities"] == [{"activity": "sweeping the path", "min": 1.0,
                                "rooms": {"Side path": "11:20", "Garden": "11:27"}}]
    text = daily.report(DAY)[0]
    assert "Seen doing (AI): sweeping the path (Side path 11:20, Garden 11:27)." in text
    assert "(AI-checked)" in text and "No falls." in text and "AI check: 1 possible fall looked at, not a fall." in text
