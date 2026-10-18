import numpy as np

from conftest import sample
from test_events import LIE_BOX, LIE_KP, STAND_BOX, stand_kp

from app import live

FRAME = np.zeros((1000, 1000, 3), np.uint8)


class FakeEngine:
    def __init__(self):
        self.events, self.asked = [], []

    def emit(self, cam, typ, severity, title, detail, **kw):
        self.events.append((typ, severity, kw.get("t")))
        return {"type": typ, "severity": severity, "title": title, "detail": detail, "snapshot": None}

    def saw(self, cam, places, t):
        pass

    def second_opinion(self, cam, ev):
        self.asked.append(ev["type"])

    def types(self):
        return [e[0] for e in self.events]


def run_camera(postures, cfg=None, wiggle=False):
    """Feed (t_from, t_to, posture) spans at 6 fps through a Camera's rolling-window evaluation, as _read does."""
    eng = FakeEngine()
    cam = live.Camera({"name": "test", "source": "x.mp4", "room": "Living room", **(cfg or {})}, eng, model=None)
    cam.clock_start, cam.t0_wall = 9 * 3600, 0
    n, last_eval = 0, -1.0
    for a, b, p in postures:
        for k in range(round(a * 6), round(b * 6)):
            t = k / 6
            kp = stand_kp() if p == "standing" else [[x + (25 * (k % 2) if wiggle else 0), y, c] for x, y, c in LIE_KP]
            s = sample(t, p, STAND_BOX if p == "standing" else LIE_BOX, kp)
            cam.window.append({"i": n, "t": t, "people": [{k2: s[k2] for k2 in ("id", "box", "kp", "conf", "posture")}]})
            n += 1
            if t - last_eval >= 1.0:
                last_eval = t
                cam._evaluate(t, FRAME)
    return eng


def test_fall_without_recovery_alerts_once_after_30s():
    eng = run_camera([(0, 4, "standing"), (4, 45, "lying")])
    assert eng.types().count("fall_suspected") == 1
    assert eng.types().count("fall_alert") == 1
    alert = next(e for e in eng.events if e[0] == "fall_alert")
    assert alert[1] == "alert" and 33 <= alert[2] <= 36  # fall at ~4 s + 30 s no-recovery
    assert "fall_recovered" not in eng.types()


def test_fall_with_recovery_is_not_an_alert():
    eng = run_camera([(0, 4, "standing"), (4, 12, "lying"), (12, 50, "standing")])
    assert eng.types().count("fall_suspected") == 1
    assert eng.types().count("fall_recovered") == 1
    assert "fall_alert" not in eng.types()


def test_room_from_camera_and_posture_changes_logged():
    eng = run_camera([(0, 10, "standing")])
    assert eng.types()[:2] == ["zone_enter", "posture_change"]


def test_source_kinds():
    assert live.is_file_source("videos/a.mp4")
    assert not live.is_file_source("0") and not live.is_file_source("rtsp://cam/1")
    assert not live.is_file_source("http://192.168.1.5:8080/video")


def sev(eng, typ):
    return [e[1] for e in eng.events if e[0] == typ]


def test_on_the_floor_not_moving_is_an_alert():
    eng = run_camera([(0, 70, "lying")])  # no fall seen (already down), still for > 60 s
    assert sev(eng, "lying_outside_rest") == ["alert"] and "fall_suspected" not in eng.types()


def test_on_the_floor_moving_is_monitor_and_asks_for_a_second_opinion():
    eng = run_camera([(0, 70, "lying")], wiggle=True)  # e.g. stretching on a mat
    assert sev(eng, "lying_outside_rest") == ["warn"] and eng.asked == ["lying_outside_rest"]


def test_floor_after_a_fall_alert_is_not_pushed_twice():
    eng = run_camera([(0, 4, "standing"), (4, 70, "lying")])
    assert sev(eng, "fall_alert") == ["alert"] and sev(eng, "lying_outside_rest") == ["info"]


def test_watch_bathroom_and_not_seen():
    from datetime import datetime
    noon = datetime(2026, 10, 18, 12, 0).timestamp()
    w = live.Watch({"bathroom_s": 1800, "bathroom_alert_s": 2700, "not_seen_s": 14400, "day_hours": [8, 21]})
    w.saw(["Hallway", "Bathroom door"], noon)
    assert w.check(noon + 1000, True) == []
    assert [x[1] for x in w.check(noon + 1900, True)] == ["warn"]
    assert w.check(noon + 2000, True) == []                       # once per episode
    assert [x[1] for x in w.check(noon + 2800, True)] == ["alert"]
    w.saw(["Living room"], noon + 3000)                           # came out
    assert w.check(noon + 3000 + 14000, True) == []
    assert [x[0] for x in w.check(noon + 3000 + 14500, True)] == ["not_seen"]
    night = live.Watch({"not_seen_s": 60, "day_hours": [8, 21]})
    night.saw(["Living room"], datetime(2026, 10, 18, 23, 0).timestamp())
    assert night.check(datetime(2026, 10, 18, 23, 30).timestamp(), True) == []   # asleep, not missing
    assert live.Watch({"not_seen_s": 60}).check(noon, True) == []                # never seen yet


def test_long_time_on_the_sofa_warns_once():
    sofa = [{"name": "Sofa", "kind": "rest", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}]
    rules = {"zones": sofa}
    eng = FakeEngine()
    cam = live.Camera({"name": "c", "source": "x.mp4"}, eng, model=None)
    cam.rules = {**cam.rules, **rules, "live": {**cam.rules["live"], "rest_long_s": 60, "rest_long_hours": [0, 24]}}
    cam.t0_wall = 0
    for t in range(0, 200, 10):
        cam._rest(t, {"zones": ["Sofa"]})
    assert sev(eng, "rest_long") == ["warn"]
