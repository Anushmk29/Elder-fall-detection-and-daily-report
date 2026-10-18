import numpy as np

from conftest import sample
from test_events import LIE_BOX, LIE_KP, STAND_BOX, stand_kp

from app import live

FRAME = np.zeros((1000, 1000, 3), np.uint8)


class FakeEngine:
    def __init__(self):
        self.events = []

    def emit(self, cam, typ, severity, title, detail, **kw):
        self.events.append((typ, severity, kw.get("t")))

    def types(self):
        return [e[0] for e in self.events]


def run_camera(postures, cfg=None):
    """Feed (t_from, t_to, posture) spans at 6 fps through a Camera's rolling-window evaluation, as _read does."""
    eng = FakeEngine()
    cam = live.Camera({"name": "test", "source": "x.mp4", "room": "Living room", **(cfg or {})}, eng, model=None)
    cam.clock_start, cam.t0_wall = 9 * 3600, 0
    n, last_eval = 0, -1.0
    for a, b, p in postures:
        for k in range(round(a * 6), round(b * 6)):
            t = k / 6
            s = sample(t, p, STAND_BOX if p == "standing" else LIE_BOX, stand_kp() if p == "standing" else LIE_KP)
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
