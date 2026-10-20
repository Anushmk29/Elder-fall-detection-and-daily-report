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

    def second_opinion(self, cam, ev, prompt=None, title=None):
        self.asked.append(ev.get("type", "lost"))

    def types(self):
        return [e[0] for e in self.events]


def run_camera(postures, cfg=None, wiggle=False, eng=None):
    """Feed (t_from, t_to, posture) spans at 6 fps through a Camera's rolling-window evaluation, as _read does.
    posture None = nobody detected."""
    eng = eng or FakeEngine()
    cam = live.Camera({"name": "test", "source": "x.mp4", "room": "Living room", **(cfg or {})}, eng, model=None)
    cam.clock_start, cam.t0_wall, cam.jpeg = 9 * 3600, 0, b"jpg"
    n, last_eval = 0, -1.0
    for a, b, p in postures:
        for k in range(round(a * 6), round(b * 6)):
            t = k / 6
            kp = stand_kp() if p == "standing" else [[x + (25 * (k % 2) if wiggle else 0), y, c] for x, y, c in LIE_KP]
            s = sample(t, p, STAND_BOX if p == "standing" else LIE_BOX, kp)
            people = [{k2: s[k2] for k2 in ("id", "box", "kp", "conf", "posture")}] if p else []
            cam.window.append({"i": n, "t": t, "people": people})
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


def test_replayed_clip_is_dated_closes_its_room_and_skips_the_watch(monkeypatch):
    from datetime import datetime

    class Cap:  # a 10 s file at 6 fps
        n = 0
        isOpened = release = lambda self: True  # noqa: E731
        get = lambda self, prop: 6.0 if prop == live.cv2.CAP_PROP_FPS else 60  # noqa: E731
        retrieve = lambda self: (True, FRAME)  # noqa: E731

        def grab(self):
            self.n += 1
            return self.n <= 60

    person = {k: v for k, v in sample(0, "standing", STAND_BOX, stand_kp()).items() if k != "t"}
    monkeypatch.setattr(live, "open_source", lambda src: Cap())
    monkeypatch.setattr(live.perception, "new_tracker", lambda: (None, None))
    monkeypatch.setattr(live.perception, "_detect", lambda *a: [dict(person)])
    start = datetime(2026, 10, 16, 7, 30).timestamp()
    eng = FakeEngine()
    eng.watched = []
    eng.saw = lambda cam, places, t: eng.watched.append(t)
    cam = live.Camera({"name": "kitchen-0730", "source": "k.mp4", "room": "Kitchen", "start_at": start,
                       "realtime": False, "notify": False}, eng, model=None)
    cam._run()
    assert cam.t0_wall == start and cam.clock_start == 7.5 * 3600
    assert eng.types()[0] == "zone_enter" and eng.types()[-1] == "zone_exit" and eng.events[-1][2] > 9
    assert cam.state["finished"] and cam.state["progress"] == 1.0 and cam.state["room"] is None


def test_replays_dont_feed_the_watch_or_push(monkeypatch):
    sent = []
    eng = live.Engine.__new__(live.Engine)
    eng.lock, eng.recent, eng.sender = live.threading.Lock(), live.deque(), lambda *a: sent.append(a)
    eng.watch, eng.cameras = live.Watch({}), {}
    live.LIVE_DIR.mkdir(parents=True, exist_ok=True)
    cam = live.Camera({"name": "c", "source": "x.mp4", "start_at": 1e9, "notify": False}, eng, model=None)
    cam.t0_wall = 1e9
    eng.saw(cam, ["Bathroom"], 5)
    assert eng.watch.bath_since is None
    eng.emit(cam, "fall_alert", "alert", "FALL", "x", t=5)
    assert sent == []


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


def test_lost_from_sight_asks_the_ai_once_only_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "SNAP_DIR", tmp_path)
    spans = [(0, 6, "standing"), (6, 20, None)]  # e.g. fell out of the detector's view
    assert run_camera(spans).asked == []         # VLM off: never asked
    eng = FakeEngine()
    eng.ai, eng.ai_prompts = object(), ("floor", "fall check")
    assert run_camera(spans, eng=eng).asked == ["lost"]
    assert len(list(tmp_path.glob("*-lost.jpg"))) == 1


def test_gru_fall_opens_the_same_alert_path_but_never_pushes_a_quick_get_up(monkeypatch):
    class Fake:  # the model "sees" a fall from 8 s on
        thr, confirm_s = 0.9, 5.0

        def p(self, tr, t):
            return 1.0 if t >= 8 else 0.0
    monkeypatch.setattr(live, "_scorer", Fake())
    eng = run_camera([(0, 8, "standing"), (8, 50, "sitting")])  # slumped: the rules see no fall
    assert eng.types().count("fall_suspected") == 1 and sev(eng, "fall_alert") == ["alert"]
    eng = run_camera([(0, 50, "standing")])                      # model fires, but they're on their feet
    assert "fall_suspected" not in eng.types() and "fall_recovered" not in eng.types()


def test_latest_frame_skips_the_backlog_and_ends_with_the_stream():
    import time

    class Cap:  # 3000 frames, as fast as it can, then the stream ends
        n = 0
        def read(self):
            time.sleep(0.001)
            self.n += 1
            return self.n <= 3000, self.n
        def release(self):
            pass

    lf = live.LatestFrame(Cap())
    assert lf.grab()
    time.sleep(0.5)  # a slow model: the camera ran on meanwhile
    assert lf.grab() and lf.retrieve()[1] > 100  # handed the newest frame, not the next one in line
    while lf.grab():
        pass  # the stream ending stops the reader instead of hanging
    lf.release()


def test_ai_fall_check_stops_only_a_got_up_and_never_a_fall_alert(monkeypatch):
    got_up = [(0, 4, "standing"), (4, 12, "lying"), (12, 50, "standing")]
    for verdict in ("NO", "UNSURE", None, "FALLEN"):
        monkeypatch.setattr(live.Camera, "_ai_check", lambda self, onset, t, v=verdict: (v, "sweeping", b"j"))
        eng = run_camera([(0, 4, "standing"), (4, 45, "lying")])
        assert "fall_alert" in eng.types() and "fall_dismissed" not in eng.types(), verdict  # stays down: always
        eng = run_camera(got_up)
        assert ("fall_recovered" in eng.types()) == (verdict != "NO"), verdict              # only a clear NO stops it
        assert ("fall_dismissed" in eng.types()) == (verdict == "NO")


def test_ai_review_segments_while_someone_is_in_view():
    eng = FakeEngine()
    eng.ai, eng.reviews = object(), []
    eng.review = lambda cam, frames, room: eng.reviews.append([t for t, _ in frames])
    cam = live.Camera({"name": "c", "source": "x.mp4", "room": "Porch"}, eng, model=None)
    person, frame = [{"conf": 0.9}], np.zeros((144, 256, 3), np.uint8)
    for k in range(50 * 6):  # in view 0-40 s, then gone
        cam._collect(frame, person if k < 40 * 6 else [], k / 6)
    cam._send_review()  # what the end of a clip does
    assert [len(r) for r in eng.reviews] == [15, 5] and eng.reviews[1][0] == 30.0  # 30 s at 2 s, then the rest
    assert cam.ai_frames == []


def test_ai_review_is_logged_and_a_fall_the_pose_model_missed_alerts(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "SNAP_DIR", tmp_path)
    eng = live.Engine.__new__(live.Engine)
    eng.lock, eng.recent, eng.sender = live.threading.Lock(), live.deque(), lambda *a: None
    eng.pool = type("Now", (), {"submit": staticmethod(lambda fn: fn())})()

    class AI:
        def review(self, jpegs, room):
            return {"people": 1, "postures": {"lying": len(jpegs)}, "activity": "lying on the ground", "fall": "FALLEN",
                    "why": "not moving"}
    eng.ai = AI()
    live.LIVE_DIR.mkdir(parents=True, exist_ok=True)
    cam = live.Camera({"name": "c", "source": "x.mp4", "start_at": 1.5e9, "notify": False}, eng, model=None)
    cam.t0_wall = 1.5e9
    eng.review(cam, [(0.0, b"a"), (2.0, b"b")], "Garden")
    got = [e for e in live.read_events("2017-07-14") if e["camera"] == "c"]
    assert [e["type"] for e in got] == ["ai_review", "floor_ai_alert"]
    assert got[0]["ai"]["frames"] == 2 and got[0]["ai"]["span_s"] == 4.0 and got[1]["snapshot"]
    cam.falls = [{"t": 1.0, "state": "alert"}]  # the local model already has this fall: no second alert
    eng.review(cam, [(0.0, b"a"), (2.0, b"b")], "Garden")
    assert [e["type"] for e in live.read_events("2017-07-14") if e["camera"] == "c"].count("floor_ai_alert") == 1
