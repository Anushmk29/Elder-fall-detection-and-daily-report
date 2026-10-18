from conftest import kp_person, sample

from app import events, tracks
from app.pipeline import load_rules

SQUARE = [[0.2, 0.2], [0.6, 0.2], [0.6, 0.6], [0.2, 0.6]]


def test_point_in_polygon():
    assert events.in_poly(0.4, 0.4, SQUARE)
    assert not events.in_poly(0.7, 0.4, SQUARE)
    assert not events.in_poly(0.4, 0.1, SQUARE)
    tri = [[0, 0], [1, 0], [0, 1]]
    assert events.in_poly(0.2, 0.2, tri) and not events.in_poly(0.8, 0.8, tri)


def test_dwell_time_and_room_timeline():
    rules = load_rules()
    meta = {"width": 1000, "height": 1000, "fps": 6, "duration": 10, "clock_start": 0}
    kp = kp_person(ankle=(0, 0))
    kp[15][2] = kp[16][2] = 0.0  # no ankles -> foot point = box bottom centre
    left = [sample(n / 6, "standing", [150, 300, 250, 700], kp) for n in range(0, 30)]     # 5 s in Kitchen
    right = [sample(n / 6, "standing", [750, 300, 850, 700], kp) for n in range(30, 48)]   # 3 s in Living room
    frames = [{"i": s["i"], "t": s["t"], "people": [s]} for s in left + right]
    res = events.analyze(tracks.build(frames, meta, rules), meta, rules)
    zones = res["people"][0]["zones"]
    assert abs(zones["Kitchen"] - 5) < 0.4 and abs(zones["Living room"] - 3) < 0.4
    assert [s["zone"] for s in res["timeline"]] == ["Kitchen", "Living room"]
