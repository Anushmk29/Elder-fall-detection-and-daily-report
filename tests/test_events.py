from conftest import kp_person, sample

from app import events, tracks
from app.pipeline import load_rules

RULES = load_rules()
META = {"width": 1000, "height": 1000, "fps": 6, "duration": 20, "clock_start": 9 * 3600}
STAND_BOX = [450, 300, 550, 700]                              # hips at ~y 500
LIE_KP = kp_person(head=(300, 680), shoulder=(350, 680), hip=(450, 685), knee=(520, 685), ankle=(600, 685))
LIE_BOX = [290, 650, 610, 710]                                # hips at ~y 685: dropped ~0.46 body heights


def stand_kp():
    return kp_person(head=(500, 320), shoulder=(500, 380), hip=(500, 500), knee=(500, 600), ankle=(500, 700))


def make_track(postures):
    """postures: list of (t_from, t_to, 'standing'|'lying') at 6 fps."""
    out = []
    for a, b, p in postures:
        for n in range(round(a * 6), round(b * 6)):
            out.append(sample(n / 6, p, STAND_BOX if p == "standing" else LIE_BOX,
                              stand_kp() if p == "standing" else LIE_KP))
    frames = [{"i": s["i"], "t": s["t"], "people": [s]} for s in out]
    return tracks.build(frames, META, RULES)


def test_fall_detected_with_got_up_time():
    trk = make_track([(0, 4, "standing"), (4, 9, "lying"), (9, 12, "standing")])
    res = events.analyze(trk, META, RULES)
    falls = [e for e in res["events"] if e["type"] == "fall"]
    assert len(falls) == 1
    assert falls[0]["severity"] == "alert" and falls[0]["who"] == "Resident"
    assert "got up in 5 s" in falls[0]["detail"]
    assert falls[0]["clock"] == "09:00:04"


def test_brief_dip_is_not_a_fall():
    trk = make_track([(0, 4, "standing"), (4, 5, "lying"), (5, 10, "standing")])  # down only 1 s
    assert not [e for e in events.analyze(trk, META, RULES)["events"] if e["type"] == "fall"]


def test_slow_lie_down_is_not_a_fall():
    s = [sample(t / 6, "standing", STAND_BOX, stand_kp()) for t in range(0, 12)]
    s += [sample(t / 6, "sitting", STAND_BOX, stand_kp()) for t in range(12, 40)]   # sits for ~4.7 s first
    s += [sample(t / 6, "lying", LIE_BOX, LIE_KP) for t in range(40, 80)]
    falls = events.detect_falls(s, RULES["fall"])
    assert falls == []


def test_lying_outside_rest_zone_and_resident_role():
    rules = {**RULES, "lying_outside_rest_s": 3}
    trk = make_track([(0, 1, "standing"), (1, 8, "lying")])
    visitor = make_track([(0, 2, "standing")])[0]
    visitor["id"] = 2
    res = events.analyze(trk + [visitor], META, rules)
    assert res["people"][0]["label"] == "Resident"
    assert any(e["type"] == "lying_outside_rest" for e in res["events"])
