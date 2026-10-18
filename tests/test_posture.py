from conftest import kp_person

from app.posture import classify


def test_standing():
    assert classify([80, 10, 120, 215], kp_person()) == "standing"


def test_sitting_thigh_horizontal():
    kp = kp_person(knee=(150, 112), ankle=(150, 160))
    assert classify([80, 10, 160, 165], kp) == "sitting"


def test_bending():
    kp = kp_person(head=(150, 70), shoulder=(140, 70), hip=(100, 110))  # torso ~45 deg
    assert classify([80, 60, 160, 215], kp) == "bending"


def test_lying():
    kp = kp_person(head=(10, 100), shoulder=(40, 100), hip=(100, 105), knee=(150, 105), ankle=(200, 105))
    assert classify([0, 85, 210, 120], kp) == "lying"


def test_box_fallback_without_keypoints():
    blank = [[0, 0, 0.0]] * 17
    assert classify([0, 0, 200, 60], blank) == "lying"
    assert classify([0, 0, 60, 200], blank) == "standing"


def test_knees_hidden_uses_camera_setting():
    kp = kp_person()
    for i in (13, 14, 15, 16):
        kp[i][2] = 0.0
    assert classify([80, 10, 120, 120], kp) == "standing"  # default: e.g. at a kitchen counter
    assert classify([80, 10, 120, 120], kp, {"legs_hidden": "sitting"}) == "sitting"
