"""Rule-based posture from COCO-17 keypoints: standing / sitting / bending / lying."""
import math

KP_CONF = 0.3
SHOULDERS, HIPS, KNEES = (5, 6), (11, 12), (13, 14)

DEFAULTS = {
    "lying_torso": 60,      # torso this far from vertical (deg) -> lying candidate
    "lying_aspect": 1.4,    # box w/h when no torso is visible
    "lying_min_aspect": 0.9,
    "lying_torso_wide": 45, # ...or this far from vertical when the box is at least lying_aspect wide
    "bend_torso": 30,
    "sit_thigh": 50,        # thigh this far from vertical -> sitting
    "sit_thigh_ratio": 0.5, # thigh shorter than this x torso (foreshortened, facing camera) -> sitting
    "legs_hidden": "standing",  # upright torso, knees out of view: at a counter (standing) or a table (sitting)
}


def _mid(kp, pair):
    pts = [kp[i] for i in pair if kp[i][2] >= KP_CONF]
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def _from_vertical(top, bottom):
    """0 deg = vertical, 90 deg = horizontal."""
    return math.degrees(math.atan2(abs(top[0] - bottom[0]), abs(top[1] - bottom[1]) + 1e-6))


def features(box, kp):
    x1, y1, x2, y2 = box
    sh, hip, knee = _mid(kp, SHOULDERS), _mid(kp, HIPS), _mid(kp, KNEES)
    f = {"aspect": (x2 - x1) / max(y2 - y1, 1e-6), "torso": None, "thigh": None, "thigh_ratio": None}
    if sh and hip:
        f["torso"] = _from_vertical(sh, hip)
    if hip and knee:
        f["thigh"] = _from_vertical(hip, knee)
        if sh:
            f["thigh_ratio"] = math.dist(hip, knee) / max(math.dist(sh, hip), 1e-6)
    return f


def classify(box, kp, th=None):
    th = {**DEFAULTS, **(th or {})}
    f = features(box, kp)
    torso, thigh = f["torso"], f["thigh"]
    if torso is None:
        # ponytail: box-shape-only fallback when the torso is hidden
        return "lying" if f["aspect"] >= th["lying_aspect"] else "standing"
    if torso >= th["lying_torso"] and (f["aspect"] >= th["lying_min_aspect"] or (thigh or 0) >= th["lying_torso"]):
        return "lying"
    if torso >= th["lying_torso_wide"] and f["aspect"] >= th["lying_aspect"]:
        return "lying"  # high/fisheye cameras foreshorten a body on the floor: tilted torso in a wide box
    if torso >= th["bend_torso"]:
        return "bending"
    if thigh is None:
        return th["legs_hidden"]  # ponytail: per-camera guess, the pose alone can't tell
    if thigh >= th["sit_thigh"] or (f["thigh_ratio"] or 1) < th["sit_thigh_ratio"]:
        return "sitting"
    return "standing"
