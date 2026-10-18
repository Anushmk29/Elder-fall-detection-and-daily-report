import os
import sys
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="eldercare-test-")  # before app modules read it
os.environ.pop("OPENROUTER_API_KEY", None)


def kp_person(head=(100, 20), shoulder=(100, 50), hip=(100, 110), knee=(100, 160), ankle=(100, 210)):
    """17 COCO keypoints built from a few joints (left/right identical)."""
    pts = [head] * 5 + [shoulder] * 2 + [shoulder] * 4 + [hip] * 2 + [knee] * 2 + [ankle] * 2
    return [[x, y, 0.9] for x, y in pts]


def sample(t, posture, box, kp=None, i=None):
    return {"i": i if i is not None else int(t * 6), "t": t, "id": 1, "box": box, "kp": kp or kp_person(),
            "conf": 0.9, "posture": posture}
