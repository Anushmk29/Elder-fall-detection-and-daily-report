import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from eval_staged import score  # noqa: E402


def row(kind, a=None, b=None, note=""):
    return {"kind": kind, "start_s": a, "end_s": b, "note": note, "take": ""}


def test_score_hits_false_alarms_and_alerts():
    rows = [row("fall_stay", 10, 12), row("lookalike", 40, 45, "sofa flop"), row("fall", 70, 72)]
    found = [{"t": 11.0, "got_up_s": None},      # stay-down fall, never got up -> hit + alert
             {"t": 12.5, "got_up_s": 3},         # second detection of the same fall: not counted twice
             {"t": 42.0, "got_up_s": 4},         # during the look-alike -> false possible fall, no alert
             {"t": 71.0, "got_up_s": 8},         # got up after 8 s -> hit, no alert
             {"t": 100.0, "got_up_s": None}]     # nothing labelled, down till the end -> false ALERT
    r = score(found, rows, duration=140, no_rec=30)
    assert set(r["hits"]) == {0, 2}
    assert r["hits"][0]["alerts"] and not r["hits"][2]["alerts"]
    assert [(e["t"], c, e["alerts"]) for e, c in r["false"]] == [(42.0, "sofa flop", False), (100.0, "other", True)]
