"""End to end through the HTTP API on the synthetic clip (runs YOLO on CPU, ~30 s)."""
import subprocess
import sys
import time

import pytest
from fastapi.testclient import TestClient

from conftest import BASE


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    out = tmp_path_factory.mktemp("clip") / "synthetic.mp4"
    subprocess.run([sys.executable, str(BASE / "scripts" / "make_test_clip.py"), str(out)], check=True)
    return out


@pytest.fixture(scope="module")
def api():
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_rejects_bad_uploads(api, tmp_path):
    bad = tmp_path / "notes.mp4"
    bad.write_bytes(b"not a video")
    assert api.post("/api/jobs", files={"file": ("x.mp4", b"x")}, data={}).status_code == 400  # no consent
    r = api.post("/api/jobs", files={"file": ("notes.mp4", bad.read_bytes())}, data={"consent": "yes"})
    assert r.status_code == 400
    r = api.post("/api/jobs", files={"file": ("a.avi", b"x")}, data={"consent": "yes"})
    assert r.status_code == 400


def test_upload_process_and_result(api, clip):
    from app import main
    main._hits.clear()
    r = api.post("/api/jobs", files={"file": ("clip.mp4", clip.read_bytes(), "video/mp4")}, data={"consent": "yes"})
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]
    for _ in range(300):
        st = api.get(f"/api/jobs/{job_id}").json()
        if st["status"] in ("done", "error"):
            break
        time.sleep(1)
    assert st["status"] == "done", st

    res = api.get(f"/api/results/{job_id}").json()
    assert [e["type"] for e in res["events"]] == ["fall"]
    assert res["report"]["ai"] is False and res["report"]["note"] == "AI text unavailable"
    assert [p["label"] for p in res["people"]] == ["Resident", "Visitor/Caregiver"]
    assert api.get(res["media_base"] + res["video"]).status_code == 200
    assert api.get(res["media_base"] + res["events"][0]["thumb"]).headers["content-type"] == "image/jpeg"
    assert api.get(f"/r/{job_id}").status_code == 200
    assert not list((main.jobs.uploads).iterdir())  # raw upload removed after processing


def test_unknown_and_listing(api):
    assert api.get("/api/results/doesnotexist").status_code == 404
    assert api.get("/r/../../etc").status_code == 404
    assert api.get("/media/results/").status_code == 404  # no directory listing
    assert api.get("/healthz").json()["ok"]


def test_rate_limit(api):
    from app import main
    main._hits.clear()
    codes = [api.post("/api/jobs", files={"file": ("x.avi", b"x")}, data={"consent": "yes"}).status_code
             for _ in range(main.UPLOADS_PER_HOUR + 1)]
    assert codes[-1] == 429 and 429 not in codes[:-1]


def test_batch_review_local_only(api, tmp_path):
    import json
    from app import main
    src = tmp_path / "dataset" / "set1"  # set1 run (cam1 done, cam2 pending); set2 never run
    src.mkdir(parents=True)
    (src / "cam1.avi").write_bytes(b"")
    (src / "cam2.avi").write_bytes(b"")
    (tmp_path / "dataset" / "set2").mkdir()
    (tmp_path / "dataset" / "set2" / "cam1.avi").write_bytes(b"")
    g = main.BATCH / "set1"
    (g / "cam1").mkdir(parents=True)
    (g / "cam1" / "result.json").write_text('{"duration": 1, "events": []}')
    (g / "summary.csv").write_text("file,falls,expected_fall,verdict\ncam1.avi,0,,\n")
    (g / "batch.json").write_text(json.dumps({"source": str(src)}))
    assert api.get("/api/batch").status_code == 404  # TestClient's host is not loopback
    assert api.get("/api/results/batch/set1/cam1").status_code == 404
    local = TestClient(main.app, client=("127.0.0.1", 5000))
    d = local.get("/api/batch").json()
    clips = {c["file"]: c for c in d["groups"][0]["clips"]}
    assert clips["cam1.avi"]["status"] == "to review" and clips["cam2.avi"]["status"] == "not processed"
    assert clips["cam1.avi"]["video"] == "/media/batch/set1/cam1/annotated.mp4"
    assert [t["name"] for t in d["todo"]] == ["set2"]
    assert local.get("/api/results/batch/set1/cam1").json()["media_base"] == "/media/batch/set1/cam1/"
    assert local.get("/media/batch/set1/cam1/result.json").status_code == 200
    assert local.get("/media/batch/..%2F..%2Fjobs").status_code == 404
    r = local.post("/api/batch/review", json={"group": "set1", "file": "cam1.avi", "fall": 1, "note": "slid off chair"})
    assert r.status_code == 200
    d = local.get("/api/batch").json()
    assert d["groups"][0]["clips"][0]["review"] == {"fall": 1, "note": "slid off chair"}
    assert d["groups"][0]["clips"][0]["status"] == "reviewed"
    assert d["history"][0]["what"] == "reviewed" and "real fall" in d["history"][0]["detail"]
    assert local.post("/api/batch/review", json={"group": "../jobs", "file": "x", "fall": 1}).status_code == 404


def test_history_lists_and_deletes(api):
    from app import main
    assert api.get("/api/history").status_code == 404     # LAN only
    api = TestClient(main.app, client=("127.0.0.1", 5000))
    jid, path = main.jobs.new_upload(".mp4")
    main.jobs._write({"id": jid, "status": "done", "progress": 100, "stage": "Done", "created": 1.0,
                      "file": path.name, "name": "grandma.mp4"})
    h = api.get("/api/history").json()
    assert any(u["id"] == jid and u["name"] == "grandma.mp4" for u in h["uploads"])
    assert api.delete(f"/api/history/upload/{jid}").status_code == 200
    assert not any(u["id"] == jid for u in api.get("/api/history").json()["uploads"])
    assert api.delete(f"/api/history/upload/{jid}").status_code == 404
    assert api.delete("/api/history/day/not-a-date").status_code == 400


def test_day_dvr_clips(tmp_path, monkeypatch):
    from app import main
    monkeypatch.setattr(main, "DVR_DIR", tmp_path)
    (tmp_path / "2026-09-23" / "porch").mkdir(parents=True)
    (tmp_path / "2026-09-23" / "porch" / "081500.mp4").write_bytes(b"0123456789")
    local = TestClient(main.app, client=("127.0.0.1", 5000))
    assert local.get("/api/day/2026-09-23").json()["clips"] == {"porch": ["081500"]}
    r = local.get("/media/dvr/2026-09-23/porch/081500.mp4", headers={"Range": "bytes=2-4"})
    assert r.status_code == 206 and r.content == b"234"  # seeking in the player needs range requests
    assert local.get("/media/dvr/2026-09-23/porch/999999.mp4").status_code == 404
    assert local.get("/media/dvr/2026-09-23/..%2Fx/081500.mp4").status_code == 404
    assert TestClient(main.app, client=("8.8.8.8", 5000)).get("/media/dvr/2026-09-23/porch/081500.mp4").status_code == 404  # off the LAN: hidden
