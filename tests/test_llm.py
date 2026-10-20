import json

import httpx

from app import llm

FACTS = {"glance": [{"label": "Observed", "value": "13 s"}, {"label": "Active", "value": "7 s"}],
         "events": [{"clock": "09:00:05", "severity": "alert", "title": "Fall detected", "detail": "got up in 6 s"}]}


def client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def ok_handler(calls):
    def h(req):
        calls.append(json.loads(req.content))
        assert req.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(200, json={"choices": [{"message": {"content": "All fine today."}}],
                                         "usage": {"total_tokens": 100, "cost": 0.01}})
    return h


def test_report_uses_llm_and_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    calls = []
    ai = llm.LLM(tmp_path, client(ok_handler(calls)))
    r1, r2 = ai.report(FACTS), ai.report(FACTS)
    assert r1 == r2 and r1["ai"] and r1["text"] == "All fine today."
    assert len(calls) == 1  # second call served from disk cache
    assert abs(ai.spent_today() - 0.01) < 1e-9


def test_caption_sends_image(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    calls = []
    strip = tmp_path / "s.jpg"
    strip.write_bytes(b"\xff\xd8fakejpeg")
    c = llm.LLM(tmp_path, client(ok_handler(calls))).caption(strip, "Resident: walking")
    assert c["ai"]
    parts = calls[0]["messages"][0]["content"]
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_fallback_without_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    r = llm.LLM(tmp_path, client(lambda req: 1 / 0)).report(FACTS)
    assert not r["ai"] and r["note"] == llm.UNAVAILABLE
    assert "Things to check:" in r["text"] and "Fall detected" in r["text"]


def test_fallback_on_http_error_and_daily_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    assert not llm.LLM(tmp_path, client(lambda req: httpx.Response(500))).report(FACTS)["ai"]
    monkeypatch.setenv("MAX_LLM_USD_PER_DAY", "0.005")
    ai = llm.LLM(tmp_path, client(ok_handler([])))
    ai._add_spend(0.006)
    assert not ai.report(FACTS)["ai"]


def test_floor_verdict_parses_one_word(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    img = tmp_path / "f.jpg"
    img.write_bytes(b"fakejpeg")
    for answer, want in (("Fallen.", "FALLEN"), ("DELIBERATE", "DELIBERATE"), ("I think she fell", None)):
        def h(req, answer=answer):
            return httpx.Response(200, json={"choices": [{"message": {"content": answer}}], "usage": {"cost": 0}})
        assert llm.LLM(tmp_path / answer[:4], client(h)).floor_verdict(img) == want
    monkeypatch.delenv("OPENROUTER_API_KEY")
    assert llm.LLM(tmp_path, client(h)).floor_verdict(img) is None   # no key: no call, rules decide


def test_ai_fall_check_adds_alert_only_on_fallen(tmp_path):
    from app import pipeline

    class Fake:
        def __init__(self, words):
            self.words = words

        def floor_verdict(self, path, prompt):
            return self.words.pop(0)

    caps = [{"t": t, "clock": f"09:00:{t:02d}", "img": f"strip{t}.jpg"} for t in (1, 5, 9)]
    for words, fell in ((["NO", "FALLEN"], True), (["NO", None], False)):
        an = {"events": [], "glance": [{"label": "Alerts", "value": "0", "note": "0 fall(s)", "tone": "ok"}]}
        pipeline.ai_fall_check(Fake(words), tmp_path, caps, an, {})
        assert bool(an["events"]) == fell
        assert an["glance"][0]["value"] == ("1" if fell else "0")
    assert an["events"] == [] and caps[-1]["img"] == "strip9.jpg"


def test_fall_check_and_review_parse_the_answers(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    answers = ["NO - bent over sweeping", "FALLEN", "maybe?",
               '```json\n{"people": 1, "postures": {"standing": 3, "bending": [1, 2, 3, 4, 5, 6, 7, 8, 9]}, "activity": "Sweeping", "fall": "no"}\n```',
               "not json"]
    sent = []

    def h(req):
        sent.append(json.loads(req.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": answers.pop(0)}}], "usage": {"cost": 0.0003}})
    ai = llm.LLM(tmp_path, client=client(h))
    img = tmp_path / "s.jpg"
    for k in range(3):
        img.write_bytes(b"jpg%d" % k)  # a different image each time, so the cache doesn't answer
        got = ai.fall_check(img)
        assert got == [("NO", "bent over sweeping"), ("FALLEN", ""), (None, "")][k]
    r = ai.review([b"a", b"b"], "Porch")
    assert r["postures"] == {"standing": 3, "walking": 0, "sitting": 0, "bending": 9, "lying": 0}
    assert r["fall"] == "NO" and r["activity"] == "Sweeping"
    assert len([c for c in sent[-1]["messages"][0]["content"] if c["type"] == "image_url"]) == 2
    assert ai.review([b"c"], "Porch") is None
    assert sent[0]["reasoning"] == {"enabled": False}
