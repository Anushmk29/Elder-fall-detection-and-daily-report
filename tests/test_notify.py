import json

import httpx

from app import notify

EV = {"type": "fall_alert", "severity": "alert", "title": "FALL", "detail": "Down 31 s", "clock": "10:00:00",
      "camera": "living", "room": "Living room", "snapshot": "1-living-fall_alert.jpg"}


def client(seen):
    def handler(req):
        seen.append(req)
        return httpx.Response(200, json={"ok": True})
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_nothing_sent_without_config(monkeypatch):
    for k in ("HA_URL", "HA_TOKEN", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        monkeypatch.delenv(k, raising=False)
    seen = []
    notify.send(EV, client=client(seen))
    notify.ha_state("eldercare_room", "Kitchen", client=client(seen))
    assert seen == []


def test_ha_webhook_and_telegram_photo(monkeypatch, tmp_path):
    monkeypatch.setenv("HA_URL", "http://ha:8123/")
    monkeypatch.setenv("PUBLIC_BASE_URL", "http://laptop:8000")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    (tmp_path / EV["snapshot"]).write_bytes(b"\xff\xd8jpeg")
    seen = []
    notify.send(EV, snapshot_dir=tmp_path, client=client(seen))
    ha, tg = seen
    assert str(ha.url) == "http://ha:8123/api/webhook/eldercare"
    body = json.loads(ha.content)
    assert body["image"] == "http://laptop:8000/live/snapshots/1-living-fall_alert.jpg" and "FALL" in body["message"]
    assert str(tg.url).endswith("/bot123:abc/sendPhoto") and b"jpeg" in tg.content


def test_failure_is_logged_without_token(monkeypatch, caplog):
    monkeypatch.delenv("HA_URL", raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:secret")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    bad = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    notify.send({**EV, "snapshot": None}, client=bad)  # must not raise
    assert "secret" not in caplog.text and "failed" in caplog.text
