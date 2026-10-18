"""Push live events to Home Assistant (webhook + sensor states) and Telegram. Every target is optional: it is
skipped when its environment variables are empty, and a failure is logged, never raised into a camera loop."""
import logging
import os
from pathlib import Path

import httpx

log = logging.getLogger("notify")
ICON = {"alert": "🚨", "warn": "⚠️", "info": "ℹ️"}
_client = httpx.Client(timeout=15)


def _env(name, default=""):
    return os.environ.get(name, default).strip()


def message(ev):
    head = f"{ICON.get(ev.get('severity'), '')} {ev['title']}".strip()
    where = " · ".join(x for x in (ev.get("room") or ev.get("camera"), ev.get("clock")) if x)
    return f"{head}\n{where}\n{ev.get('detail', '')}".strip()


def send(ev, snapshot_dir=None, client=None):
    """ev: an event dict from live.Engine.emit. snapshot_dir: where ev['snapshot'] lives."""
    c = client or _client
    snap = Path(snapshot_dir) / ev["snapshot"] if snapshot_dir and ev.get("snapshot") else None
    for fn in (_ha_webhook, _telegram):
        try:
            fn(c, ev, snap)
        except Exception as e:  # network, HTTP or bad config: the camera loop must keep running
            token = _env("TELEGRAM_BOT_TOKEN")
            msg = str(e).replace(token, "***") if token else str(e)  # httpx errors include the URL
            log.warning("%s failed: %s", fn.__name__, msg)


def send_text(text, client=None):
    """Plain message (the daily report) to Telegram, and to HA as a webhook event of type 'report'."""
    send({"type": "report", "severity": "info", "title": "Daily report", "detail": text}, client=client)


def ha_state(entity, state, attrs=None, client=None):
    """Create/update sensor.<entity> in Home Assistant so its dashboard and logbook show it."""
    url, token = _env("HA_URL").rstrip("/"), _env("HA_TOKEN")
    if not (url and token):
        return
    try:
        (client or _client).post(f"{url}/api/states/sensor.{entity}", headers={"Authorization": f"Bearer {token}"},
                                 json={"state": state, "attributes": attrs or {}}).raise_for_status()
    except Exception as e:
        log.warning("ha_state %s failed: %s", entity, e)


def _ha_webhook(c, ev, snap):
    url = _env("HA_URL").rstrip("/")
    if not url:
        return
    public = _env("PUBLIC_BASE_URL").rstrip("/")
    body = {**ev, "message": message(ev),
            "image": f"{public}/live/snapshots/{ev['snapshot']}" if public and ev.get("snapshot") else ""}
    c.post(f"{url}/api/webhook/{_env('HA_WEBHOOK_ID', 'eldercare')}", json=body).raise_for_status()


def _telegram(c, ev, snap):
    token, chat = _env("TELEGRAM_BOT_TOKEN"), _env("TELEGRAM_CHAT_ID")
    if not (token and chat):
        return
    api = f"https://api.telegram.org/bot{token}"
    text = message(ev)
    if snap and snap.exists():
        r = c.post(f"{api}/sendPhoto", data={"chat_id": chat, "caption": text[:1024]},
                   files={"photo": (snap.name, snap.read_bytes(), "image/jpeg")})
    else:
        r = c.post(f"{api}/sendMessage", data={"chat_id": chat, "text": text[:4096]})
    r.raise_for_status()
