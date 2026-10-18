# Home Assistant for ElderCare (free, runs on the laptop)

Home Assistant is the family-facing side: phone push notifications with a snapshot, a dashboard with the live
annotated cameras, and a history of which room the resident is in. Telegram is the second channel and works away
from home Wi-Fi.

## 1. Start Home Assistant

1. Start Docker Desktop.
2. From the ElderCare folder: `docker compose -f deploy/homeassistant/docker-compose.yml up -d`
3. Open http://localhost:8123 and create the owner account.
4. Allow other devices on the Wi-Fi to reach the laptop (PowerShell **as administrator**, once):
   `New-NetFirewallRule -DisplayName "ElderCare + HA" -Direction Inbound -Protocol TCP -LocalPort 8000,8123 -Action Allow -Profile Private`
   (your Wi-Fi network must be set to *Private* in Windows settings).

## 2. Phone

1. Install **Home Assistant** (Companion app) on the family phone, connect to `http://<laptop-ip>:8123`
   (`ipconfig` shows the laptop's Wi-Fi IPv4 address).
2. In HA: Developer tools → Actions → find `notify.mobile_app_<your phone>`. Put that name into
   `deploy/homeassistant/config/automations.yaml` instead of `mobile_app_YOUR_PHONE`, then
   Developer tools → YAML → *Reload automations*.

## 3. Connect ElderCare to HA

1. HA profile (bottom left) → Security → **Long-lived access tokens** → create one.
2. Set these before starting ElderCare (e.g. in `.env` or the PowerShell session):

| Variable | Value |
|---|---|
| `HA_URL` | `http://localhost:8123` |
| `HA_TOKEN` | the long-lived token (only for the room/posture/status sensors) |
| `PUBLIC_BASE_URL` | `http://<laptop-ip>:8000`, so the phone can load the snapshot image |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | optional, see below |

ElderCare then creates `sensor.eldercare_room`, `sensor.eldercare_posture` and `sensor.eldercare_status`
(NORMAL / MONITOR / ALERT); add them to a dashboard card and HA keeps their history.

## 4. Live cameras in the HA dashboard

Settings → Devices & services → Add integration → **MJPEG IP Camera** → URL
`http://host.docker.internal:8000/live/<camera name>.mjpg` (one per camera in `cameras.yaml`). These show the
annotated stream (skeletons, rooms, fall banner).

## Telegram (works from anywhere)

1. In Telegram, talk to **@BotFather** → `/newbot` → copy the token → `TELEGRAM_BOT_TOKEN`.
2. Send any message to your new bot, then open `https://api.telegram.org/bot<token>/getUpdates` and copy
   `"chat":{"id": ...}` → `TELEGRAM_CHAT_ID`. For a family group, add the bot to the group and use the group's id.

## Test

With ElderCare running, play `videos/synthetic_fall.mp4` as a live camera from http://localhost:8000/live; the
phone should get "Fall, got up" (the person gets up in that clip). A person lying still for 30 s after a fall gives
"FALL — not getting up".
