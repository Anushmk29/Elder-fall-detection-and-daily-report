# ElderCare Vision (POC)

An offline video-analysis demo. Visitors browse pre-processed examples or upload a short clip (≤60 s, ≤100 MB,
MP4/MOV/WebM). The server analyses it on CPU and shows an annotated video, a clickable timeline of rooms, alerts
(falls, lying outside bed or sofa, long stillness) and a short AI-written note for family and caregivers.

**Not a medical device.** Detections come from a small pose model and readable rules, and they can be wrong.

## How it works

```
video ─► perception.py  YOLO11n-pose + ByteTrack, 6 fps, 640 px, CPU      → per-person box, 17 keypoints, posture
      ─► tracks.py      re-join broken IDs, smooth posture, walking/running (body heights per second)
      ─► events.py      resident = longest-visible person; zones; falls; lying outside rest zones; stillness; minutes
      ─► render.py      annotated H.264 MP4 (≤720p, faststart), event thumbnails, 3-frame strips
      ─► llm.py         OpenRouter: captions for the strips plus one daily report (cached, daily $ cap, template fallback)
      ─► result.json    read by static/result.html
```

`app/pipeline.py` wires these together. `app/jobs.py` runs uploads one at a time on a single worker thread, keeps job
state as JSON files, deletes the raw upload as soon as it is processed, and purges results after `RETENTION_HOURS`.

### Rules (`rules.yaml`)

| Rule | Default |
|---|---|
| Posture | torso angle from vertical ≥60° → lying, ≥30° → bending; thigh ≥50° or foreshortened → sitting; box shape if no torso |
| Fall | upright → lying within 2 s, hips drop ≥25% of body height, stays down ≥2 s. Always an alert, with "got up in X s" |
| Lying outside rest zones | lying for >120 s outside a `kind: rest` zone (bed, sofa) |
| Long stillness | moved less than 0.15 body lengths for 300 s |
| Walking / running | foot-point speed ≥0.35 / ≥1.5 body heights per second |
| Zones | polygons in normalised 0-1 coordinates, `kind: room` or `kind: rest`, giving dwell time per zone |

The default zones match the synthetic test scene. For a real camera, write a rules file with its own zones (start
from `rules.yaml`) and pass it with `--rules` when you process an example. Uploads use `rules.yaml`.

### Per-camera rules

`--rules videos/<name>.yaml` overrides top-level keys of `rules.yaml` (zones, `clock_start`, thresholds). Useful per-camera keys: `posture.legs_hidden` (knees out of view → `standing` by default for counters; set `sitting` for a sofa or table camera). Examples shipped in `examples/` were made this way; see `videos/CREDITS.md` for the stock footage and its camera files.

## Live mode: "Guardian at home"

The same pose model and rules run continuously on live cameras: phones (Android **IP Webcam** app), the laptop
webcam, RTSP/CCTV cameras, or a recorded video played as if it were a camera.

```
cameras.yaml ─► app/live.py  one thread per camera, 6 fps, GPU if available, rules re-run every second on a
                             rolling ~6 min window (the same tracks.py / events.py as uploads)
             ─► fall → "Possible fall" → still down after 30 s → FALL ALERT · got up → "Fall, got up"
             ─► data/live/events-YYYY-MM-DD.jsonl  (room enter/exit, posture, falls, lying on the floor, offline)
             ─► app/notify.py  Home Assistant (phone push + snapshot, room/posture/status sensors) and Telegram
             ─► app/daily.py   21:00 report: wake/bed, meals, bathroom, active time, rest, night, vs 7-day baseline
```

1. `copy cameras.example.yaml cameras.yaml` and edit the sources (phone URL, `0` for the webcam, RTSP URL, or a
   video file). Each camera has a `room`; the daily report recognises kitchen / bathroom / bedroom by name.
2. Optional: Home Assistant + Telegram, see [deploy/homeassistant/README.md](deploy/homeassistant/README.md).
3. Start the app with `--host 0.0.0.0` so phones and Home Assistant can reach it, and open
   http://localhost:8000/live. It shows every camera with skeletons, the current room and today's events, and has a
   **Play as live camera** form to push a recorded clip (e.g. a staged fall) through the live rules.
4. Report: http://localhost:8000/live/report, or `python scripts/daily_report.py [--date ..] [--send]`.
   `python scripts/seed_history.py` writes 7 days of clearly-marked demo history so the baseline comparison
   shows on the first day.

Live pages are served only to private-network addresses (this laptop, phones on the Wi-Fi, HA in Docker).

Live thresholds are in `rules.yaml` → `live:` (`no_recovery_s: 30`, `lying_outside_rest_s: 60`, `stillness_s`).
Environment: `CAMERAS_FILE`, `POSE_DEVICE` (`cpu` / `0`), `HA_URL`, `HA_TOKEN`, `HA_WEBHOOK_ID`,
`PUBLIC_BASE_URL`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `RESIDENT_NAME`, `REPORT_AT` (default `21:00`).

### Fall-detection benchmark

`python scripts/eval_mcfd.py "<folder with data_tuple3.csv>"` scores the rules on the Multiple Cameras Fall
Dataset (24 scenarios × 8 cameras) against its labelled fall windows: recall, false alarms per hour, and detection
delay. YOLO output is cached, so posture/fall rule changes re-score all 192 clips in seconds.

## Run locally (Windows PowerShell)

Python 3.11 or 3.12 and ffmpeg (with ffprobe) must be on PATH.

```powershell
cd "D:\Autonoetic Edge\ElderCare"
py -3.12 -m venv .venv          # or py -3.11
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # NVIDIA GPU: /whl/cu130
.venv\Scripts\pip install -r requirements-dev.txt
$env:OPENROUTER_API_KEY = "sk-or-..."    # optional; without it the text is template-based
.venv\Scripts\python -m uvicorn app.main:app --port 8000
```

Open http://localhost:8000. The pose weights download to `models/` on first use.

## Scripts

| Command | What it does |
|---|---|
| `python scripts/make_test_clip.py [out.mp4]` | Synthetic 13 s clip (people cut from ultralytics' bundled `bus.jpg`) where one person falls and gets up |
| `python scripts/process_example.py videos/x.mp4 --slug x --title "..." [--description ..] [--rules ..] [--clock 08:30]` | Run the pipeline and publish to `examples/<slug>/` |
| `python scripts/zone_helper.py videos/x.mp4` | First frame with a 10% grid, for reading zone coordinates |
| `python scripts/check_llm.py` | Test the OpenRouter key and model, and show today's spend |

To add a real example, put the clip in `videos/`, write its zones (for example `videos/kitchen.yaml`), run
`process_example.py`, then rebuild the image. `examples/` ships inside the image. `videos/` does not.

Demo videos to collect (30–60 s each, Pexels/Pixabay stock or footage with written consent): an older person in a
kitchen and on a sofa, walking down a hallway, and a staged fall onto a mattress.

## Tests

```powershell
.venv\Scripts\python -m pytest -q
```

These cover posture, fall/zone/stillness rules, point-in-polygon and dwell time, the LLM client (`httpx.MockTransport`:
cache, cost cap, fallback) and the HTTP API end to end on the synthetic clip (about 30 s on CPU).

## Configuration (environment)

| Variable | Default | |
|---|---|---|
| `OPENROUTER_API_KEY` | none | Without it, captions and the report use templates labelled "AI text unavailable" |
| `OPENROUTER_MODEL` | `google/gemini-2.5-flash` | Any vision-capable OpenRouter model |
| `MAX_LLM_USD_PER_DAY` | `2` | Uses the cost OpenRouter reports. After the cap, templates are used until midnight (server time) |
| `DEMO_PASSCODE` | none | If set, uploads require it |
| `RETENTION_HOURS` | `24` | Uploaded results are deleted after this |
| `UPLOADS_PER_HOUR` | `5` | Per IP |
| `MAX_UPLOAD_MB` | `100` | |
| `DATA_DIR` | `./data` | Jobs, results, LLM cache and spend ledger |
| `POSE_MODEL` | `models/yolo11n-pose.pt` | |

## Safeguards

Per-IP rate limit; unguessable result IDs (`secrets.token_urlsafe`); uploads are never listed and the raw file is
deleted after processing; ffprobe checks the container, the video stream and the duration; a consent checkbox and an
optional passcode; `Referrer-Policy: no-referrer` and `noindex`; the container runs as non-root on a read-only root
filesystem with a CPU-only torch and the weights baked in; LLM output is rendered as text, never HTML.

## Licences and references

Ultralytics YOLO is **AGPL-3.0**. That's fine for this open demo, but check it (or get an Ultralytics enterprise
licence) before any commercial deployment. ByteTrack ships inside ultralytics.

References: GajuuzZ/Human-Falling-Detect-Tracks (ST-GCN fall detection), taufeeque9/HumanFallDetection,
rusiru-erandaka/eldercare_monitor_system-Assignment (output format), mikel-brostrom/boxmot (trackers).

See [DEPLOY.md](DEPLOY.md) for VPS deployment.
