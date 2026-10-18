# ElderCare POC demo readiness — recheck, 17 October 2026

**Assessment: functional technical POC; suitable for a rehearsed walkthrough of saved examples, but not yet signed off for a polished public demo with live uploads.**

This assessment separates demo value from full PRD compliance. The missing full-day aggregation, clinical metrics and advanced UI do not prevent showing the basic concept.

## What is implemented

- Offline processing of short MP4/MOV/WebM uploads on CPU, with ffprobe validation, consent, optional passcode, per-IP rate limiting, a single job worker and progress polling.
- Real YOLO11n-pose detection and Ultralytics ByteTrack tracking; standing/sitting/bending/lying classification and movement-derived walking/running labels.
- Heuristic resident assignment: the longest-visible track is labelled Resident. It does not recognize an individual's identity, age or actual household role.
- Basic fall detection and estimated recovery time; lying-outside-rest and stillness rules.
- Configured polygon zones and dwell estimates. Zones must be matched to the camera; uploads currently inherit the synthetic scene, which is a known defect.
- Annotated H.264 video, skeletons/boxes/labels, fall banner, event thumbnails and three-frame caption strips.
- Example gallery and result view with tiles, room/event timeline, alert cards, seek handlers and a text report.
- OpenRouter caption/report code with template fallback. No local `.env` or process OpenRouter key was present during the recheck, and all saved examples contain template reports.

## What can be shown now

| Example | Duration | Saved output |
|---|---|---|
| kitchen | 12.01 s | Two tracked people, kitchen zone, activity totals, no alert. |
| sofa | 10.32 s | Sitting/activity classification, sofa/living-room zones, no alert. |
| synthetic-fall | 13 s | A generated fall at 5 s, reported recovery in 6 s, alert and annotated fall banner. This is synthetic, not validation on real fall footage. |
| walking-cane | 18.12 s | Outdoor pose/movement overlay and yard zone. Result contains six tracks; do not claim those are independently verified unique people. |

All four example page/result routes and all referenced videos, posters, event thumbnails and caption images returned successful HTTP responses. Range requests succeeded. Saved annotated posters and selected caption strips were visually inspected. Browser rendering/playback, seek interactions and mobile appearance were not exercised in this recheck.

## Demo limitations that matter

1. The configured lying threshold is 120 s and stillness threshold 300 s, while uploads are capped at about 60 s. Those alerts cannot be demonstrated through normal uploads without appropriate demo overrides or longer CLI examples.
2. Falls in rest zones are not excluded, and recovery does not determine the required critical/high severity.
3. No-person clips can return reassuring summaries. Fix this before accepting arbitrary audience clips.
4. The current report is a summary of one short clip, not an aggregated day. AI-generated reports are not currently configured in this local setup.
5. Bathroom-overstay, visitor-arrival events, sit-to-stand metrics, live-camera ingestion and notifications are not implemented.
6. Target Docker deployment, browser smoke and predictable live-processing time still need verification. Existing deployment files alone do not establish deployed readiness.

## Fresh verification

- **16 focused tests passed in 7.70 s**: posture, events, zones and LLM mock tests. `YOLO_OFFLINE=true` was set only for the test process.
- The earlier audit's complete **20-test run passed in 50.05 s**. That is earlier evidence, not a fresh full-suite pass.
- Fresh full-suite attempts stalled and were stopped. The later attempt reached synthetic-video fixture generation. An isolated 30-second diagnostic also timed out and captured an importlib filesystem-stat frame; this does not establish the root cause. A later standalone default application import succeeded in 3.81 s.
- No source-code fixes or persistent environment changes were made.

## Smallest useful demo-preparation pass

1. Fix empty-person results, default upload zones and fall rest-zone/recovery handling.
2. Prepare one believable recovered-fall example and one sustained-floor example with documented demo thresholds.
3. Configure and verify OpenRouter if AI-written captions/reports are part of the pitch, and regenerate those examples; otherwise clearly present the current text as template summaries.
4. Rehearse the exact browser flow and one known-good upload on the actual demo machine. Investigate the intermittent fixture/startup delay before promising reliable live processing.
5. Keep preprocessed examples available as the primary presentation path.

See [the full compliance audit](PRD_COMPLIANCE_AUDIT.md) for implementation evidence and the wider PRD backlog.
