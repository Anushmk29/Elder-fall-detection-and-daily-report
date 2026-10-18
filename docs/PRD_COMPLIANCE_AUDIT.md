# ElderCare PRD and open-source implementation audit

Audit date: 17 October 2026 (Asia/Calcutta). Scope: `D:\Autonoetic Edge\ElderCare`, compared with the supplied `ElderCare_PRD.md` v1.0 and `OPEN_SOURCE_INVENTORY.md` in Downloads.

## Verdict

**This is a working, simplified proof of concept, but it does not meet the supplied PRD's definition of done.** The main technology choices are implemented and the existing 20 tests pass. Several core safety rules, required events, metrics, API contracts, report fields, upload controls and UI features are missing or materially different.

**The principal open-source runtime components are genuinely used.** The fall-detection reference repositories are not installed or integrated as engines: the application uses its own Python posture/fall heuristics on top of Ultralytics. This is consistent with inventory section D, which explicitly classifies those repositories as references, not dependencies. Their absence is not itself a PRD failure. Future-roadmap repositories in section E are likewise not required for v1.0.

The supplied documents were used as audit criteria. Their instructions to build, publish, install tools or follow milestones were not treated as instructions to modify the product. This audit adds evidence and this report; application code is unchanged.

## Evidence and limits

Reviewed all application modules, browser HTML/JS/CSS, scripts, rules, deployment files, README, all tests, all four generated result JSON files, dependency declarations and the installed environment. Source locations below refer to the audit snapshot.

| Check | Result |
|---|---|
| Existing Windows tests: `.venv\Scripts\python -m pytest -q` | **20 passed**, one Starlette/httpx deprecation warning, **50.05 s** total. Includes real YOLO processing through the upload API. |
| Additional PRD probes | Reproduced rest-zone false fall, missed bending-origin fall, detection-gap confirmation, missing safety distinction, permissive re-linking, unsafe AI acceptance, corrupt-cache exception, parallel budget overshoot, and debug-file exposure. |
| Real blank-video pipeline | A generated 3 s gray video ran through actual YOLO/rendering and returned success with no people and “Nothing needed attention in this clip.” |
| Dependency consistency | `pip check`: no broken requirements found. This does not establish license or security compliance. |
| Bundled video encoding | All four annotated videos are H.264/yuv420p at 15 fps and <=1280×720; their observed size rates are below the 15 MB/min target. |
| Media Range | **206** for `Range: bytes=0-99`. |
| Basic encoded traversal | **404** for the tested encoded traversal path. This is a spot check, not an exhaustive security assessment. |
| Passcode / oversized file | **403 / 413**, tested with temporary in-process configuration. |
| Standard Compose validation | Blocked because project `.env` does not exist. This is a setup prerequisite, not proof of invalid Compose syntax. |
| Compose structure | `docker compose -f deploy/docker-compose.yml config --no-env-resolution --quiet` succeeds. |
| Docker build | Attempted, then canceled after roughly nine minutes of slow/stalled Debian package downloads. No completed image or container runtime validation; this is an environment/download limitation, not a demonstrated Dockerfile failure. |
| Browser smoke / Lighthouse | Not executed: no connected browser was available, and Playwright is absent from the project's dev dependencies. UI findings are from source inspection. |
| Paid OpenRouter calls | Not executed. HTTP behavior was tested with mock transports; all bundled examples currently contain template reports. |

Reproducible evidence: [probe script](audit/prd_probes.py), [probe results](audit/probe-results.json), [environment check](audit/environment_check.py), [installed dependencies](audit/installed-dependencies.json), and [environment/video/blank-clip results](audit/environment-results.json). The blank-video run took 46.78 s while a Docker build was also active; this was a functional probe, not a controlled performance benchmark. Torch reported 6 threads at import and 8 after inference, rather than the specified configurable default of 2.

Passing the current tests does not demonstrate PRD compliance: several tests explicitly assert the implementation's incompatible values, such as upload HTTP 200, invalid-video HTTP 400 and fall severity `alert`.

## Priority findings

### P1 — Correct before calling this PRD-complete or using it for the public summit demo

1. **Fall detection ignores rest/bed exclusions and loses the required recovery distinction.** [events.py](../app/events.py#L39), [event construction](../app/events.py#L131). `detect_falls()` receives no zone information, and the caller does not filter falls in rest zones. A whole-frame bed zone still produced a fall in the probe. Every fall gets `severity="alert"`; there is no 15-second recovery cutoff, `data.recovered`, or `data.time_to_get_up_s`. Sitting is accepted as recovery. Both a six-second recovery and an unrecovered 23.8-second fall got the same severity. PRD §§6.7, 6.9 require high vs critical and code-enforced report status.

2. **Fall confirmation can credit time when the person was not observed.** [events.py](../app/events.py#L45). Confirmation subtracts timestamps until a later standing/sitting sample or the final sample, without requiring continuous lying or bounding observation gaps. A lying observation at 4 s followed by another at 20 s was accepted as a 16-second fall despite no intervening observations. Upright candidates are restricted to `posture == "standing"`, missing an explicitly required bending-to-lying transition. Walking is often represented internally as standing posture, so this finding does not claim all walking-origin falls are missed. The reference height is one earlier box, not the previous-three-second median required by §6.7.

3. **No-person analysis can produce reassuring output.** [pipeline.py](../app/pipeline.py#L56), [events.py](../app/events.py#L153), [llm.py](../app/llm.py#L103). There is no empty-track rejection. A real generated 3-second gray clip passed through YOLO and rendering, returned zero people/alerts, and the template said “Nothing needed attention in this clip.” This is not equivalent to evidence that the resident was safe. Required behavior: fail with `NO_PERSON_FOUND` and a clear retry message (§8.5).

4. **Arbitrary uploads inherit fictional room boundaries and a sofa exclusion area.** [rules.yaml](../rules.yaml#L37), [pipeline.py](../app/pipeline.py#L57). Every upload uses the synthetic scene's Kitchen, Living room and Sofa polygons. This invents room labels for unrelated cameras and can suppress lying-outside-rest events in an arbitrary part of the image. §6.6 explicitly requires no zones by default for uploads and hidden room UI when zones are absent.

5. **Three required events and the mobility metrics are absent.** [events.py](../app/events.py#L93). No implementation exists for `LONG_BATHROOM_STAY`, `VISITOR_PRESENT`, or `GOT_UP`. Visitors are counted as tracks but do not produce a first-seen event after five seconds. There is no sit-to-stand count/duration or walking-speed index. Per-person minutes contain only active/sitting/lying, not all six states. Coverage percentage and alerts by required severity are absent (§§6.7–6.8).

6. **AI output has no enforced safety status or required structured report.** [llm.py](../app/llm.py#L22), [llm.py](../app/llm.py#L89), [pipeline.py](../app/pipeline.py#L82). Report output is unrestricted plain text. The required headline/status/summary/highlights/checks/mobility/caregiver JSON is absent, as is the deterministic urgent/attention/ok override. A mocked response saying “Everything is safe. No fall occurred.” was accepted with `ai=true` even when fall facts were supplied. Prompting alone does not implement §6.9.

7. **The daily LLM cap is race-prone and AI failures can still fail a job.** [pipeline.py](../app/pipeline.py#L78), [llm.py](../app/llm.py#L38). Four caption calls run concurrently, each checks the ledger before any has recorded its charge. A mocked $0.01 cap allowed $0.04 of calls. Reads/writes of cached JSON and `_add_spend()` are outside the HTTP exception handler; corrupt cache JSON raised `JSONDecodeError` rather than returning a fallback. Ledger writes are locked, but check-and-reserve is not atomic. Spend uses local server date rather than UTC; no pricing-model fetch/cache is implemented. Fix cap accounting and make all AI-path failures degrade to a deterministic report.

8. **The result format is a separate, incompatible contract.** [pipeline.py](../app/pipeline.py#L87). All four on-disk examples lack `id`, `kind`, `version`, `created_at`, `clips`, `segments`, `zone_segments`, `metrics`, `ai`, `processing`, and `disclaimer`. The API injects `id`, but does not repair the remaining fields. Events use lowercase types, numeric IDs, `who` instead of `person`, and `alert/warn/info` severities. No result schema validator exists. This affects the API, UI, CLI and acceptance criteria (§6.11), not just naming.

9. **Media serving has no debug-file exclusion.** [main.py](../app/main.py#L44). Entire example/result directories are mounted with `StaticFiles`. A disposable `debug/observations.jsonl` under an audit result was served with HTTP 200. Current code does not generate that debug file; the verified defect is lack of the mandatory deny rule if such files are present. Range requests and the tested traversal guard worked. Add an explicit public-file allowlist or deny `debug/` (§§6.11, 8.1).

10. **Public-demo readiness and licensing evidence are incomplete.** `LICENSE` and `THIRD_PARTY_NOTICES.md` are missing. The README's statement that AGPL is “fine for this open demo” does not establish a source-publication/licensing arrangement. The supplied inventory also says Ultralytics sprites should be test-only; `examples/synthetic-fall` is served publicly and baked into the image. Resolve the intended licensing/distribution route and use suitable public example footage; this report does not infer that a commercial license does or does not exist outside the workspace.

### P2 — Required implementation and operational gaps

11. **Posture and tracking rules differ without documented decisions.** [posture.py](../app/posture.py#L43), [tracks.py](../app/tracks.py#L43), [rules.yaml](../rules.yaml#L9). An ambiguous 1:1 box with no keypoints becomes standing instead of unknown. Lying, bending, hidden-knee and thigh-distance tests differ from the PRD. Re-linking allows three seconds plus a speed-based distance expansion, without the 40% median-height gate; the probe merged 400-pixel and 100-pixel-height people. Minimum track length is 12 samples (~2 s), walking threshold .35 instead of .30, and smoothing ties do not deliberately preserve prior state. No posture segments or short-segment merge exists. `docs/DECISIONS.md` is absent. These changes need validation or explicit spec agreement, not an assumption that all alternative rules are necessarily worse.

12. **Stillness is a different behavior and severity.** [events.py](../app/events.py#L82), [rules.yaml](../rules.yaml#L33). It checks displacement from an anchor, not speed below .05 body-heights/s plus unchanged posture. It can span posture changes and long detection gaps. Default duration is 300 s rather than 900 s; events are `info` rather than medium and are omitted from alert totals. A standing-to-sitting sequence with fixed position still produced a stillness event in the probe.

13. **Validation and normalization are incomplete.** [main.py](../app/main.py#L129), [pipeline.py](../app/pipeline.py#L26). ffprobe verifies a video stream and duration, but MKV is rejected, limits are partly hardcoded, and the API accepts up to 60.5 s. There is no normalized `source.mp4` or pre-inference H.264/size conversion. Invalid video and excessive duration return HTTP 400 instead of 415/422. ffprobe `TimeoutExpired` is not caught in the upload validation block. Size checking happens after multipart parsing; the application does not enforce the limit while receiving the request body. Caddy does provide an outer 110 MB limit in the supplied deployment.

14. **Upload configuration and API contracts differ.** [main.py](../app/main.py#L24). `start_clock` and `blur_faces` fields are ignored because the route does not accept them. Default time is 09:00 rather than 00:00. POST returns HTTP 200 with `id` rather than HTTP 202 with `job_id`. `/p/{id}` is missing. Config/example/health payloads differ from the PRD. Queue limit is hardcoded to 20 rather than configurable default 10; busy returns 503 rather than 429 `SERVER_BUSY`. There are no stable PRD error codes. Fixing the backend contract requires updating the JS and tests together.

15. **Configuration is not centrally loaded or validated.** [.env.example](../.env.example), [main.py](../app/main.py#L24), [perception.py](../app/perception.py#L16). No settings module, dotenv load, dataclass configuration or Pydantic settings exists. Merely copying `.env` does not configure local Python execution; Compose supplies it separately. Missing/ignored PRD variables include `OPENROUTER_CAPTION_MODEL`, `APP_BASE_URL`, `MAX_CAPTIONS`, `CAPTION_INTERVAL`, `MAX_CLIP_SECONDS`, `RATE_LIMIT_PER_HOUR`, `QUEUE_MAX`, `SAMPLE_FPS`, `TORCH_THREADS` and `LOG_LEVEL`. Some counterparts live in YAML or use different names. No startup ffmpeg check or model-exists guard is present.

16. **Rendering and captions only partially meet the contract.** [render.py](../app/render.py#L14), [render.py](../app/render.py#L62), [pipeline.py](../app/pipeline.py#L45). Caption strips are 3 × 768 = 2304 pixels wide instead of 1152, captured before box/label annotation. Caption selection does not implement fall+2 s, a 20-second interval, five-second deduplication or configurable ten-caption maximum. It can select 12 event moments. Event thumbs are full rendered resolution rather than 640 px. Poster selection is event+0.5 s or midpoint rather than 10%. Video is H.264/yuv420p/faststart at <=15 fps and <=1280×720, but CRF is 26 instead of 28. Rendering accepts older samples than required. Banners omit the clock and recovery text; the clock is bottom-left, without the specified top-left posture HUD. Face blur is available through rules but lacks the upload control and exact requested geometry.

17. **A stuck encoder can stall the single worker indefinitely.** [render.py](../app/render.py#L41), [render.py](../app/render.py#L74). `Popen` pipe writes and `ff.wait()` have no timeout/kill path. The synthetic-video generator has the same pattern. Only ffprobe has a timeout. PRD §9 requires bounded subprocess execution; cleanup also runs on this same worker, so a stuck job delays retention cleanup.

18. **Job restart and retention semantics differ.** [jobs.py](../app/jobs.py#L30), [jobs.py](../app/jobs.py#L102). Startup requeues previously queued/processing jobs instead of marking interrupted work failed. If the raw source was deleted before final job state was saved, a stale processing job is neither resumed nor failed, and cleanup skips it. Cleanup runs every ten minutes within the worker rather than a separate 30-minute cleanup thread. A long/stuck job blocks cleanup. Raw uploads are normally deleted immediately after processing, which is a positive privacy measure. The queue limit check and submission are not atomic, so simultaneous accepted requests can exceed the intended limit.

19. **Required UI is incomplete.** [index.html](../static/index.html), [index.js](../static/index.js), [result.html](../static/result.html), [result.js](../static/result.js). Present: gallery, drag/drop upload, consent/passcode, inline polling, video, stat tiles, room/event timeline, alert cards, text report and caption gallery. Missing: start-clock/blur controls, client-side size/extension/duration probes, separate resumable progress page, stepper/elapsed time, status/headline banner, posture lane, SVG timeline, mobility/coverage tiles, room bars, severity-sorted alerts, structured report sections and “How this was made” details. Alert and timeline clicks seek and play, and the playhead/highlighting logic exists, but timeline items and caption moments use clickable non-focusable divs rather than keyboard controls. No theme toggle/dark mode. [style.css](../static/style.css#L1) imports Google Fonts despite the explicit no-CDN requirement. These source findings do not establish a Lighthouse score or cross-browser behavior.

20. **CLI and curated example plan are incomplete.** [process_example.py](../scripts/process_example.py#L16), [make_test_clip.py](../scripts/make_test_clip.py), [credits](../videos/CREDITS.md). The CLI takes one positional video and supports `--rules/--clock`; it does not support the PRD's repeatable `--video path@clock`, `--scene`, `--config`, `--no-ai`, `--blur-faces`, `--max-seconds`, optional title or structured summary. Config merging is shallow, so a partial nested threshold override discards sibling defaults. No custom resident names. Four examples exist (`kitchen`, `sofa`, `walking-cane`, `synthetic-fall`), but no unrecovered-floor alert example or combined day. Slug names alone are not the concern: required scenarios are missing. All four reports are templates. The synthetic clip is 13 s with different motion/visitor timing, not the specified 14 s; no per-example scene YAML is generated.

21. **Deployment and reproducibility gaps remain.** [Dockerfile](../Dockerfile), [docker-compose.yml](../deploy/docker-compose.yml), [requirements.txt](../requirements.txt). CPU Torch installation, one Uvicorn worker, non-root UID, persistent data, Caddy, read-only root and healthcheck are implemented. Missing Compose `cpus: 2` and `mem_limit: 3g`; Docker installs GUI OpenCV via Ultralytics and explicitly installs GUI libraries instead of headless OpenCV. Dev uses Python 3.11.6, while the image targets 3.12. Most dependencies are ranges/transitive, with no lockfile. `tests/` is excluded from the image and pytest is not installed in it, so running the required Linux test suite needs a test stage or mounted tests/dev dependencies. Logs are plain text rather than structured per-job stage timings. No processing time/cost metadata is emitted. HTTPS and shared-Caddy deployment instructions exist, but no image-size/RAM/2-vCPU performance evidence is included.

22. **The folder is not its own Git repository.** `git rev-parse --show-toplevel` resolves to `D:/`; ElderCare has no local `.git`. This does not stop the app running, but does not meet the milestone's independent-repo requirement and makes the README's project-specific commit/pull/source-publication assumptions unverified. No repository initialization or Git changes were made during this audit.

## Requirement coverage matrix

“Partial” means useful code exists but at least one material requirement in that row is missing. “Unverified” is not a claimed failure. Optional items are not counted as release blockers.

| PRD reference | Requirement group | Assessment | Evidence / key gap |
|---|---|---|---|
| US-1, §7.3 | 2–4 examples and upload entry | Partial | Four cards and upload exist; no required status/headline/two-stat card contract. |
| US-2, §7.5 | Complete result view | Partial | Video/tiles/alerts/text present; required posture/status/structured report views absent. |
| US-3 | Click-to-seek | Present in code | Alert/event/room seek handlers and playhead highlighting; browser behavior unverified. |
| US-4, §7.4 | Upload progress | Partial | Inline 1.5 s polling, stage, percent, queue position; no progress route/stepper/elapsed time. |
| US-5, §8.5 | Clear rejection codes/messages | Partial | Several useful messages, incompatible HTTP values and no specified error codes. |
| US-6, §11.1 | Example command | Partial | Single-clip custom CLI, incompatible required command/options. |
| US-7 (SHOULD) | Multi-clip day | Missing | One video/result, no clip indices/tabs/combined report. |
| US-8 (SHOULD) | Passcode and daily cap | Partial | Passcode works; concurrent calls can exceed cap. |
| US-9/10 (COULD) | Zone editor / PDF | Not implemented; optional | Not a MUST failure. |
| §6.2 | ffprobe validation and bounded limits | Partial | Real ffprobe validation; no MKV, some hardcoded limits and wrong errors. |
| §6.2 | Normalized source | Missing | Original video passed directly into perception/render. |
| §6.3 | YOLO11n-pose + ByteTrack, CPU, per-clip reset | Present with deviation | Fresh model/tracker, real CPU prediction; direct BYTETracker rather than `model.track`, conf .25 rather than .35. |
| §6.3 | Sampling / observations / threads | Partial | `cap.grab` sampling works; different observation schema, no stored torso/hip/body-height fields; no TORCH_THREADS control. |
| §6.4 | Pure tested posture function | Partial | Pure and tested, several thresholds/rules differ. |
| §6.5 | Re-link, smoothing, motion | Partial | Implemented using different criteria; no height gate/prior-state tie rule. |
| §6.5 | Posture segments | Missing | No state-segment output or <.5 s merging. |
| §6.5 | Resident/visitor assignment | Partial | Longest visible becomes resident; labels/custom naming differ. |
| §6.6 | Zones and dwell | Partial | Normalized polygons and dwell exist; `kind` schema, overlapping totals, no Other area, no bed/bathroom types. |
| §6.6 | Uploads have no zones | Fails | Synthetic zones applied universally. |
| §6.7 | FALL | Partial, incorrect cases reproduced | Exclusion, continuity, bending, median height, recovery severity/data missing. |
| §6.7 | LYING_OUTSIDE_REST | Partial | Basic rule exists; wrong severity and incompatible bed schema. |
| §6.7 | LONG_STILLNESS | Partial | Different signal, threshold and severity; posture changes ignored. |
| §6.7 | LONG_BATHROOM_STAY / VISITOR_PRESENT / GOT_UP | Missing | No event-generation code. |
| §6.8 | Required metrics | Mostly missing | Visible seconds and coarse minutes/dwell only. |
| §6.9 | OpenRouter transport | Partial | HTTPX/auth/image parts/60 s timeout/usage request present; no retries, referer or separate caption model. |
| §6.9 | Cache and daily spend | Partial | Cache and ledger work in normal case; errors/race/pricing/UTC/status gaps. |
| §6.9 | Caption selection/strips/JSON | Partial | Three-frame images exist; size, labels, selection, schema differ. |
| §6.9 | Structured report + code-enforced status | Missing | Plain text is accepted without safety override. |
| §6.9 | Deterministic fallback | Partial | Works for missing key and ordinary HTTP errors; wrong schema, other exceptions escape. |
| §6.10 | Browser-safe annotated video | Substantially present | H.264/yuv420p/faststart, <=15 fps, bounded dimensions; exact style/CRF/thumb requirements differ. |
| §6.10 (SHOULD) | Face blur | Partial | YAML-controlled head pixelation; no upload checkbox or dedicated CLI flag. |
| §6.11 | Result contract/output organization | Fails | All example JSONs lack mandatory keys, directories differ, no validator. |
| §7.1 | Vanilla JS / no build / no CDN | Partial | Vanilla JS yes; external Google Fonts. |
| §7.1 | Themes and accessibility | Partial/missing | Responsive CSS exists; no dark theme or keyboard timeline controls; scores unverified. |
| §7.5 | Timeline / status / room/report views | Partial | Room/event div timeline only; required posture/SVG/status/details absent. |
| §8.1 | Endpoints and response contracts | Partial | Core routes exist; `/p` missing, payloads/status codes differ. |
| §8.2 | Unguessable IDs / no upload listing | Present | `token_urlsafe(18)` is stronger than required 16-byte entropy; not a security defect. |
| §8.3 | Storage layout | Partial | Separate jobs/uploads/results, but flatter files and different LLM directories. |
| §8.4 | FIFO/single worker | Present | Thread and Queue implemented. |
| §8.4 | Lifecycle/restart/cleanup | Partial | Different status names; requeues; worker-coupled cleanup and stale-state issue. |
| §9 | Security/abuse | Partial | Consent/passcode/rate/size/proxy/non-root exist; CSP/debug/body handling gaps. |
| §9 | Secrets/browser rendering | Positive evidence | No key in browser files/config; model text uses `textContent`; no CORS middleware found. |
| §9 | Privacy | Partial | No upload listing, deletion routine, raw upload removal; default rooms/blur controls/CDN deviations. |
| §9 | Performance/resources/observability | Unverified/partial | No bounded 2-vCPU benchmark or memory/image proof, no structured stage logs. |
| §9 | Licensing artifacts | Missing | No LICENSE or THIRD_PARTY_NOTICES. |
| §10 | Environment + rule configuration | Partial | Environment reads/YAML exist, no validated settings; several variables missing. |
| §11 | Supporting scripts/docs | Partial | No fetch_model, DECISIONS, result schema; DEPLOY at root is a harmless location difference by itself. |
| §12 | Docker/Caddy/Windows support | Partial | Config largely present, missing resource/headless requirements; current env is Python 3.11. |
| §13 | Three safety/routine scenarios | Partial | Real routine clips and synthetic recovered fall; no persistent-floor scenario. |
| §14.1 | Automated tests | Partial | 20 tests pass; no dedicated tracks or browser smoke suite and many mandatory cases absent. |
| §14.2 | Definition of done | Not met | See acceptance checklist below. |
| §14.3 | Synthetic acceptance asset | Partial | Working synthetic fall test with different duration/choreography/config files. |
| §15 | Milestones | Incomplete | M2–M7 have useful implementations but their PRD completion gates are not met; M8 not complete. |
| §17 | Live video/ST-GCN/notifications/etc. | Correctly out of scope | No requirement to add them to close v1.0 gaps. |

## Open-source inventory cross-check

Installed versions below are from this workspace's venv, not copied from the inventory's historical environment. “Indirect” means a dependency supplies/uses it; it is not explicitly imported by ElderCare code.

| Inventory item | Actual integration | Version / evidence | Assessment |
|---|---|---|---|
| Ultralytics | Direct `YOLO` imports/prediction | 8.4.178; `app/perception.py:8,56` | Used. |
| YOLO11n-pose weights | Local `models/yolo11n-pose.pt`, loaded per clip | 6,255,593 bytes; Docker build invokes YOLO to fetch weights | Used; no explicit fail-fast guard if absent at runtime. |
| ByteTrack | `ultralytics.trackers.byte_tracker.BYTETracker` | `perception.py:9,22,58`; bundled bytetrack YAML | Used, Ultralytics implementation. Original repo is not separately installed. |
| Torch / torchvision | CPU inference dependencies | 2.14.1+cpu / 0.29.1+cpu | Used indirectly; Docker CPU index instruction present. |
| OpenCV headless | Actual import is `cv2`, installed distribution is `opencv-python` | 5.0.0.93 | OpenCV used, **wrong distribution** for PRD; headless absent. |
| NumPy | Direct rendering/synthetic-video arrays | 2.4.6 | Used. |
| FFmpeg / ffprobe | Separate subprocesses | Local FFmpeg 8.1.1 full build | Used; ffprobe validation and libx264 rendering. No source normalization stage. |
| x264 | FFmpeg `-c:v libx264` | Encoder command in render.py | Used through FFmpeg; not a Python package. |
| FastAPI | Routes, request/form parsing, responses | 0.143.3 | Used. |
| Starlette | Underlying ASGI/static/file-response layer | 1.8.0 | Used indirectly; Range tested. |
| Uvicorn | Documented/server/container entrypoint | 0.54.0 | Used. |
| Pydantic | FastAPI dependency | 2.14.1 | Installed indirectly, but no ElderCare result/config models. |
| pydantic-settings | None found | Not installed / declared | Missing configuration component; allowed dotenv+dataclass alternative is also absent. |
| python-multipart | FastAPI upload parsing | 0.0.32 | Used indirectly. |
| HTTPX | OpenRouter client and mock tests | 0.28.1 | Used. |
| PyYAML | `yaml.safe_load` for rules | 6.0.3 | Used safely for parsing. |
| Pillow | Ultralytics dependency | 12.3.0 | Installed indirectly. App builds thumbnails/strips with OpenCV, not Pillow. Harmless implementation substitution by itself. |
| Python | Local interpreter vs container | Local 3.11.6; Docker `python:3.12-slim` | Local environment differs from target. |
| Debian/Python base | Dockerfile FROM | `python:3.12-slim` | Configured. |
| Caddy | Reverse-proxy service/Caddyfile | `caddy:2.10-alpine` | Configured; HTTPS deployment not exercised. |
| Docker / Compose | Available Docker engine, compose files | Engine 29.4.2 observed | Used for build/config checks. |
| pytest | Actual executed test suite | 9.1.1 | Used. |
| Playwright | None | Not installed/declaration/tests | Missing required UI test tooling. |
| pip-licenses | None | Not installed/declaration | Optional suggestion, not a runtime requirement. Metadata inventory captured without installing it. |
| OpenRouter / vision LLM | HTTP implementation present | Default `google/gemini-2.5-flash` | Proprietary service integration, not open-source code; live API/cost not verified. |

**Additional dependencies not enumerated in the supplied inventory:** `lap>=0.5.12` is explicitly required (installed 0.5.13) for tracking assignment. Installed transitive packages include `ultralytics-thop`, `ultralytics-platform`, `polars`, `matplotlib` and others. The generated installed-dependency JSON records the full observed environment. A version-locked deployment SBOM should be generated from the final image.

### Reference repositories and future projects

| Repository/group from inventory | Evidence in this project | Conclusion |
|---|---|---|
| GajuuzZ/Human-Falling-Detect-Tracks | README reference only; no AlphaPose/ST-GCN/SORT integration | Inspiration, not runtime engine. |
| taufeeque9/HumanFallDetection | README reference only; no LSTM fall model | Inspiration, not runtime engine. |
| rusiru-erandaka/eldercare_monitor_system-Assignment | README reference only; own events/result implementation | Inspiration, not integrated upstream pipeline. |
| mikel-brostrom/boxmot | README reference only | Not used; no Re-ID dependency. |
| elderly-bed-monitor, child-activity-monitoring, daycare-system, NVIDIA video-search-and-summarization | No imports/declarations/vendored implementation identified | Not integrated; inventory marks these as references. |
| pyskl, mmaction2, SlowFast, deep-person-reid, PANNs, YAMNet, Frigate, Bermuda, ESPresense, RuView | No imports/declarations identified | Not used; correctly future/out-of-scope. |
| KidsCare code | No application imports/dependencies found; deployment docs discuss co-hosting | No observed runtime shared-code dependency. This is not a whole-history copying/provenance proof. |

A dependency/reference search cannot prove that no snippets were ever copied. The folder lacks independent Git history, and this audit did not perform a complete source-similarity or license-provenance investigation. It establishes observable integration, not authorship.

### License verification boundaries

The principal license facts were cross-checked against official upstream sources: [Ultralytics LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE), [original ByteTrack LICENSE](https://github.com/FoundationVision/ByteTrack/blob/main/LICENSE), and [FFmpeg legal documentation](https://ffmpeg.org/legal.html). Ultralytics is AGPL-3.0; original ByteTrack is MIT, but this app imports the Ultralytics implementation; FFmpeg terms depend on enabled components, including libx264. The local FFmpeg build advertises `--enable-gpl --enable-version3 --enable-libx264`, so the inventory's GPL-2.0+ shorthand is not a complete description of this particular binary.

The installed-package metadata file is evidence of package metadata, not a legal clearance of all transitive code/model weights/assets. This audit did not verify every inventory license, commercial license entitlement, footage consent, or source-publication arrangement. Stock footage credits exist in `videos/CREDITS.md`, but `.gitignore` excludes that Markdown file via `videos/*`; archive/publish the provenance in a tracked location.

## Acceptance checklist (§14.2)

| Acceptance criterion | Outcome |
|---|---|
| 1. pytest Windows and Docker | Windows passes on Python 3.11.6. Python 3.12/Linux test run not established. |
| 2. Exact synthetic generation + example CLI command; valid schema; H.264; <=60 s laptop | Existing upload integration test passes and bundled video is H.264. Exact PRD CLI command is unsupported; no required result schema; total suite time is not the specified isolated CLI benchmark. |
| 3. Synthetic fall produces FALL | Actual API test produces a fall, but type/severity/schema differ. No DECISIONS document. |
| 4. Live AI succeeds with captions/report below $0.05 | Unverified; all four bundled examples use templates; per-result AI cost not recorded. |
| 5. 60 s UI upload <=120 s on 2-vCPU Docker | Unverified; no browser available and no valid target benchmark yet. Do not extrapolate from the shorter Windows test. |
| 6. Full flow without key and Template badge | Core no-key flow works; UI uses “AI text unavailable” instead of required template-report wording. |
| 7. Compose, built non-root image, /healthz | Compose structure and local health checked. Dockerfile declares non-root UID 10001. Build canceled during stalled dependency downloads; image startup, actual UID and container health remain unverified. |
| 8. Lighthouse >=90 and no overflow at 360 px | Unverified; source contains clear keyboard-accessibility omissions. |

## Suggested repair order

1. Restore the behavioral/data contract: event enums/schema, status override, continuous fall confirmation and zone exclusions, explicit no-person failure, correct default upload zones, missing events and metrics.
2. Make the LLM path dependable: structured report/captions, schema validation, deterministic safety override, bounded retries, atomic budget reservations, UTC ledger, exception-safe cache/storage, correct frame strips and per-result cost.
3. Align API/config/jobs: validated settings, normalized sources, all formats/options, stable error codes, progress route, bounded queue, restart failure handling, independent cleanup, media allowlist and encoder timeouts.
4. Update the UI and CLI to consume the restored contract; add posture/status/mobility views, keyboard controls, local fonts/themes, correct upload fields, and the documented example command.
5. Finish deployment/content verification: CPU/memory limits, headless OpenCV, locked dependencies, licensing notices/provenance, real recovered/unrecovered examples, and the missing acceptance tests on Windows plus Python 3.12 Docker.

Do not add the reference repositories or roadmap models merely to increase the number of open-source projects used. The v1.0 gap is completing and validating the specified pipeline.
