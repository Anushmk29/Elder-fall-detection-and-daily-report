# ElderCare "Guardian at home": how it was built, and why

A learning guide for cross-checking the build. It explains every layer, the logic behind each rule, how the fall
detector was tuned and tested on a real dataset, what the numbers mean, and where the limits are.
Commands assume PowerShell in `D:\Autonoetic Edge\ElderCare`.

---

## 0. The one-paragraph version

A camera frame goes into a **pre-trained pose model** (YOLO11-pose), which returns a box and 17 body keypoints for
each person. **Hand-written, readable rules** turn those keypoints into a posture (standing, sitting, bending,
lying) and posture-over-time into events (fall, lying on the floor, long stillness, entering a room). A **live
engine** does this continuously for every camera and keeps a **daily event log**. Alerts go to **Home Assistant**
and **Telegram**. Every evening, a **report** compares the day with the previous 7 days.

The fall rules were **tuned on part of a public fall dataset** and **tested on the rest**, which the tuning never
looked at.

No neural network was trained in this project. Section 6 explains why, what was "trained" instead, and what real
training would be the next step.

---

## 1. The goal (from the POC plan)

| MVP feature | Where it lives |
|---|---|
| Fall alert: fall followed by no recovery within ~30 s, push with snapshot | `app/live.py` (`Camera._fall`), `app/notify.py` |
| Long inactivity / lying on the floor | `app/events.py` (`runs`, `stillness`), `rules.yaml → live:` |
| Room timeline | `app/live.py` (`zone_enter` / `zone_exit` events) |
| Routine: wake, bed, meals, bathroom, active time, night | `app/daily.py` (`metrics`) |
| Daily report vs 7-day baseline | `app/daily.py` (`baseline`, `compare`, `report`) |

Success measures from the plan: fall **recall** on staged falls, **false alarms per day** (target < 1), and **time
from fall to alert** (target < 10 s for detection, plus the 30 s no-recovery wait).

---

## 2. Architecture

```
                 ┌─────────────────── app/live.py: one thread per camera ───────────────────┐
cameras.yaml ──► │ open source  ─► sample 6 fps ─► perception._detect (YOLO pose on the GPU)  │
 webcam 0        │      │                              │                                     │
 phone URL       │      │                              ▼                                     │
 rtsp://...      │      │                 rolling window (last ~6 min of detections)        │
 video file      │      │                              │ every 1 s                           │
                 │      │               tracks.build ─► events.analyze  (same code as uploads)│
                 │      │                              │                                     │
                 │      │      fall state machine · room / posture changes · lying · still  │
                 │      ▼                              ▼                                     │
                 │  annotated JPEG (MJPEG)        Engine.emit(event)                         │
                 └───────────────────────────────────┬──────────────────────────────────────┘
                                                     ├─► data/live/events-YYYY-MM-DD.jsonl
                                                     ├─► notify.send → Home Assistant webhook / Telegram
                                                     └─► 21:00: daily.report → Telegram + HA
uploads (/api/jobs) ─► pipeline.run (whole clip at once) ─► annotated video + result page
```

**Design decision: one rule engine for uploads and live.** `tracks.build` and `events.analyze` are *pure
functions*: same input, same output, no hidden state. The live engine does not re-implement anything. Every second
it re-runs the same functions on a rolling window of recent detections. This keeps the logic in one tested place.
It is cheap because the window holds at most ~2,000 detections.

**Design decision: no Frigate.** The plan listed Frigate (an NVR) for cameras and zones. This app already does
person detection and zones itself. Frigate would add a second detector, MQTT and Docker GPU passthrough on Windows,
for recording that the POC doesn't need.

---

## 3. Each layer, and the logic inside it

### 3.1 Perception: `app/perception.py`
- **Model:** YOLO11-pose from Ultralytics, pre-trained on COCO-keypoints (people photos with 17 labelled joints).
  Output per person: box `[x1, y1, x2, y2]`, 17 keypoints `[x, y, confidence]`, and a detection confidence.
- **Sampling:** 6 frames per second, not every frame. Falls take about 1 s, so 6 fps gives ~6 samples during a
  fall, and it is 5× cheaper than 30 fps.
- **Tracking:** ByteTrack assigns a stable ID per person across frames. ByteTrack is driven by hand
  (`tracker.update`) so that detections it hasn't confirmed are kept, with a negative "fragment" ID.
  `tracks.rejoin` stitches them back.
- **Device:** GPU (`device=0`) when CUDA is available, else CPU. The model is `yolo11m-pose` on a GPU and
  `yolo11n-pose` on a CPU (see §7 for why).
- **`det_conf: 0.15`:** the minimum detection confidence. Lowered from 0.25 because people lying on the floor score
  very low (§7, finding 4).
- **`source_fps`:** an override for files that report the wrong frame rate (§5, the 4× bug).

### 3.2 Posture: `app/posture.py` (pure geometry, no learning)
Angles are measured **from vertical**: 0° = upright, 90° = horizontal.

```
torso = angle(shoulder midpoint → hip midpoint)      thigh = angle(hip midpoint → knee midpoint)
aspect = box width / box height
```

Rules, in order:
1. No torso visible → `lying` if the box is wide (`aspect ≥ 1.4`), else `standing`.
2. `torso ≥ 60°` and (`aspect ≥ 0.9` or `thigh ≥ 60°`) → **lying**.
3. **New:** `torso ≥ 45°` and `aspect ≥ 1.4` → **lying**. This was added because a high, fisheye camera
   foreshortens a body on the floor, so the torso measured 54–59° and read as "bending" (§7).
4. `torso ≥ 30°` → bending.
5. Knees not visible → the per-camera guess `legs_hidden` (counter = standing, table/sofa = sitting).
6. `thigh ≥ 50°`, or the thigh is foreshortened (shorter than half the torso) → sitting; else standing.

Then in `tracks.py` the posture is **smoothed** by a majority vote over 5 samples (~0.8 s). This removes one-frame
flicker.

### 3.3 Tracks: `app/tracks.py`
- **`dedupe` (new):** at a low detection threshold, YOLO returns partial duplicate boxes of the same person. Any box
  that is ≥ 60% inside a more confident box is dropped. Without this, a duplicate started a bogus second track that
  "stole" the lying samples, and neither track looked like a fall.
- **`rejoin`:** a track that starts shortly after another ended, near where it ended, is the same person.
- **`min_samples: 12`:** tracks shorter than ~2 s are dropped (passers-by, false detections).
- **Motion:** foot-point speed in body-heights per second gives walking (≥ 0.35) and running (≥ 1.5).

### 3.4 Events: `app/events.py` (the fall logic)
A fall is reported by **one of three paths**. They exist because each catches a failure the others miss (§7).

**Path A, "lying":** standing or bending, then **lying within 2 s**, with a real drop: either
- the hips move down ≥ 0.25 body-heights, or
- the box shrinks to ≤ 0.7 of its upright height (a high camera sees a fall as the box getting shorter more than
  the hips moving down);

and the person **stays down ≥ `min_down_s` (5 s, tuned)**. Sitting → lying is *not* a fall: that's lying down.
The drop is judged over the first **`settle_s` (0.5 s)** of lying, not on the first lying frame alone: at 6 fps
that frame is often caught *mid-fall* (§7 run 10).

**Path B, "floor_drop":** the hips drop **≥ 0.6 body-heights within 1 s**, and the person doesn't stand again for
`min_down_s`. This catches falls straight toward the camera: the torso stays vertical in the image, so the pose
reads "sitting", but the hips drop fast.

**Path C, "vanished":** an upright person's body drops (box shrinks or hips fall), they are **last seen as a wide
shape** (`aspect ≥ 1.2`, tuned) away from the frame edges, and then **nobody is seen for ≥ 3 s**. This exists
because the pose model often *fails to detect a person lying on the floor* (§7, finding 4). Losing someone right
after their body dropped is itself a fall signature.

**Path D, "collapse" (off by default, `collapse_frac: 0`):** whatever the posture classifier says, the **top of the
box drops ≥ 0.45 body-heights within 1 s while the feet stay put**, and the person isn't that tall again for
`min_down_s`. Built for falls that end *sitting or slumped on the floor*, which at low resolution still read as
"standing" (§7, Le2i). The "feet stay put" check is what separates it from walking away from the camera, which
also shrinks the box but moves the feet up the image. It costs false alarms where there are no rest zones, so it is
enabled per camera (`collapse_frac: 0.45` in that camera's rules file) once the sofa and bed are drawn.

**Always:** a fall that *ends* inside a `kind: rest` zone (bed, sofa) is ignored. "Ends" = where the person is
1–3 s after the trigger, because the trigger frame is often mid-fall, swinging past the sofa (§7 run 11).

Other events: `lying_outside_rest` (lying outside bed/sofa for > N s) and `stillness` (moved < 0.15 body-lengths
for > N s).

### 3.5 Live engine: `app/live.py`
- `open_source`: `0` → webcam (DirectShow on Windows), `http://` / `rtsp://` → stream, anything else → video file
  (played at real speed, optionally looped). Streams reconnect with back-off (2 s, 4 s, … up to 60 s).
- **Rolling window:** the window is `max(lying, stillness, no_recovery) + 30 s` long, so every rule sees enough
  history.
- **Fall state machine** (`Camera._fall`):

  ```
  detect_falls hit ──► FALL_SUSPECTED (logged, snapshot, banner)
        │  got_up_s set                    │  still down for no_recovery_s (30 s)
        ▼                                  ▼
  FALL_RECOVERED (warn → pushed)     FALL_ALERT (alert → pushed, high priority)
  ```

  The same fall is seen again every second (the window re-evaluates). Matching by fall time (±3 s) makes sure each
  fall fires each message once. `tests/test_live.py` checks exactly this.
- **Severities:** `alert` and `warn` are pushed to the family; `info` is only logged. The page/HA status is
  **ALERT** if an alert happened in the last hour, **MONITOR** for a warning, else **NORMAL**.
- **Falling is not the same as lying on the floor** (§3.5.1).
- **The slow rules** (`Watch` and `Camera._rest`; they outlast the rolling window, so they keep their own timers):

  | event | when | severity |
  |---|---|---|
  | `bathroom_long` | location named bath*/toilet*/wc (camera room or zone, e.g. a "Bathroom door" zone in the hallway) for > 30 min, counted also while out of sight behind the door | warn; > 45 min alert |
  | `rest_long` | on a `kind: rest` zone (sofa) > 2 h between 09:00–20:00 | warn |
  | `not_seen` | nobody on any camera > 4 h between 08:00–21:00, while a camera is online | warn |
  | night activity | 23:00–05:00 | counted in the daily report, not pushed |

  Each fires once per episode. `tests/test_live.py::test_watch_bathroom_and_not_seen` walks through a whole
  episode, including "asleep at night is not missing".

#### 3.5.1 Falling vs lying on the floor

They're two different things:
- **falling** is an *event*: upright to the floor, fast, not controlled;
- **lying on the floor** is a *state*, and it can be on purpose (exercise, a floor mat, playing with a child).

A fall can happen without a long lie (they get up), and a long lie without a seen fall (a slow collapse, or the
fall happened off camera). So the system has two separate alerts:

| | what triggers it | message | severity |
|---|---|---|---|
| **FALL — not getting up** | the fall *event* (§3.4), then still down 30 s | "fell and has not got up for 31 s" | alert |
| **On the floor, not moving** | lying outside bed/sofa ≥ 60 s, keypoints barely moving | "lying outside bed/sofa for 60 s, barely moving" | alert |
| **On the floor, moving** | same, but moving ≥ `floor_moving_bl_s` (0.4 body lengths/s) | "…, moving around" | warn (MONITOR) |
| *(already alerted)* | on the floor after a FALL ALERT | logged only | info: no second ping |

- **Motion** (`events.motion`): the median, over the lying stretch, of how far the visible keypoints move per
  second, in body lengths. It's a weak cue (see §7 run 13), so it only chooses between alert and warn: **both are
  pushed**, and a wrong call changes the wording, not whether the family is told.
- **AI second opinion, escalate-only** (`Engine.second_opinion`, off by default): for the unsure case ("on the
  floor, moving") the snapshot goes to a vision model with one question: FALLEN, DELIBERATE or UNSURE?
  Only FALLEN does anything (a new alert). It can never cancel or downgrade a rule's alert, so a wrong or missing
  answer can't hide a fall. It needs `VLM_SECOND_OPINION=1` and `OPENROUTER_API_KEY`, and it sends a home image to
  a cloud API, which is why it's opt-in. The daily cost cap (`MAX_LLM_USD_PER_DAY`) applies.
- **"On the floor" without the lying posture** (`events.mark_low`, `floor_low_ratio: 0.5`): someone who slides
  down and ends **sitting slumped on the floor** isn't "lying", so at first neither alert fired. Now a person also
  counts as on the floor when their box is ≤ half their own **walking** height *at the same spot in the image*
  (the median of the 6 walking samples whose feet were nearest in image height).
  - *Their own height, at the same spot:* a person far from the camera is small, so a fixed pixel height would
    flag everyone at the back of the room. Comparing with themselves where they stood removes the perspective.
  - *Walking, not standing:* the pose model often calls a slumped person "standing", so with "standing" the slumped
    frames became their own reference and never looked low (§7 run 14). Nobody walks while slumped.
  - *Chairs:* sitting on a chair is ~0.6–0.8 of walking height, above the line. A low armchair can be given a
    `kind: seat` zone: sitting there never counts as on the floor (sofas/beds are `rest` zones as before).
  - *Limits:* it needs ≥ 6 walking samples in the 330 s window (normal at home; rare in 10 s dataset clips), and a
    person who fell *towards* the camera has a bigger box, not a lower one; the lying rules catch that case.
- **Room:** the camera's `room:` from `cameras.yaml`, or a `kind: room` zone inside the view.

### 3.6 Notifications: `app/notify.py`
- Home Assistant **webhook** (`/api/webhook/eldercare`) → the shipped automation sends a phone push with the
  snapshot image.
- Home Assistant **states** (`sensor.eldercare_room / posture / status`) → HA's dashboard and history.
- Telegram **Bot API**: `sendPhoto` with the snapshot. Works away from the home Wi-Fi.
- Every sender is optional (skipped when its env var is empty). Failures are logged and never crash a camera. The
  bot token is **redacted** from error logs, because httpx error messages include the URL.

### 3.7 Daily report: `app/daily.py`
- **Visits** = `zone_enter` / `zone_exit` pairs, with visits to the same room merged when < 5 min apart. Brief
  occlusions would otherwise count as many visits.
- **Meals** = kitchen visits ≥ 3 min, labelled by clock: before 10:30 breakfast, before 15:00 lunch, before 18:00
  tea, else dinner.
- **Wake** = first active posture after 04:00. **Bed** = last lying in a bedroom after 19:00.
  **Night** = visits between 23:00 and 05:00.
- **Baseline** = mean, min and max of up to 7 earlier days.
- A metric is **flagged ⚠️ only if it is outside the 7-day range AND far from the mean** (> 25%, or > 45 min for
  times). Requiring both avoids flagging normal day-to-day wobble.
- The LLM is **optional**: the report text is built from rules, and the LLM (if a key is set) adds 1–2 warm
  sentences about the ⚠️ items. Facts never come from the LLM.

---

## 4. Testing strategy (how we know each piece works)

| Level | What | Command |
|---|---|---|
| Unit | Posture angles, each fall path, sitting/sofa/edge non-falls, zones, LLM cache/cap/fallback | `python -m pytest -q tests/test_events.py tests/test_posture.py ...` |
| Unit | Live state machine: alert exactly once at ~34 s; recovered → no alert; room/posture events | `tests/test_live.py` |
| Unit | Notifications: nothing sent without config, correct HA/Telegram calls, token never logged | `tests/test_notify.py` (httpx `MockTransport`, no network) |
| Unit | Daily metrics from a hand-written day; baseline flags | `tests/test_daily.py` |
| End to end | Upload API on a generated fall clip with the real model | `tests/test_api.py` |
| End to end | Live engine with the real model on `videos/synthetic_fall.mp4` as a camera | manual (§9) |
| Benchmark | Fall recall / false alarms on 192 real clips | `scripts/eval_mcfd.py` (§5) |

Run everything: `$env:YOLO_OFFLINE="true"; .venv\Scripts\python -m pytest -q` → **34 passed**.

**Testing pattern used here:** each unit test builds synthetic samples (boxes plus keypoints, see
`tests/conftest.py`) for a known situation and asserts the rule's decision. They need no model, no GPU and no
video. That is why the rules are pure functions.

---

## 5. The dataset and how scoring works

**Multiple Cameras Fall Dataset** (Auvinet et al., University of Montreal, 2010): 24 scenarios (`chute01–24`), each
filmed by 8 wide-angle ceiling cameras, one actor. Location:
`C:\Users\ABC\Downloads\archive (4)\dataset\dataset\chuteNN\camN.avi`, labels in `..\data_tuple3.csv`.

- **Labels:** `data_tuple3.csv` rows are `chute, cam, start_frame, end_frame, label`; `label = 1` marks a fall
  window. Adjacent label-1 windows are merged into one fall. Findings:
  - **chute23 also contains falls**, despite the "23–24 have no falls" claim found online. Always trust the label
    file over secondary sources.
  - Only **chute24 has no falls**.
  - In total there are **200 falls** across 192 clips.
- **The 4× bug:** the `.avi` files claim **120 fps** but are really **30 fps**. The frame counts match the 30 s–4
  min scenarios only at 30 fps. Reading them at 120 fps ran every timing rule 4× too fast: "stay down 2 s" really
  meant 8 s. Fix: `source_fps: 30`. Lesson: check the time base of any dataset before trusting results.
- **Scoring** (`scripts/eval_mcfd.py`):
  - A detection is a **hit** if it starts inside a labelled window, widened by 2 s before and 4 s after.
  - Any other detection is a **false alarm**.
  - **Recall** = hits / labelled falls.
  - **Per scenario** = the fall was caught by *any* of the 8 cameras. A home has one camera per room, so per-camera
    recall is the realistic but pessimistic number; per-scenario is optimistic.
  - **Alert-level false alarms** = false alarms where nobody was up after 30 s, i.e. ones that would actually reach
    the family's phone.
- **Caching:** the YOLO output of each clip is saved once in `data/mcfd_cache/<model>-c<conf>/`. Posture is
  recomputed from the cached keypoints at scoring time, so **any rule change re-scores all 192 clips in about a
  minute** instead of hours. This made fast experiments possible. Change the model, `det_conf` or `imgsz`, and a
  new cache folder is made.

### 5.1 Second dataset: Le2i (home-like rooms)

**Le2i fall dataset** (Charfi et al., Univ. Burgundy, 2013): one fixed camera at about head height, 320×240 at
25 fps (Home_01 is 24 fps), rooms that look like a home: a kitchen/coffee room and a living room. Extracted from
`archive (6).zip` to `D:\datasets\le2i\` (only the four annotated rooms: Coffee_room_01/02, Home_01/02 = 130
videos, 96 falls, 34 without). Each `Annotation_files\video (i).txt` starts with the first and last *frame* of the
fall, then one box per frame. Three files have no fall lines and are treated as no-fall.

- `scripts/le2i_labels.py` converts them into the same `labels.csv` your staged recordings use, so the **same
  scorer** (`scripts/eval_staged.py`) works on both. One scorer for every dataset = numbers you can compare.
- **Clips end ~4 s after the fall.** The rules want the person down for 5 s, so on the raw clips nothing can ever
  fire (run 6 below). `--hold 10` repeats the last frame for 10 s: "they stayed as they were when the camera
  stopped". A person standing at the end stays standing, so it can't invent falls. Lesson: **know how your test
  data was cut**; a harness that doesn't match how the product runs measures the harness.
- Lecture_room and Office (60 videos) have no annotations and weren't extracted.
- **The camera moved.** A contact sheet of every video's first frame (do this for any dataset!) showed Home_01/02
  were filmed from **3 positions**. Rest zones belong to a camera *view*, not a folder: `learning/le2i_zones/`
  holds one file per view (coffee, home_A/B/C), and `le2i_labels.py` assigns each video its view (`VIEWS`, read by
  eye) and copies the zones next to the videos, exactly like a live camera's rules file.

### 5.2 Third dataset: UR Fall (a fresh test, plus "on the floor on purpose")

**UR Fall Detection** (Kwolek & Kepski, Univ. of Rzeszów, 2014, fenix.ur.edu.pl/~mkepski/ds/uf.html, free for
research): 30 falls and 40 daily activities, Kinect at chest height, 320×240 at 30 fps. Downloaded to
`D:\datasets\urfall\` (the cam0 videos + `urfall-cam0-falls.csv` / `-adls.csv`, 117 MB).
- Each video is **depth | RGB side by side**; `scripts/urfall_labels.py` keeps the RGB half. (Look at a frame
  before writing a converter: the first contact sheet showed this immediately.)
- The csv labels **every frame**: -1 not lying, 0 falling/lying down, 1 lying. Falls become `fall_any` (first 0 →
  first 1); activities with lying frames become `lookalike` ("lying down on purpose": on a bed, a sofa, or **the
  floor**: ADL 30–35 crawl, kneel and lie on the office floor). That's the data §8 said was missing.
- 5 rooms (falls all in an office; activities in two bedrooms, a dark living room, an empty room, the office).
  Zones in `learning/urfall_zones/`: beds/sofa as `rest`, the bedroom armchair as `seat`.
- Used **only as a test**: no rule threshold was chosen on it. (The GRU *trains* on it in TRAINING.md run 4; its
  val/test sets stay MCFD + Le2i, so that's still fair.)
- Many fall clips **start mid-fall** (labelled fall start 0–1 s). The fall rules need ~2 s of upright before; the
  stay-down "on the floor" alert doesn't, which is exactly why the two alerts are separate.
- UP-Fall needs a sign-up form and per-trial Google Drive downloads of PNG frames (hundreds of GB): skipped.

### 5.3 Fourth dataset: CAUCAFall (a real home, high corner camera)

**CAUCAFall** (Eraso et al., Univ. del Cauca, 2022, Mendeley Data 7w7fccy7ky, CC BY 4.0): 10 subjects × 5 falls
(backwards, forward, left, right, from sitting) + 5 activities (hop, kneel, pick up, sit down, walk), one room of a
real home, a camera high in a corner, 720×480 at 20 fps, day, artificial light and night (infrared).
`scripts/cauca_labels.py D:\datasets\cauca` downloads it straight from the Mendeley API.
- Every frame has a YOLO-format `.txt` (class 1 = fall). Downloading only the 100 `.avi` (~200 MB) and finding the
  first fall frame by **binary search** over the label files (~8 requests per clip, not ~250, not the 10 GB of
  PNGs) works because the label is 0…0 then 1 to the end: checked on one whole clip first. Check an assumption
  once, cheaply, before building on it.
- **"Fall" there means "already down"**: a contact sheet at −1 s / 0 s / +1 s showed standing, then on the floor.
  So the labelled fall is the first fall frame ± 1 s. Every dataset defines "the fall" differently (Le2i: the
  falling motion; UR: falling *and* lying down; CAUCAFall: on the floor). Read the definition, then look.
- One clip (S9 Fall forward) has no fall frames: left out, since as "none" a correct detection would score false.
- Mendeley once answered a download with an error message in a 200 OK, which was saved as `S3_Hop.avi`. The
  converter now treats that as a failed download; a video that won't open is the symptom to look for.

---

## 6. "Training" in this project, honestly

**What was not trained:** the neural network. YOLO11-pose came pre-trained from Ultralytics. Training or
fine-tuning a pose model needs thousands of images with keypoint labels. This dataset has only *time windows*
("a fall happens between frames 1083 and 1113"), not boxes or keypoints, so it can't train a pose model.

> **Update:** a small neural network *was* later trained, on keypoint sequences rather than images. That one is
> real training from scratch, explained step by step in [TRAINING.md](TRAINING.md). Result: about as good as the
> rules, so it isn't wired in yet.

**What was "trained": the thresholds.** A rules-based detector has a handful of numbers (`min_down_s`,
`height_ratio`, `vanish_min_aspect`, …). Choosing them from labelled data is the same idea as training: fit
parameters to data. Here there are 4 parameters, not millions.

**The discipline that makes it honest: a train/test split.**
- **Tune split = chute01–16** (128 clips, 136 falls): look at these, analyse misses, pick thresholds.
- **Test split = chute17–24** (64 clips, 64 falls, including the no-fall chute24): never looked at while tuning,
  only scored at the end.
- The split is **by scenario, not by camera**. The 8 cameras of one scenario show the *same* fall. Putting cam1 in
  tune and cam2 in test would leak the answer and inflate the test score.

**How the thresholds were chosen:**
1. **Feature analysis** (`scripts/fall_features.py`): for every fall *candidate* on the tune split, record its
   measurements (`drop_s`, `h_ratio`, `hip_drop`, `down_s`, `aspect_down`, `torso_down`) and whether it was a real
   fall. Then compare the distributions. Finding: real falls stayed down a median **7.2 s**; the false alarms
   (sinking into an armchair or sofa) were back up after **3.3 s**. False "vanished" cases were last seen
   **upright-shaped** (aspect 0.77, against 1.41 for real ones).
2. **Grid search** (`scripts/tune_falls.py`): try every combination of `min_down_s ∈ {2,3,4,5}`,
   `vanish_min_aspect ∈ {0,0.9,1.0,1.2}`, `floor_height_ratio ∈ {1.0,0.8}` and `height_ratio ∈ {0.6,0.7}`
   (64 settings) on the tune split. Score each with `recall − 0.5 × false_alarms_per_hour / 10`, which encodes how
   much recall you would give up for fewer false alarms.
3. **Report the winner once on the test split.**

---

## 7. Experiment log (what was tried, what happened, what it taught)

| # | Change | Data | Recall (per camera) | False alarms | Lesson |
|---|---|---|---|---|---|
| 0 | Original rules, nano model, fps read as 120 | first 40 clips | **0 falls** | 0 | The fps bug plus rules built for a side-on camera |
| 1 | `source_fps: 30`; wide-box lying; `height_ratio`; falls can start bending; rest-zone skip | chute01–03 | 4/21 (19%) | 0 | Posture fixed; detection gaps remain |
| 2 | Compared nano / small / medium pose on a frame of a person lying face-down | 1 frame | nano conf 0.07, small 0, **medium 0.67** | — | Bigger model sees bodies on the floor far better |
| 3 | Medium model, `det_conf 0.15` + dedupe + floor_drop + vanished paths | all 192 clips | 95/200 (48%), per scenario 24/25 | **85 (35/h)** | Recall up, but look-alikes (sinking into a sofa) fire |
| 4 | Same as #3, split | tune / test | 54% / 33% | 25/h / 46/h | Test is worse than tune: baseline for honest comparison |
| 5 | Feature analysis + grid search on tune → `min_down_s 5`, `vanish_min_aspect 1.2` | tune | 35%, per scenario 16/17 | **3.2/h** (alert-level **0**) | Big false-alarm cut, some recall given up |
| 5 | Same thresholds, **held-out test** | test | **27%**, per scenario 6/8 | **19.5/h** (alert-level **0.8/h**) | Generalises partly. The gap from tune is overfitting |
| — | Final rules, model comparison on the same 48 falls (chute01–05) | 40 clips | nano@0.25: 9/48 · **medium@0.15: 18/48** | 0 / 0 | Medium model doubles recall |
| 6 | Run-5 rules on **Le2i**, never tuned on, raw clips | 130 videos | **4/96 (4%)** | 0 | Zero hits *and* zero false alarms = suspicious: something systematic |
| 7 | Same, `--hold 10` (clip end ≠ person got up) | 130 videos | **63/96 (66%)** | 3 (≈ 6/h), alert-level **0** | Pose was fine all along (conf 0.9); the clips were too short for `min_down_s` |
| 8 | Error analysis of the 33 misses | Le2i | — | — | 16 end **seated/slumped, read as "standing"** · 8 got up within 5 s · 5 lying but missed · 4 not detected |
| 9 | New Path D "collapse", grid on **tune** (MCFD 01–16 + Le2i Coffee) | tune | 47% → **62%** (frac 0.35) | 3.5/h → 13.8–16.7/h | Score says **off wins**. On Le2i alone +10 falls for +1 FA; on MCFD +6 falls for +10 FA (sofa/armchair, zone-fixable). None of them alert-level |
| 10 | The 5 "lying but missed": first lying frame is mid-fall (box 0.75 of upright vs threshold 0.7). `settle_s` grid | tune → **test** | tune 91 → 97/195; **test 37 → 44/101** | tune +1, test ±0 | A sampling-rate bug, not a threshold problem: look at the frames |
| 11 | Rest zones per Le2i camera view; collapse grid again | tune | zones cost 3 falls (trigger frame mid-fall next to a bench) → fixed by judging the zone where the person settles: 97/195 with or without zones | test −1 FA | Zones are a trade-off: they also hide falls *next to* furniture. Path D still loses on MCFD |
| 12 | **Trained GRU** on keypoint sequences (TRAINING.md), mirror augmentation, threshold on val | **test** | model 35% · rules 44% · rules OR model 49% | 7.9/h · 15.8/h · 17.8/h | ~200 falls: a learned model ties hand-written rules; they make different mistakes |
| 20 | **`imgsz` 960 instead of 640** (yolo11m-pose, GPU), live engine + GRU, own YOLO cache | **test** (same as run 19) | FALL ALERT 28 → **26/37**; family told 76% → **70%** | 3.0/h → **3.7/h** | Dropped. Also too slow: 0.7× real time for *one* camera on the GTX 1650. More pixels didn't help: the misses are angle/occlusion, not size |
| 19 | **GRU wired into the live engine** as a second fall signal (`live.fall_model`; a hit = fall at t-1 s, needs `confirm_s` 5 s still down, then the same 30 s state machine). Decided before the run: keep it only if more falls reach the family with no extra false pushes | **test** (Le2i Home held 70 s; MCFD 17-24 + Le2i Home raw) | FALL ALERT 24 → **28/37**; family told 68% → **76%**, median 31 s | 3.0/h → **3.0/h** (all 'fell, got up' check-ins from the rules; the model alone never pushes those) | Kept on. The model's falls are the slumped/foreshortened ones the rules read as sitting |
| 18 | **GRU, 3 seeds each**: + UR Fall, and + UR Fall + CAUCAFall (TRAINING.md run 5) | **test** | UR: 45/47/43% (mean **45%**); UR+CAUCA: 46/49/61% (mean **52%**); rules 44% | UR 11.4/h; UR+CAUCA 11.9/h; rules 15.8/h | Fewer false alarms than the rules in 6/6 runs, more falls in 5/6: not luck. Seeds spread ±8 points: report the spread, not the best run |
| 17 | **CAUCAFall, never tuned on** (`eval_staged --hold 10`, `eval_live --extra`) | 49 falls, 50 ADL | fall seen **63%**; family told **63%** | 1 "possible fall" (logged, picking up an object); 0 false pushes | 8 of 9 *forward* falls missed: the high corner camera sees a forward fall head-on, so it's foreshortened, not "lying" |
| 16 | **GRU + UR Fall in train** (TRAINING.md run 4; val/test unchanged) | **test** | model **45%** · rules 44% · rules OR model 51% | 12.5/h · 15.8/h · 19.8/h | +70 clips (incl. 16 look-alikes) and the model now beats the rules on both axes. One seed: encouraging, not proven |
| 15 | **UR Fall, never tuned on**: `eval_staged --hold 10` and `eval_live --extra` | UR 30 falls, 40 ADL | fall seen **63%**; FALL ALERT 63%; family told **70%** (floor alert adds 2) | 2 "possible fall" (logged) while lying down on purpose; **0** false pushes | Misses = clips starting mid-fall: no upright history for the fall rules; the floor alert covers some |
| 14 | **Floor state without the lying posture** (`floor_low_ratio`, `tune_floor.py`) | tune / **test** | on-the-floor alert 69 → **75%** tune; test 62 → 62%; eval_live family told 72 → **74%** | +0 (10 s-run proxy) | The test misses aren't "low": person undetected (6), fell towards the camera (2), ~55 px tall with < 1 s of walking (6). Walking-height reference, not standing (standing gave 86% on tune by accident of ties, but slumped "standing" frames referenced themselves) |
| 13 | **Replay through the live engine** (`eval_live.py`): Le2i with the last frame held 70 s ("never gets up") | Le2i 96 falls + all 2.9 h raw | **FALL ALERT 71%** at median **31 s**; floor alert adds 1; family told **72%** | **0** false FALL/floor alerts; 2.4/h "fell, got up" warnings | The floor alert rarely adds coverage: it needs "lying", the same thing the fall rules need. Fallen people "move" 0.15 bl/s (jitter + settling), so the moving threshold went 0.1 → 0.4 |

Findings in plain words:
1. **Check the data's time base.** One wrong number (fps) made a working system look completely broken.
2. **The camera angle changes everything.** Rules written for a side-on camera (torso angle, hip drop) fail on a
   ceiling fisheye. The box *shrinking* is the stronger signal from above.
3. **Look at the false alarms, not just the misses.** Most false alarms were *the same moment across several
   cameras*. That pointed to a real action in the scene, not noise: the dataset's deliberate look-alikes (sitting
   or flopping into an armchair or sofa).
4. **The pose model is the ceiling.** In 4 of 8 cameras of chute01, the fallen person was detected in only 2–4 of
   36 frames, even with the medium model. COCO is mostly upright people, so bodies lying flat seen from above are
   out of its training distribution. No rule can fix a person the model can't see. Path C (vanished) works *around*
   it; real training (§10) would fix it.
5. **Tune and test on separate scenarios.** On the tune set, false alarms went from 25/h to 3.2/h; on the test set
   only from 46/h to 19.5/h. Without the split we would have reported the flattering number.
6. **"Zero hits and zero false alarms" is a bug signal, not a result.** A weak detector misses *some* falls and
   fires on *some* look-alikes. Missing everything while firing on nothing means a gate that can never open, here
   the 5 s stay-down check on clips that end 4 s after the fall. First print what the model saw, then blame the
   rules.
7. **Sort misses into causes before changing anything.** The 33 Le2i misses split into 4 different problems. Only
   one (16 seated/slumped falls) was worth a new rule; "got up within 5 s" is the price of `min_down_s` that we
   chose deliberately (it cut the sofa-flop false alarms on MCFD).
8. **Decide the scoring rule before you look, then obey it.** Path D raised recall by 15 points and the score
   still said no, because false alarms quadrupled. Changing the score after seeing that would be fooling ourselves.
   Instead the rule ships *off*, as an option to try on your staged session 1 *with rest zones*, where its false
   alarms (sofa/armchair) should disappear. That's a new, pre-stated test, not a re-scored old one.
9. **A better tune score that doesn't move the test is a warning.** The floor rule's first version went 69 → 86% on
   tune and 62 → 62% on test. Printing the test misses showed *why* (self-reference, people too small), and the
   fixed version is the one that ships. Test numbers that don't follow tune mean "look", not "tune harder".
10. **More data beat more cleverness.** Three rounds of model changes (runs 12, mirror, stay-down) moved the GRU's
    test recall by a few points; adding one small dataset (70 clips) moved it 10, a second one (99 clips) 7 more
    on average over 3 seeds (run 18).
11. **Motion while on the floor is a weak cue, now measured from both sides.** Fallen: median 0.16 body lengths/s
    (90% below 0.40). On the floor on purpose (UR ADL, only 5 usable): median 0.37. They overlap, so `0.4` stays a
    wording choice between alert and warn, never a reason not to tell the family.
12. **Filter the same way everywhere.** The live stream showed one seated person as two overlapping skeletons
    and "3 person(s)": with `det_conf: 0.15` the model returns partial duplicate boxes, and only `tracks.build`
    (the rules) dropped them, not the live overlay or its person count. The live loop now runs the same
    `tracks.dedupe` right after detection (living-room clip: 53 → 50 detections, duplicates in 3 of 50 frames).
    Skeleton lines are now 1 px thick, so the person underneath stays visible.

---

## 8. Where it stands (honest)

| Measure | Value | Target | Verdict |
|---|---|---|---|
| Fall recall per camera (test) | 27% | high | ❌ limited by the pose model on ceiling fisheye views |
| Fall caught by any camera (test) | 75% | — | partial |
| False "possible fall" (test) | 19.5/h | — | logged only, not pushed |
| False **FALL ALERT** (still down 30 s) (test) | 0.8/h of video | < 1/day | ❌ not yet; needs rest zones + staged-fall data |
| Le2i (home-like, one camera, never tuned on) | 66% of 96 falls, 0 alert-level false alarms | — | much better than MCFD's ceiling fisheye |
| Detection delay after the fall starts | ~1 s (median) | < 10 s | ✅ |
| Fall → FALL ALERT | ~1 s + 30 s wait | as designed | ✅ (unit-tested, smoke-tested) |

**Why the dataset is harsher than a real home:**
- The actor gets up after a few seconds in nearly every scenario, so the most important case (*not* getting up)
  is barely in the data.
- Ceiling fisheye cameras at 720×480 MPEG-4 are a worst case for COCO pose models. A phone on a shelf at chest
  height sees people side-on, which is what COCO was trained on.
- The benchmark runs **without rest zones**, because the furniture moves between scenarios. In a home you draw the
  sofa and bed once (`scripts/zone_helper.py`), which removes the biggest false-alarm source found here.

**Done since:** bathroom > 30/45 min, sofa > 2 h, not seen > 4 h, the falling-vs-floor split with a motion cue,
the escalate-only VLM second opinion, and `scripts/eval_live.py` (§7 run 13); the posture-free floor state for
seated/slumped falls (run 14); UR Fall and CAUCAFall as untuned test sets (runs 15, 17); the GRU retrained with
them over 3 seeds (runs 16, 18).

**Still open:**
- **People the model can't see** are now the largest miss group on every dataset (Le2i test: 6 of 14 floor-alert
  misses). That's the detector fine-tune (§10 step 3).
- **Deliberate floor activity:** still only 5 usable examples (CAUCAFall's kneeling is labelled as whole
  "no fall" clips, not as on-the-floor stretches). Not a calibration.
- **Forward falls towards a high camera** (CAUCAFall: 8 of 9 missed): foreshortened, neither "lying" nor much
  lower. A detector fine-tune on such frames, or the GRU, which already learns from them.
- **The GRU now beats the rules** over 3 seeds (run 18), and is wired into `live.py` as a second fall signal
  (run 19: 76% of stay-down test falls reach the family, no extra false pushes).
- **Real-home numbers:** false alerts per *day* need a laptop running for days with real cameras. Datasets give
  per hour of acted video, which is far busier than a real day.
- **Stays POC-only:** YOLO is AGPL; a product would swap to a permissively licensed pose model (MediaPipe).

---

## 9. Reproduce everything yourself

```powershell
cd "D:\Autonoetic Edge\ElderCare"
$env:YOLO_OFFLINE = "true"

# tests (47)
.venv\Scripts\python -m pytest -q

# benchmark: the first run caches YOLO (~1.5 h on the GTX 1650 with the medium model); re-runs take ~1 min
$env:POSE_MODEL = "models/yolo11m-pose.pt"
.venv\Scripts\python scripts\eval_mcfd.py "C:\Users\ABC\Downloads\archive (4)" --split tune
.venv\Scripts\python scripts\eval_mcfd.py "C:\Users\ABC\Downloads\archive (4)" --split test --list

# the "training" step
.venv\Scripts\python scripts\fall_features.py "C:\Users\ABC\Downloads\archive (4)"   # -> data\fall_features.csv
.venv\Scripts\python scripts\tune_falls.py   "C:\Users\ABC\Downloads\archive (4)"

# Le2i: unzip the 4 annotated rooms to D:\datasets\le2i, then
.venv\Scripts\python scripts\le2i_labels.py D:\datasets\le2i
.venv\Scripts\python scripts\eval_staged.py "D:\datasets\le2i\Home_01\Home_01\Videos" --hold 10 --list
.venv\Scripts\python scripts\tune_collapse.py "C:\Users\ABC\Downloads\archive (4)" D:\datasets\le2i

# the "on the floor" height rule (floor_low_ratio)
.venv\Scripts\python scripts\tune_floor.py "C:\Users\ABC\Downloads\archive (4)" D:\datasets\le2i

# CAUCAFall: downloads itself (§5.3)
.venv\Scripts\python scripts\cauca_labels.py D:\datasets\cauca
.venv\Scripts\python scripts\eval_staged.py D:\datasets\cauca\Videos --hold 10

# UR Fall: download the cam0 videos + 2 label csvs to D:\datasets\urfall (§5.2), then
.venv\Scripts\python scripts\urfall_labels.py D:\datasets\urfall
.venv\Scripts\python scripts\eval_staged.py D:\datasets\urfall\Videos --hold 10

# what the family would receive: replay clips through the live engine (cached, no GPU needed, ~15 min)
.venv\Scripts\python scripts\eval_live.py "C:\Users\ABC\Downloads\archive (4)" D:\datasets\le2i --extra D:\datasets\urfall\Videos D:\datasets\cauca\Videos

# train the fall model from scratch (learning/TRAINING.md)
.venv\Scripts\python scripts\train_fall_model.py "C:\Users\ABC\Downloads\archive (4)" D:\datasets\le2i --extra D:\datasets\urfall\Videos D:\datasets\cauca\Videos

# your own recordings (learning/STAGED_TEST.md)
.venv\Scripts\python scripts\eval_staged.py data\staged\session1 --list

# live, with the synthetic fall as a camera
copy cameras.example.yaml cameras.yaml       # edit sources first
.venv\Scripts\python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
# open http://localhost:8000/live
```

The saved outputs of these runs are in `learning/results/`.

---

## 10. Next steps, and where real training comes in

1. **Staged falls on your own setup** (most valuable): 2–3 phones at shelf height, falls onto a mattress, lying
   still for 40 s, and normal sitting and sofa lying. This measures what matters: FALL ALERT recall and false
   alerts per day, from *your* camera angles.
   **Started:** the recording/labelling protocol is [STAGED_TEST.md](STAGED_TEST.md), scored by
   `scripts/eval_staged.py` (same matching as `eval_mcfd.py`, plus FALL-ALERT recall and latency).
2. **Rest zones per camera:** draw bed and sofa polygons. Expect most remaining "possible fall" noise to disappear.
3. **Real training: fine-tune the detector on fallen people.** Extract frames from these videos (and your staged
   ones) where a person lies on the floor, label boxes, possibly keypoints (tools: CVAT or Label Studio), and
   fine-tune YOLO11-pose with `yolo pose train data=... model=yolo11m-pose.pt epochs=50`. Keep the same scenario
   split: never train on test scenarios. This attacks finding 4 directly.
4. **Temporal model (the plan's ST-GCN idea):** once you have many labelled sequences, train a small GRU or 1D-CNN
   on 30-frame keypoint windows. Rules stay as a readable fallback.

---

## 11. Exercises to check your understanding

1. In `rules.yaml` set `fall.min_down_s: 2.0` and re-run `eval_mcfd.py --split tune`. Which number goes up, which
   goes down, and why?
2. Open `learning/results/fall_features_tune.csv` in Excel and filter `real_fall = 0`. Find the clip and time of
   one false alarm and look at that moment in the video. What was the actor doing?
3. Why is the split by *chute* and not by *camera*? What would happen to the test score otherwise?
4. In `tests/test_live.py`, why does the alert arrive between 33 and 36 s, and not exactly at 34 s?
5. Path C triggers on "the person disappeared". Name two everyday situations in a home that could trigger it
   falsely, and which existing setting (or zone kind) reduces each.

---

## 12. Glossary

- **Pose model / keypoints:** a network that finds people and 17 body joints (nose, shoulders, hips, knees,
  ankles, …).
- **COCO:** the public image dataset the pose model was trained on, mostly upright people in everyday photos.
- **Tracking (ByteTrack):** keeping the same ID for the same person across frames.
- **Recall:** of the real falls, the fraction we caught.
- **False alarm:** we said "fall" and there was none. Reported per hour of video.
- **Tune / test split:** tune parameters on one part of the data, report on another part never used for tuning.
- **Overfitting:** parameters that fit the tune data better than they generalise to new data (tune 3.2/h against
  test 19.5/h).
- **Grid search:** try every combination of a few parameter values and keep the best by a stated score.
- **Rest zone:** a polygon (bed, sofa) where lying down is normal and never a fall.
- **No-recovery window:** the 30 s wait after a suspected fall before alerting, which filters falls the person
  recovers from.
