# Connecting the home Hikvision cameras: what was changed, and why

A change log for learning. For each change it gives the **problem** you saw, **how it was diagnosed**, the
**numbers** the decision was based on, **what changed** (file and code), and **how to check it yourself**.
Commands assume PowerShell in `D:\Autonoetic Edge\ElderCare`. `USER:PASS` stands for the DVR login. The real one
is only in `cameras.yaml`, which git ignores.

Date: 2026-10-19.

---

## 0. Summary

| # | Problem you saw | Root cause | Change |
|---|---|---|---|
| 1 | How do we connect the cameras? | The cameras are wired to a Hikvision **DVR** (HVR 4.0) | `cameras.yaml`: one RTSP stream per DVR channel |
| 2 | "It is not live" | Frames queued up in the decoder's buffer, plus the DVR clock is ~4 min 40 s fast | `LatestFrame` in `app/live.py`: always take the newest frame |
| 3 | Low fps | The GPU does ~15 pose frames/s in total, shared by 5 cameras | Per-camera `sample_fps`: living room 6, outdoor 2 |
| 4 | "Use nothing until I press a button" | The engine started at server launch | **Start live / Stop live** button and `/api/live/start`, `/api/live/stop` |
| 5 | Person on the floor not flagged | A top-down camera shows a body on the floor as a diagonal, so it was classed as "bending" | Per-camera posture thresholds in `videos/home/living.yaml` |
| 6 | Furniture and a cloth read as people | The pose model sometimes finds "people" in shapes | New zone kind `ignore`, plus `seat` and `rest` zones |

---

## 1. Connecting the DVR

**What we found** (from your photos of the DVR menus and a few network checks from the laptop):
- The box is a Hikvision **DVR**, with analog cameras wired in. The "IP Camera" list was empty because these
  cameras aren't network cameras; the DVR itself streams each channel.
- DVR address: `192.168.1.5`. Laptop: `192.168.1.6`, on the same network.
- Port 554 (RTSP, the video-stream protocol) was open, which means the DVR can stream video to the laptop.

```powershell
Test-NetConnection 192.168.1.5 -Port 554      # TcpTestSucceeded : True
```

**Hikvision RTSP address format:**

```
rtsp://USER:PASS@192.168.1.5:554/Streaming/Channels/<channel><stream>
  channel 1..8,  stream 01 = main (full resolution), 02 = sub-stream (small, light)
  e.g. 502 = camera 5, sub-stream
```

We use the **sub-stream** (960x576). The model resizes every frame to 640 pixels anyway, so the main stream would
only cost more decoding work for no gain. On the DVR the sub-stream was set to **H.264** (the safest codec for
OpenCV) at 12–15 fps, under Camera → Video Parameters → Encoding Parameters → Sub-stream.

**Special characters:** an `@` in the password must be written `%40`. Otherwise the address parser takes
everything after the first `@` as the host name.

**Finding the cameras:** we tried channels 1–16. Channels 1–5 gave real pictures. Channels 6–8 were black (brightness
6.7: nothing plugged in). Channels 9–16 returned `404` (the DVR has only 8 channels).

| Channel | View | Name in `cameras.yaml` |
|---|---|---|
| 1 | porch with the swing | porch |
| 2 | garden | garden |
| 3 | side path | sidepath |
| 4 | outside stairs | stairs |
| 5 | living room, from the ceiling corner | living |

**Check it yourself:**
```powershell
.venv\Scripts\python.exe -c "import cv2;c=cv2.VideoCapture('rtsp://USER:PASS@192.168.1.5:554/Streaming/Channels/502');ok,f=c.read();print(ok, f.shape if ok else '')"
# True (576, 960, 3)
```

**Lesson:** always use a separate, view-only user for the app, never `admin`. The password ends up in a file.

---

## 2. "It is not live": the buffer problem

**Symptom:** the time burned into the picture by the camera was minutes ahead of the app's own timestamp
(bottom-left corner), and the gap was different on each camera (1.5 to 3 minutes).

**Two causes:**
1. **Buffered frames (the real bug).** The old loop called `cap.grab()` on *every* frame from the camera (25 fps)
   and ran the model on some of them. When the model is slower than the camera, frames wait in the decoder's
   buffer and the app reads older and older video. The delay grows over time. That's why each camera had a
   different gap.
2. **The DVR's clock is wrong.** At 21:33:22 on the laptop, the DVR's picture said 21:38:02, so the DVR runs about
   4 min 40 s fast. This doesn't affect the app, but it makes a delay look bigger than it is. To fix it, turn on
   NTP time sync on the DVR (Settings → Network → NTP).

**Fix:** `LatestFrame` in [app/live.py](../app/live.py). A background thread reads the stream non-stop and keeps
**only the newest frame**. The analysis loop always takes that frame, so if the model is slow it skips frames
instead of falling behind. Recorded video files don't use this: there we want every frame, in order.

```python
class LatestFrame:
    def _pump(self):            # its own thread: decode non-stop, keep only the newest
        while self.ok:
            ok, frame = self.cap.read()
            with self.cond:
                self.frame, self.seq = frame, self.seq + 1
                self.cond.notify_all()

    def grab(self):             # the analysis loop: wait for a frame NEWER than the last one it was given
        ...
```

**Verification:** a frame taken from the living room stream at 21:33:23 (laptop clock) carried the app timestamp
21:33:22, so about 1 second of delay. Test: `test_latest_frame_skips_the_backlog_and_ends_with_the_stream` in
[tests/test_live.py](../tests/test_live.py). A fake camera produces frames fast while the test "model" sleeps
0.5 s. The next frame it receives must be the newest one, not the next one in the queue.

**Lesson:** with live video, a slow consumer has to **drop** frames, not queue them. This is the most common
"my live view lags" bug in OpenCV projects.

---

## 3. Low fps: the GPU budget

**Measured, not guessed:**
```
nvidia-smi                 → NVIDIA GeForce GTX 1650 with Max-Q Design, 4 GB
torch.cuda.is_available()  → True        (so the GPU IS being used)
yolo11m-pose, 640 px       → ~15 frames/s on this GPU
```
15 frames/s is the **total** for all cameras. With 5 cameras each wanting 6 frames/s (30 frames/s in all), each
camera got only ~2.8. We also tried FP16 ("half precision", normally faster): it ran slower on this GPU, so we
don't use it.

**Decision:** spend the budget where falls matter. Indoors, where the person lives, gets 6 frames/s (what the fall
rules were tuned for). The 4 outdoor cameras get 2 each. Total: 6 + 4×2 = 14 frames/s, which fits.

**Change:** `cameras.yaml` now accepts `sample_fps` per camera (a one-word change in `Camera.__init__`, see
[app/live.py](../app/live.py)):
```yaml
- {name: living, source: "rtsp://...502", room: Living room, sample_fps: 6, rules: videos/home/living.yaml}
- {name: porch,  source: "rtsp://...102", room: Porch, sample_fps: 2}
```

**Result:** living room 5.8–5.9 fps, outdoor cameras 2.0, GPU 44% busy.

**Lesson:** "fps" means two different things here. The camera sends 15–25 frames/s. The app *analyses* 2–6.
The video on the page only shows analysed frames, so it looks choppy, and that's expected. Fall detection needs
~6 analysed frames/s, not 25.

---

## 4. Start / Stop live button

**Request:** nothing should run (no camera connections, no GPU use) until you press a button.

**Change:**
- [app/main.py](../app/main.py): the server no longer starts the cameras when it launches. New endpoints:
  `POST /api/live/start` connects the cameras in `cameras.yaml`, and `POST /api/live/stop` stops them, waits for
  the camera threads to finish, then frees the model (`gc.collect()` + `torch.cuda.empty_cache()`).
- [static/live.html](../static/live.html), [static/live.js](../static/live.js): the button, labelled from the
  `running` flag in `/api/live/state`.
- [app/live.py](../app/live.py): `load_cameras()` reads the camera list (used by the engine and by Start).

**Measured:**

| State | GPU busy | GPU memory |
|---|---|---|
| Server just started | 0% | 0 MB |
| Live running | 44% | 429 MB |
| After Stop | 0% | 263 MB |

The 263 MB left after Stop is PyTorch's GPU set-up (its "CUDA context"), which stays until the server process
exits. It does no work.

**Side effects to know about:** Stop also stops a running Day replay (same engine). The bathroom and "not seen"
checks and the automatic daily report only run while live is on.

---

## 5. Why the person on the floor wasn't flagged

**Method:** don't guess. Get the real footage and look at the numbers. The DVR keeps recordings and serves them
over RTSP, so we downloaded the 4 minutes around the event:
```powershell
ffmpeg -rtsp_transport tcp -i "rtsp://USER:PASS@192.168.1.5:554/Streaming/tracks/502?starttime=20261019T212900Z&endtime=20261019T213300Z" -an -c copy -t 240 videos/home/living_fall_2130.mp4
```
(`tracks/502` = channel 5 sub-stream, playback. `-an` drops the audio, because MP4 can't hold the DVR's audio
format. The time is the DVR's own clock.)

Then we printed, once a second, every person's box, posture and the two numbers the posture rule uses (script in
section 8):

| Clip time | What happened | Posture | Torso (deg from vertical) | Box width/height |
|---|---|---|---|---|
| 93–99 s | sits down on the floor | sitting / standing | 4–21 | 0.7–1.1 |
| 100–116 s | **lying on the floor** | **bending** | **30–54** | **1.23–1.38** |
| (whole clip) | people standing or at the table | standing | under 26 | 0.9–1.2 |

**The posture rule** ([app/posture.py](../app/posture.py) `classify`) says "lying" if

- the torso is at least 60° from vertical (`lying_torso`), or
- the torso is at least 45° (`lying_torso_wide`) **and** the box is at least 1.4 times wider than tall
  (`lying_aspect`).

Those numbers were tuned on datasets filmed from the side. **From the ceiling, a body on the floor looks like a
diagonal**, not a flat line: 30–54° and 1.2–1.4 times wider than tall. So it fell just short and came out as
"bending".

**Change:** this camera's own thresholds, in [videos/home/living.yaml](../videos/home/living.yaml) (the global
`rules.yaml` is unchanged, so the other cameras and the dataset results aren't affected):
```yaml
posture: {lying_torso_wide: 40, lying_aspect: 1.2, legs_hidden: sitting}
```
- `40` is below most of the floor frames (30–54°) and well above standing or sitting people (under 26°).
- `1.2` is just below the floor boxes (1.23–1.38).
- With this, about 8 of 10 floor frames count as lying. The posture is smoothed over neighbouring frames, so the
  track reads as lying.
- `legs_hidden: sitting`: at the dining table the legs are hidden under it. Without this, those people were
  guessed as "standing".

**Even then, this test wouldn't have alerted, and that is correct.** The rules ([app/events.py](../app/events.py)
`detect_falls`, `rules.yaml → fall:` and `live:`) separate:

| What happens | What the rules need | Result |
|---|---|---|
| Fall | standing/bending → lying within **2 s**, a real drop, down **≥ 5 s** | "possible fall", then **FALL ALERT** if still down **30 s** later |
| On the floor | lying outside the sofa/bed for **60 s** | "on the floor" alert |
| Lying down on purpose | sitting → lying (slowly) | no alert |

The test went standing → *sitting on the floor* → lying, and lasted only ~16 s. So it was neither a fall (it went
through sitting) nor long enough to count as on the floor (60 s).

**Lesson:** when a detector "misses", check each stage separately. (1) Did the model find the person? Yes. (2) Was
the posture right? No: a threshold problem for this camera angle. (3) Did the event rule apply? No, because the test
didn't match what it looks for. Only the second was a real bug.

---

## 6. Zones

A zone is a polygon in **fractions of the frame**: `[x, y]`, with `[0, 0]` the top-left and `[1, 1]` the
bottom-right. That way it doesn't depend on resolution. Which point of a person decides the zone: their **feet**,
or the **centre of the body** when they're lying ([app/tracks.py](../app/tracks.py)).

| Kind | Meaning | Used for |
|---|---|---|
| `room` | which room they're in | (the camera's `room:` already covers it) |
| `seat` | being low here is sitting, not fallen | dining chairs, armchairs |
| `rest` | lying here is normal (sofa, bed); long stays are logged | sofa |
| `ignore` | **new**: detections centred here are dropped | the chair stack by the door, the cloth on the stairs |

**How the coordinates were chosen:**
- **Seats and sofa:** drawn over the furniture on a snapshot, then checked by drawing them back onto the picture.
  The point where the person actually lay (0.67, 0.52) had to stay **outside** every zone. Otherwise a real fall
  there would be excused as "sitting". The dining-chair zone has a notch cut out for exactly this reason.
- **Ignore, living room:** the false "Person #?" box was always `x 0.00–0.13, y 0.63–1.00` (seen in every second of
  the recording, confidence 0.16–0.28). The zone covers `x 0–0.14, y 0.6–1.0`.
- **Ignore, stairs:** the false person was always a big box `x 0–0.47, y 0.03–1.0` with its centre at about
  (0.23, 0.55), checked over 10 live frames. The zone is only a small square around that centre, so a real person
  on the stairs (x 0.45–0.75) is never dropped. Moving the cloth is still the better fix.

**Code for `ignore`** ([app/perception.py](../app/perception.py) `_detect`): after the model runs, any detection
whose box centre falls inside an `ignore` polygon is skipped, using the same `in_poly` test the other zones use
([app/events.py](../app/events.py)).

---

## 7. Next tests to run

On the open floor between the armchairs and the dining table (on a mattress), with live started:
1. **Fall:** standing → lying fairly fast, stay still 40 s. Expect "possible fall", then a FALL ALERT.
2. **On the floor:** lie down slowly, stay 70 s. Expect an "on the floor" alert.

Note the DVR clock time of each, download the recordings as in section 5, and compare frame by frame. If a
test fails, check the three stages in section 5's lesson, in order.

---

## 8. Do it yourself

**Per-second postures for a clip** (how the table in section 5 was made):
```python
import sys, cv2
from ultralytics import YOLO
from app.posture import classify, features
m = YOLO('models/yolo11m-pose.pt')
cap = cv2.VideoCapture(sys.argv[1]); fps = cap.get(cv2.CAP_PROP_FPS); i = 0
while True:
    ok, f = cap.read()
    if not ok: break
    if i % int(fps) == 0:                       # one frame per second
        r = m.predict(f, imgsz=640, device=0, conf=0.15, verbose=False)[0]
        for b, kp in zip(r.boxes.xyxy.tolist(), r.keypoints.data.tolist()):
            ft = features(b, kp)
            print(f"{i/fps:5.0f}s", classify(b, kp), "torso", ft["torso"] and round(ft["torso"]), "aspect", round(ft["aspect"], 2))
    i += 1
```
Save it as e.g. `postures.py` and run `.venv\Scripts\python.exe postures.py videos\home\living_fall_2130.mp4`.

**See exactly what changed in the code:**
```powershell
git diff app/live.py app/main.py app/perception.py static/live.js static/live.html tests/test_live.py
```
`cameras.yaml` and `videos/home/` are git-ignored (password, home footage), so read them directly.

**Files touched in this session:**

| File | Change |
|---|---|
| `cameras.yaml` (ignored) | 5 DVR channels, `sample_fps`, per-camera `rules` |
| `videos/home/living.yaml`, `stairs.yaml` (ignored) | zones and posture thresholds |
| `videos/home/living_fall_2130.mp4` (ignored) | DVR recording of the floor test |
| `app/live.py` | `LatestFrame`, `sample_fps` per camera, `load_cameras()` |
| `app/main.py` | no automatic start; `/api/live/start`, `/api/live/stop` |
| `app/perception.py` | `ignore` zones |
| `static/live.html`, `static/live.js` | Start/Stop live button |
| `tests/test_live.py` | test for `LatestFrame` |

## 9. Replaying an old DVR day (23 Sep 2026)

**Problem.** `scripts/dvr.py` only fetches whole hours and joins runs of clips. For 23 Sep, 08:00–22:00, the clips came from a scratch downloader instead, as `data/recordings/2026-09-23/<cam>/<HHMMSS>.ps` (raw DVR files). But `scripts/replay_day.py` reads `data/dvr/<day>/<cam>/<HHMMSS>.mp4`.

**Fix, without new project code:**
- A junction `data/dvr/2026-09-23` points to `data/recordings/2026-09-23`.
- A small helper converts each finished `.ps` into `<HHMMSS>.mp4` beside it (copying the video, not re-encoding it).
  - It writes to `.tmp.mp4` first and then renames, so the replay never reads a half-written file.
  - When the download is complete it writes `.done`, which tells the replay to stop waiting.
- Then run `python scripts/replay_day.py 2026-09-23 --follow`. It starts on the clips already converted and keeps going while the download continues.

**Evidence.** After 2.5 minutes the replay had taken 22 clips across 4 cameras. It skips clips with no person, the same way the 18 Oct run did.

**Lesson.** When you join two tools, use the format the next tool already reads, such as a link plus a conversion step. Don't add a second input format to the tool.

**Result.** Open `http://localhost:8000/day?date=2026-09-23` to see the daily report and timeline. The 18 Oct run took 284 min for about 1,800 clips; 23 Sep has 3,032.

## 10. Watching the real clips from the day report

**Problem.** The `/day` report said things like "Porch 25 min" or "Possible fall 14:05", but there was no way to see the footage behind them. You had to find the right file on disk by hand.

**Fix.**
- **`/api/day/{day}` returns the clip list.** It now also sends `clips`: each camera's `<HHMMSS>` clip names from `data/dvr/<day>/<cam>/`.
- **New route `GET /media/dvr/{day}/{cam}/{HHMMSS}.mp4`.** It is LAN-only, like the rest of `/day`, because this is footage of the home. The names are checked against strict patterns, so a crafted path can't escape the folder.
  - It uses Starlette's `FileResponse`, which answers range requests. That is what lets the player seek.
- **The page finds the clip for each event.** Events record a camera and a time but not a file. Clips are named after their start time, so the page takes the last clip that started at or before the event, then starts playback 3 s before the event.
- **What's clickable.** Alerts, timeline bars and "Path through the day" entries open the clip at that moment.
  - A "Videos with someone in them" list shows, per camera, every clip that produced an event.
  - The player has Earlier/Later buttons and a Full-screen button; double-clicking the video also goes full screen.

**Limit.** The DVR records HEVC (H.265). Edge plays it. Chrome plays it only when hardware decoding is on, which this GPU supports. A browser that can't play it shows a note saying so.

**Check.** `tests/test_api.py::test_day_dvr_clips` covers the clip list, a range request (206), a missing clip (404), a path-escape attempt (404), and a request from off the LAN (404, like every LAN-only route).

## 11. The vision model reviews the footage (no time to retrain)

**Problem.** I checked the 23 Sep report against the real frames, and every fall it raised was false:

| Time | Camera | Said | Frame shows |
|---|---|---|---|
| 08:42 | porch | Possible fall, got up | A woman bending over the scooter, read as "lying" |
| 11:25–11:26 | sidepath | Possible fall, then **FALL — not getting up** | Sweeping, bent over. She then walks out at the bottom edge, and the alert fired on an **empty frame** |
| 11:27 | garden | Possible fall, got up | Sweeping with a short broom, bent double |

- **18 Oct:** 6 FALL alerts and 11 "fell, got up". All would have gone to the phone, and almost certainly nobody fell.
- **Why:** the pose model calls a bent-over person "lying". The rules then hold that state after the person walks out of view.
- **The report's posture minutes are wrong too,** because they come from the same pose model.
- **Constraint:** no time to retrain before the demo.

**Choice of how much the AI sees** (Gemini 2.5 Flash via OpenRouter, about 258 tokens per 768 px frame):

| AI looks at | Cost a day (5 cameras) |
|---|---|
| Only possible falls | about $0.01–0.03 |
| **Every stretch with a person, 1 frame every 2 s** (chosen) | about $0.4–1.5 |
| Everything, 1 frame/s | about $5–7 |

**Fix.** The fix runs in the camera loop, so the same code serves live and replay:
- **Segment reviews.** While someone is in view at confidence ≥ 0.5, the camera keeps a raw 768 px frame every 2 s.
  - Each 30 s of this (at most 15 frames), or the stretch up to when the person leaves, is one request: `LLM.review`.
  - The request runs on a 4-thread pool, so cameras never wait.
  - The JSON answer (people, posture frame counts, activity, fall, why) is logged as an `ai_review` event.
- **Fall check before any fall push (`_ai_check`).**
  - Frames about 1 s before the fall, 2 s after, and at the moment of the push, side by side, go to `LLM.fall_check`.
  - **Only a clear NO stops the push**, and it becomes "Not a fall (AI check)". UNSURE, an error, no key or the cap reached all mean the alert goes out, so an AI outage can't hide a fall.
  - The "on the floor, not moving" alert goes through the same check.
- **AI escalation.** If a review says FALLEN and the local model has no fall there, it raises "Possible fall (AI check)". The pose model often doesn't see someone lying on the ground at all.
- **The report.** When the day has reviews, active, sitting and lying minutes come from the AI and are marked "(AI-checked)".
  - New lines: "Seen doing (AI): …" and "AI check: N possible falls looked at…".
  - Time in each room still comes from the local presence events. Detecting a person works; the posture is the weak part.
- **`.env` now loads on import** (`app/__init__.py`, never overriding). The server had been started without `--env-file`, so it had no AI and no Telegram, and `replay_day.py` never read `.env` at all. Under pytest it is skipped, so tests never see real keys.

**Evidence so far.**
- First real review, side-path clip at 11:25, 8 frames: `bending 4, walking 4, activity "bending down to pick something", fall NO`. It took 3.3 s and cost $0.00093.
- **Model quirks seen:** the first answer listed frame numbers instead of counts and put some frames under two postures. The prompt now asks for whole numbers with each frame counted once, and the parser accepts lists anyway.
- **Tests:**
  - **The fall gate:** only NO stops a push, and that covers "got up" too.
  - **Segments:** 15 frames per 30 s, and the rest is sent when the person leaves.
  - **Escalation:** a review saying FALLEN raises one alert, and none if the local model already has that fall.
  - **Parsing:** the answers parse.
  - **The report:** AI postures replace the pose model's.

**Lesson.** Check the report against the frames before trusting a metric. The numbers looked plausible, but every fall was a broom.

**Measured on labelled falls** (`eval_live.py … --vlm`, cached YOLO, $0.01 for the whole run). The rule was set before the run: keep the veto only if it keeps ≥ 95 % of real FALL ALERTs.

| | Real FALL ALERTs | AI kept (not "NO") |
|---|---|---|
| Le2i | 78 | 68 (87 %) |
| UR Fall | 24 | 24 (100 %) |
| CAUCAFall | 35 | 35 (100 %) |
| **All** | **137** | **127 (93 %)**, fails the rule |

- **False pushes:** over 3.3 h of raw clips there were 7 (2.1/h), all "fell, got up". The AI stopped **7/7**.

**Decision, from the rule.**
- **A FALL ALERT (stays down) is never cancelled.** The AI only adds "AI check: not a fall (…)" to it, and the same goes for the "on the floor, not moving" alert.
- **"Fell, got up" can still be cancelled on a clear NO.** That is where every false push came from, and the person is already up.
- **Effect on 23 Sep:** the sidepath empty-frame FALL ALERT would still be sent, labelled by the AI.
- **Upgrade path:** look at the 10 Le2i strips the AI wrongly called NO. If a better prompt or more frames fixes them, re-measure on UR Fall and CAUCAFall as fresh data before letting it cancel stay-down alerts.

**Pilot (23 Sep 08:00–about 10:00, stopped at 58 min when the session closed).**
- **Volume and cost:** 119 AI reviews for **$0.049**.
- **Answers matched the footage:** for example "Sitting at dining table", "Walking around the porch", "Bending over a scooter". All 119 said fall: NO.

**One wrong call, from drawn labels.** The 08:42 porch fall check said "looks like a fall: person is on the ground in the second frame", while the review of the same moment said "Bending over a scooter".
- **Cause:** the fall check was sent the *annotated* frames, with the stick figure labelled "lying". The review and `eval_live --vlm` both use raw frames.
- **Fix:** the fall check's ring (`shots`) now keeps raw 768 px frames every 0.5 s, the same as the reviews.
- **Lesson:** don't show the second opinion the first opinion's labels.

## 12. Day page redesign: one place to review a day

**Problem.** The page had the right parts, but they were spread out:
- the plain-text report came first and the alerts were at the bottom;
- the video player sat far from a wall of clip buttons;
- clips where the AI saw nobody were mixed in with the rest;
- the 7-day table was all dashes when there was no history yet.

**Fix** (`static/day.html`, `day.js`, `style.css`; no server change):
- **Header:** the date as a title, with ◀ / date picker / ▶ to move between days. The upload form folds into "Add recordings" whenever a day is open.
- **Tiles:** first up, trips outside, active time (AI-checked), falls and AI reviews. The Falls tile filters the list to falls.
- **Player and Moments side by side,** both sticky on wide screens.
  - **Moments** has one row per clip that had events: time, coloured room, what the AI saw, and fall badges with the AI's reason.
  - **Filter chips:** All, Falls & alerts, or one camera.
  - Clips where the AI saw nobody are hidden unless you ask for them.
- **Keys:** N/P (or J/K) for the next or previous moment, F for full screen.
- **Timeline:** sits under the player, with a line marking what's playing. Clicking a bar or a mark plays it.
- **Below:** the family report text, and the 7-day comparison only once earlier days exist.

**Check.** Screenshot in headless Edge of `/day?date=2026-09-23`. Headless Edge can't decode HEVC, so the video there is black.
