# Staged-fall test: how to record, label and score your own falls

This is the next step after [GUIDE.md](GUIDE.md) §10. You record, I score and tune, and we both read the numbers.
Each section says **what** to do and **why**, so you can run this kind of test yourself on the next project.

---

## 1. Why this step exists (what the dataset could not tell us)

The benchmark (MCFD) gave us a number, but it answers the wrong question for a home:

| MCFD | Your home |
|---|---|
| Ceiling fisheye cameras, a worst case | Phones on a shelf, the angle we'd actually install |
| The actor almost always gets up within seconds | The case we care about: the person **stays down**, so the **FALL ALERT** fires |
| Furniture moves between scenarios, so no rest zones | Fixed sofa and bed, so rest zones can be drawn and tested |
| Someone else's look-alikes | **Your** look-alikes: your sofa, your floor, your habits |

So this test measures the three numbers that decide whether the demo is convincing:
1. **FALL ALERT recall:** of the falls where the person stayed down, how many reached the phone.
2. **False FALL ALERTs:** how often the family would be woken for nothing.
3. **Latency:** fall → alert on the phone. Expected ≈ 30 s (the deliberate wait for recovery) + a few seconds.

General lesson: **a benchmark tells you if the method can work; a test in the real setting tells you if the
product works.** You need both, in that order.

---

## 2. Safety and consent (read first)

- A **healthy adult acts** the falls, never an elderly person. Practise as a *controlled* fall: drop to the knees,
  then roll onto the side or back. It should *look* like a fall from the camera; it must not *be* one.
- Fall onto a **mattress or thick duvets**. Move tables, chairs with hard edges and glass out of the fall area.
- Have a second person in the room.
- Everyone filmed agrees to it. Footage stays on the laptop: put it in `data/staged/` (git-ignored, never committed).

---

## 3. Camera setup

**Phones:** use the normal camera app (recording, not streaming). Settings:
- landscape, **720p or 1080p, 30 fps**;
- no zoom, no slow-motion, no "auto-framing / subject tracking";
- stabilisation off if you can.

*Why:* the system sees ~6 frames per second at 640 px, so more resolution doesn't help. Auto-framing moves the
image, and that breaks rest zones.

**Placement, the way a real install would be:**
- **High:** top of a cupboard or shelf, ~1.8–2.2 m, in a corner, tilted down. *Why:* from high up, furniture hides
  the floor less and the whole room fits in view. Low cameras get blocked by the sofa.
- The **fall area (mattress) fully inside the frame, not at the edge.** *Why:* the "vanished" rule (Path C)
  deliberately ignores people who disappear at the frame edge, because that is just walking out.
- **2 phones at different angles, recording at the same time** (e.g. opposite corners). *Why:* one phone misses
  a fall that faces away or is hidden. The question "caught by **any** camera" is how a real multi-camera home
  behaves (the dataset's per-scenario recall, §5 of the guide).
- Optional 3rd: the laptop webcam at table height, as a deliberately bad angle to learn from.
- **Don't move the phones during a session** (tape or tripod). *Why:* zones are drawn on the exact view.
- Plug in the chargers and turn on airplane mode so a call doesn't stop the recording.

**Lighting:** do one round in daylight and, if possible, a short round in evening lamp light. Night falls are
real falls, and the pose model gets worse in the dark.

---

## 4. What to record (session 1, about 45 minutes)

Start each phone recording and leave it running for the whole session. One long video per phone is easiest.
Before each action, **say the take number and what you'll do out loud** ("take 4, fall toward the camera").
*Why:* the audio is then your labelling guide, and it tells you which moment is the same take on both phones.

Before each fall, stand visible for ~5 s. *Why:* the rules compare the body against its **upright height from
the seconds before** (Path A's height ratio). With no upright "before", there is nothing to compare against.

**A. Falls, then stay down still for 45 s (`fall_stay`).** These must produce a FALL ALERT.

| Take | Action | What it tests |
|---|---|---|
| 1 | Forward trip onto the mattress | the classic fall (Path A, lying) |
| 2 | Backward: sit-roll onto the back | hips drop, body ends wide |
| 3 | Sideways collapse (knees buckle, slump) | slow-ish fall, small height change |
| 4 | Fall **toward** a camera | foreshortening: the torso looks upright, so Path B (floor drop) must catch it |
| 5 | Fall partly **behind furniture** | the person disappears, so Path C (vanished) |
| 6 | Slide off a chair to the floor | starts from sitting, which is the hardest start |
| 7 | Very slow collapse down the side of the sofa | slow falls are the classic miss |

**B. Falls, then get up within 5–15 s (`fall`).** These must **not** alert: "recovered" only.

| Take | Action |
|---|---|
| 8 | Forward trip, get up after ~5 s |
| 9 | Sideways collapse, get up after ~15 s |

**C. Look-alikes (`lookalike`).** These must stay quiet. They are the false alarms we found on MCFD, plus
home ones.

| Take | Action |
|---|---|
| 10 | Drop hard onto the sofa (the MCFD false-alarm #1) |
| 11 | Lie down on the sofa for 2 min |
| 12 | Lie down on the bed (if a bed is in view) |
| 13 | Bend down to pick something up off the floor |
| 14 | Kneel to tie shoes / look under the bed or sofa |
| 15 | Sit on the floor on purpose and stretch for 1 min. **Expect this one to alarm.** It *is* hard. |
| 16 | Walk out of the room (out of view) |

**D. Normal life (`none`).** 10 minutes of ordinary moving around: walking, sitting, standing, a second person
walking through. *Why:* this is what most of a real day looks like, so false alarms **per hour** come mostly from here.

If you have energy, repeat takes 1–7 a second time with variations (other side, other clothes).

> **Statistics lesson.** With 14 stay-down falls, every fall is 7 % of the result. 10/14 = 71 %, but the honest
> 95 % range is roughly 45–88 %. Small tests give rough numbers, so more falls give tighter numbers. That's why
> repeats matter, and why one good demo is not proof.

---

## 5. Session 2 = the test set (another day)

After session 1, I tune: rest zones and, if needed, thresholds. Then you record **session 2** on a different day:
- the same shot list in a new order;
- phones moved a little, other clothes, maybe another room.

Session 2 is scored **once**, and that's the number we report.

*Why:* it is the same rule as the chute01–16 / 17–24 split. If I tune on the same falls I score on, the number
measures my memory, not the system. On MCFD that gap was 3.2/h on the tune set vs 19.5/h on the test set for
"possible fall" false alarms. Expect a gap here too, and believe the session-2 number.

---

## 6. Labelling (the "ground truth")

Labels are the answer key. The score is only as true as the labels, so this is the part to do carefully.

1. Open each video in **VLC**. The time is shown bottom-left. `E` steps one frame; `Shift+←/→` jumps 3 s.
2. For each take write one row in `labels.csv` (start from
   [staged_labels_template.csv](staged_labels_template.csv); Excel → *Save as CSV*):

| column | what to put |
|---|---|
| `video` | file name, e.g. `phoneA_s1.mp4` |
| `camera` | which phone: `phoneA`, `phoneB` (same name for all its videos) |
| `take` | the number you said out loud. **Same number on both phones for the same fall.** |
| `kind` | `fall_stay`, `fall`, `lookalike`, or `none` (a video with nothing at all) |
| `start_s` | seconds into **this** video when balance is lost (falls) or the action starts (look-alikes) |
| `end_s` | when the body has landed (falls) or the action ends (look-alikes) |
| `note` | free text: "toward phoneA", "behind sofa" |

- Times are **per video**: the phones started at different moments, so take 4 is at a different second on each phone.
- ±1 s is accurate enough. *Why:* the scorer accepts a detection from 2 s before the labelled start up to 4 s after
  the labelled end (`WINDOW_PAD`), because "when did the fall start?" has no exact answer.
- Put everything in `data/staged/session1/`: videos + `labels.csv`.

---

## 7. Rest zones (sofa / bed) per phone

A fall that starts inside a `kind: rest` zone is ignored, so lying on the sofa is never a fall. On MCFD we couldn't
use zones. Here we can, and it should remove most look-alike alarms.

```
.venv\Scripts\python scripts/zone_helper.py data/staged/session1/phoneA_s1.mp4
```
This saves the first frame with a 10 % grid. Read the sofa's corners as fractions of width/height, then create
`data/staged/session1/phoneA.yaml` (the file name = the `camera` name in labels.csv):
```yaml
zones:
  - {name: Sofa, kind: rest, polygon: [[0.55, 0.45], [0.95, 0.45], [0.95, 0.85], [0.55, 0.85]]}
  - {name: Living room, kind: room, polygon: [[0, 0], [1, 0], [1, 1], [0, 1]]}
```
The same file later becomes that camera's `rules:` in `cameras.yaml` for the live demo: same format, same logic.
I'll draw these for session 1 if you prefer. Try one yourself first.

---

## 8. Scoring

```
.venv\Scripts\python scripts/eval_staged.py data/staged/session1 --list
```
The first run runs YOLO on every video (≈ real time ÷ 3 on the GTX 1650) and caches it in `data/staged_cache/`.
After that, re-scoring takes seconds, so changing a rule and checking it is quick.

| Output line | Meaning | Goal for the demo |
|---|---|---|
| `falls seen (possible fall)` | falls the rules noticed at all | as high as possible |
| `per take, any camera` | falls caught by at least one phone | the real-home number |
| `FALL ALERT on stay-down falls` | the ones that would reach the phone, + median latency | ≥ 80 %, ~31–40 s |
| `wrong ALERT on got-up falls` | alerted although the person got up | 0 |
| `false possible falls` | logged only, never pushed | low, not critical |
| `false FALL ALERTs` | pushed to the family for nothing | **0** on look-alikes |
| `false alarms during:` | which look-alike caused each one | tells us what to fix |

Lines per video say **MISSED**, **NO ALERT** (seen, but the system thought they got up), or **FALSE …** with
the rule (`lying`, `floor_drop`, `vanished`) that fired. That's the starting point of error analysis: watch the
video at that second and ask *why*. That loop, measure → look at failures → change one thing → measure again,
is the whole method from GUIDE.md §7.

The scoring logic is `score()` in [scripts/eval_staged.py](../scripts/eval_staged.py), and
[tests/test_staged.py](../tests/test_staged.py) checks it on a hand-made example. Read the test first; it's the
quickest way to see what counts as a hit, a false alarm and an alert.

---

## 9. The live demo (showing the idea)

The same recording can drive the **real** alert path:
1. Start the app (`uvicorn app.main:app --host 0.0.0.0 --port 8000`) and open `http://localhost:8000/live`.
2. **Play as live camera** → choose `phoneA_s1.mp4`.

It plays in real time. A stay-down fall shows "possible fall", then about 30 s later the **FALL ALERT** arrives on
the HA app and Telegram with the snapshot. That's the "Guardian at home" moment to show people. Then do the same
live: an IP Webcam phone in `cameras.yaml`, and someone lies on the mattress for 40 s.

---

## 10. What to send me

- The path of the folder: `data/staged/session1/` with the videos.
- `labels.csv` if you labelled it. If not, a list like "phoneA: take 4 at 3:12, toward the camera, stayed
  down" is enough, and I'll make the CSV. Labelling yourself is the better lesson, though.
- A photo of the room showing where each phone was.

**What I'll do with it:** score it, watch every miss and false alarm, draw or fix rest zones, change rules only if
the session-1 numbers say so, and write each step into GUIDE.md §7 (the experiment log). Then session 2.

**Later, the same videos become training data:** frames where you're lying on the floor, from *your* camera
angles, are exactly what's needed to fine-tune the pose model (GUIDE.md §10, step 3). Its biggest weakness is
not seeing people lying on the floor. Record well now and you'll have that dataset for free.
