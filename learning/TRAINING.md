# Training a fall model from scratch (and comparing it honestly with the rules)

Companion to [GUIDE.md](GUIDE.md). The code is one file, [scripts/train_fall_model.py](../scripts/train_fall_model.py),
written to be read top to bottom. Run it yourself:

```powershell
.venv\Scripts\python scripts\train_fall_model.py "C:\Users\ABC\Downloads\archive (4)" D:\datasets\le2i --extra D:\datasets\urfall\Videos D:\datasets\cauca\Videos
```
It takes ~5 minutes on the GTX 1650 (the YOLO output is already cached) and writes `models/fall_gru.pt`.
Results of the three runs: `learning/results/train_run*.txt`.

---

## 1. What exactly is being trained?

The **pose model (YOLO) is not retrained**. It stays a fixed "eye" that turns each frame into 17 keypoints and a
box. What we train is a **second, small network that watches those keypoints over time** and answers one
question: *"in the last 3 seconds, did this person fall?"*

That's the idea behind ST-GCN and the other "skeleton action recognition" repos in the original plan, done
small enough to understand completely. Why this split of work:
- Training a pose model needs **images labelled with keypoints**. Our datasets only say *when* a fall happens.
- Training a model on keypoint **sequences** needs exactly that: sequences + "fall / no fall" labels. We have it.

## 2. The data: what one training example looks like

**One example = one person's last 3 s = 18 time steps (6 per second) × 56 numbers.**

The 56 numbers per step (`track_series`):

| numbers | what | why this form |
|---|---|---|
| 17 × (x, y) | each keypoint, **minus the person's recent centre / feet position, divided by their recent height** | makes it independent of where in the image the person is and how far from the camera. A fall far away and a fall close up look the same |
| 17 × conf | how sure YOLO is about each keypoint | a body lying on the floor gives low confidence, which is itself a signal |
| 4 box values | width, height, top and bottom, relative to the recent height and feet | box shrink and "top drops while feet stay" are what the rules use too |
| 1 present flag | 1 if the person was detected at this step, else 0 | "dropped, then disappeared" (Path C) can be learned too |

*"Recent" = the previous 3 s.* That gives a moving reference: before a fall the height is the standing height,
so a fall shows up as numbers moving far from 0 and 1.

**This normalisation is the most important design decision in the file.** Without it the network would learn
"people at the bottom of the MCFD image are lying", which is useless on any other camera.

**Label:** 1 if a labelled fall *started* between 1 and 3 s before the end of the window, else 0. The window must
contain at least 1 s *after* the fall, so the model has seen the outcome, not just a stumble.

## 3. The splits: three sets, three jobs

| set | data | used for |
|---|---|---|
| **train** | MCFD chute01–12 + Le2i Coffee_room_01 | fitting the weights (the network sees these) |
| **validation** | MCFD chute13–16 + Le2i Coffee_room_02 | deciding *when to stop training* and *which threshold to use* |
| **test** | MCFD chute17–24 + Le2i Home_01/02 | reported **once**. No decision is ever made from it |

It's split by scenario/room, as always, never by frame or camera. Neighbouring frames are near-copies; if they
landed in train *and* test, the test would measure memory.

Why a **validation** set, when the rules only had tune/test? Because a network has a new choice the rules didn't:
*how long to train*. If we picked the epoch by looking at test, the test would stop being a test.

## 4. The model: a GRU

```
18 steps × 56 numbers ──► GRU (memory of 64 numbers, updated step by step) ──► last memory ──► Linear ──► 1 score
```
A **GRU** (gated recurrent unit) reads the sequence in order and keeps a running summary. At every step it
decides how much of its memory to keep and how much to overwrite with what it sees now. That fits the problem:
"upright… upright… dropping fast… low… still low" only means something in order.

Size: ~23,000 weights (YOLO11m has ~20 million). Small on purpose: with ~200 falls, a big model would just
memorise them.

`Dropout(0.3)` randomly switches off 30% of the memory during training, which forces the model not to rely on
one single clue.

## 5. Training: the loop, line by line

```python
loss_fn = nn.BCEWithLogitsLoss(pos_weight=negatives / positives)
for epoch in ...:
    for each batch of 256 windows:
        x = x + noise; mirror half of them         # augmentation
        loss = loss_fn(model(x), y)                # how wrong are we?
        opt.zero_grad(); loss.backward(); opt.step()   # nudge every weight to be a bit less wrong
    measure the loss on validation; keep the best weights; stop if no improvement for 6 epochs
```

- **Loss = binary cross-entropy.** It punishes being confidently wrong. **`pos_weight`** fixes the *class
  imbalance*: only ~24% of windows are falls (2,038 of 8,421; most negatives are sampled every 3rd step). Without
  the weight, "always say no fall" scores well and the model learns nothing.
- **Adam, learning rate 0.001**: the standard optimiser; `weight_decay` keeps the weights small (less memorising).
- **Augmentation**, which gives free extra data:
  - a little random noise, because real keypoints jitter;
  - **mirroring** left↔right (`mirror()`: negate x, swap left/right joints). A fall to the left is still a fall,
    so mirroring doubles the variety. In run 3 this cut the test false alarms in half.
- **Early stopping.** Read the curve (run 1):

```
epoch  1  train loss 0.896  val loss 0.676
epoch  8  train loss 0.439  val loss 0.498
epoch 16  train loss 0.337  val loss 0.470   <- best on validation
epoch 22  train loss 0.311  val loss 0.483   <- train keeps improving, validation got worse: memorising
```
  When **train loss keeps going down but validation loss goes up, the model is learning the training set, not
  falls.** That is *overfitting*, seen live. We keep the epoch-16 weights.

## 6. From scores to alarms, scored like the rules

The network outputs a probability every 1/6 s. To compare it fairly with the rules, it has to produce **events**:
- a fall event when the probability ≥ a **threshold** (`events()`), then 10 s of quiet on that person;
- optionally the **stay-down check** (`got_up()`): discard it if they stand as tall as before within 5 s. This is
  the same idea as `min_down_s` in the rules.

The **threshold** is chosen on *validation*, with the same score the rules were tuned with
(`recall − 0.5 × false alarms per hour / 10`). The events are scored by the same function as the rules
(`eval_staged.score`). Same metric, same matching, same test clips: that's what makes the comparison honest.

## 7. Results (test set, never used for any choice)

| run | what | falls caught | false alarms |
|---|---|---|---|
| rules | `rules.yaml` (after this session's fixes) | **44/101 = 44%** | 15.8/h |
| 1 | GRU, threshold 0.98 | 40/101 = 40% | 15.2/h |
| 2 | + stay-down check | 37/101 = 37% | 12.5/h |
| 3 | + mirroring augmentation | 35/101 = 35% | **7.9/h** |
| 3 | rules **OR** model | **49/101 = 49%** | 17.8/h |
| 4 | + **UR Fall in train** (`--extra`: 30 falls, 16 lying down on purpose, 24 other) | **45/101 = 45%** | 12.5/h |
| 4 | rules **OR** model | **52/101 = 51%** | 19.8/h |
| 5 | run 4, seeds 0/1/2 | 45 / 47 / 43% (mean **45%**) | 12.5 / 8.6 / 13.2 (mean 11.4/h) |
| 5 | + **CAUCAFall** in train, seeds 0/1/2 | 46 / 49 / 61% (mean **52%**) | 9.2 / 11.9 / 14.5 (mean 11.9/h) |

**What it means, in plain words:**
1. **A model trained from scratch on ~200 falls is about as good as rules built by hand, not better.** It sits at
   a different trade-off: fewer falls, but half the false alarms.
2. On validation the model reached **91% of falls** at a low threshold, but with far too many false alarms.
   It *finds* falls well; it can't yet tell them from look-alikes (sinking into a sofa, kneeling). That needs more
   look-alike examples, not a bigger network.
3. The stay-down check helped the rules a lot (§7 of the guide) but the model only a little. The model's
   false alarms are not quick get-ups; they look like falls *and* stay low.
4. Rules OR model catches the most falls (+5 over the rules alone for +3 false alarms) because they make
   *different* mistakes. Combining different methods ("ensembling") is a standard trick, and this is why.

5. **Run 4: one small extra dataset moved test recall by 10 points** (35 → 45%), more than every model change
   before it. Now the model beats the rules on both falls and false alarms. Val and test are unchanged, so it's a
   fair comparison, but it's one training run (one random seed): the next step is 3 seeds and CAUCAFall.
6. **Run 5: 3 seeds each.** The same data and code, different random starting weights and batch order, gives
   43–47% (UR) and 46–61% (UR + CAUCAFall). That spread is why one run proves little. Across all 6 runs the model
   had fewer false alarms than the rules every time and caught more falls in 5. `--seed N` reruns it.
   `models/fall_gru.pt` is the UR + CAUCAFall seed-0 model: the default seed, *not* the best test score, since
   picking by test would make the test a tuning set.

**Decision (after run 3):** not wired in: it didn't clearly beat the rules, and a component that doesn't earn
its place is just more to maintain. **After run 5** it does beat them, consistently. Wiring it into `live.py` as a
second fall signal is the next decision; it isn't done yet.

## 8. What would make the model win

In order of expected value:
1. **More falls, especially more look-alikes.** Done: UR Fall (run 4) and CAUCAFall (run 5), GUIDE §5.2–5.3.
   Next: UP-Fall (17 people, 11 activities; needs a sign-up). Each one is another
   `labels.csv` via a converter like `scripts/le2i_labels.py`. Keep splitting by scenario/person.
2. **Better eyes:** the pose model still misses people lying on the floor (GUIDE §7 finding 4). Fine-tuning YOLO
   pose on floor frames would help both rules *and* model. Le2i even ships a person box per frame, a start for
   box labels.
3. **A longer window** (e.g. 6 s), so the model itself sees "stayed down".
4. **Pre-training trick:** train on all the "normal" movement first (no labels needed), then fine-tune on falls.
   That's more advanced; only worth it after 1–3.

## 9. Exercises

1. In `train_fall_model.py` set `hidden=16` and then `hidden=256`. What happens to train loss vs val loss in each case,
   and why?
2. Remove `pos_weight` from the loss. What does the model predict for almost every window? Check the VAL recall.
3. Remove the division by `H` in `track_series` (keep raw pixels). Which split gets worse, val or test, and why
   does that tell you the normalisation was about *generalising to new cameras*?
4. Why would choosing the threshold on the test set make the test result worthless, even though it's "just one
   number"?
5. The model and the rules make different mistakes. Add a print in `main()` of the test clips where the model
   has a hit and the rules don't, then watch one of them. What did the rules miss?
