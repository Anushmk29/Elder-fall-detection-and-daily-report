"""Train a small neural network to spot falls from pose keypoints, and compare it with the rules on held-out data.

Usage: python scripts/train_fall_model.py "C:/.../archive (4)" D:/datasets/le2i [--extra D:/datasets/urfall/Videos ...] [--epochs 40]
Writes models/fall_gru.pt and prints validation + test results. Explained step by step in learning/TRAINING.md.

Data: the cached YOLO output of MCFD (192 clips) and Le2i (130 clips), tracked with tracks.build exactly like the
rules see it. Splits (by scenario/room, never by frame, so the same fall can't be in two splits):
  train = MCFD chute01-12 + Le2i Coffee_room_01        -> the weights are fitted here
          (+ every --extra labels.csv folder, e.g. UR Fall, CAUCAFall; val/test stay the same)
  val   = MCFD chute13-16 + Le2i Coffee_room_02        -> stop training / pick the threshold here
  test  = MCFD chute17-24 + Le2i Home_01/02            -> reported once, never used for any choice
One sample = one person's last 3 s (18 steps at 6 fps). Label 1 if a labelled fall started 1-3 s before the end
of the window (so at least 1 s after the fall is visible).
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "scripts"))
import eval_mcfd as ev  # noqa: E402
import eval_staged as es  # noqa: E402
import tune_collapse as tc  # noqa: E402
from app import pipeline, tracks  # noqa: E402

FPS = 6
WIN = 18            # 3 s window
TAIL = 4 * FPS      # keep 4 s after a track's last detection (present=0): "disappeared" is a fall signal too
N_FEAT = 17 * 3 + 5
VAL_MCFD = range(13, 17)


# ---------- 1. features: one vector per person per time step ----------
def track_series(tr, duration):
    """(T, N_FEAT) array on a fixed 6 fps grid + the grid times. Everything is expressed relative to the person's
    own size and position over the previous 3 s, so the model learns "how the body moved", not "where in this
    camera's image it was" (that is what lets it work on a camera it has never seen)."""
    S = tr["samples"]
    t0, t1 = S[0]["t"], min(S[-1]["t"] + TAIL / FPS, duration)
    grid = np.arange(round(t0 * FPS), round(t1 * FPS) + 1) / FPS
    by_step = {round(s["t"] * FPS): s for s in S}
    X = np.zeros((len(grid), N_FEAT), np.float32)
    hist = []  # recent (height, centre x, bottom) of seen frames
    for i, t in enumerate(grid):
        s = by_step.get(round(t * FPS))
        hist = [h for h in hist if h[0] >= t - 3]
        if s is None:
            continue  # all zeros incl. present flag = 0
        x1, y1, x2, y2 = s["box"]
        hist.append((t, y2 - y1, (x1 + x2) / 2, y2))
        H = max(h[1] for h in hist) or 1.0              # tallest recent height ~ upright height
        cx = float(np.median([h[2] for h in hist]))
        bot = float(np.median([h[3] for h in hist]))
        kp = np.array(s["kp"], np.float32)
        X[i, 0:51:3] = (kp[:, 0] - cx) / H
        X[i, 1:51:3] = (kp[:, 1] - bot) / H
        X[i, 2:51:3] = kp[:, 2]
        X[i, 51:] = [(x2 - x1) / H, (y2 - y1) / H, (y1 - (bot - H)) / H, (y2 - bot) / H, 1.0]
    return X, grid


def build(clips, rules):
    """[(X, grid, fall_starts, clip_index)] for every track of every clip."""
    out = []
    for c, (frames, meta, rows, _) in enumerate(clips):
        starts = [r["start_s"] for r in rows if r["kind"] in es.FALLS]
        for tr in tracks.build(frames, meta, rules):
            if len(tr["samples"]) >= 6:
                X, grid = track_series(tr, meta["duration"])
                if len(grid) >= WIN:
                    out.append((X, grid, starts, c, tr["samples"]))
    return out


def windows(series, stride):
    """Index list of (series, end step, label). Every positive window is kept, negatives every `stride` steps."""
    idx = []
    for k, (X, grid, starts, *_) in enumerate(series):
        for e in range(WIN - 1, len(grid)):
            T = grid[e]
            y = int(any(T - 3.0 <= s <= T - 1.0 for s in starts))
            if y or e % stride == 0:
                idx.append((k, e, y))
    return idx


def batch(series, idx):
    x = np.stack([series[k][0][e - WIN + 1:e + 1] for k, e, _ in idx])
    return torch.from_numpy(x), torch.tensor([y for *_, y in idx], dtype=torch.float32)


# COCO keypoints: 0 nose, then left/right pairs (eyes, ears, shoulders, elbows, wrists, hips, knees, ankles)
SWAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def mirror(x):
    """Left-right mirror of (..., N_FEAT) features: negate keypoint x (already centred on the person), swap L/R."""
    kp = x[..., :51].reshape(*x.shape[:-1], 17, 3)[..., SWAP, :].clone()
    kp[..., 0] *= -1
    return torch.cat([kp.reshape(*x.shape[:-1], 51), x[..., 51:]], -1)


# ---------- 2. the model ----------
class FallGRU(nn.Module):
    """A GRU reads the 18 steps in order and keeps a running memory; the last memory -> one fall score."""
    def __init__(self, hidden=64):
        super().__init__()
        self.gru = nn.GRU(N_FEAT, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, 1))

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(-1)  # a logit; sigmoid(logit) = probability of a fall


# ---------- 3. from window scores to fall events, scored exactly like the rules ----------
@torch.no_grad()
def predict(model, series, dev):
    model.eval()
    out = []
    for X, grid, _, c, S in series:
        x = torch.from_numpy(np.lib.stride_tricks.sliding_window_view(X, WIN, axis=0).transpose(0, 2, 1).copy())
        p = torch.sigmoid(model(x.to(dev))).cpu().numpy()
        out.append((grid[WIN - 1:], p, c, S))
    return out


def got_up(S, t, secs):
    """The same stay-down idea as the rules: standing as tall as before the fall within `secs` = not a fall."""
    before = [s["box"][3] - s["box"][1] for s in S if t - 3 <= s["t"] <= t - 0.5]
    if not before:
        return False
    H = max(before)
    return any(s["posture"] == "standing" and s["box"][3] - s["box"][1] >= 0.85 * H
               for s in S if t + 1 <= s["t"] <= t + secs)


def events(preds, thr, refractory=10.0, confirm_s=0):
    """{clip: [fall events]}: a fall when p >= thr (and, with confirm_s, not back up within confirm_s), then quiet
    for `refractory` s on that track."""
    ev_by_clip = {}
    for times, p, c, S in preds:
        last = -1e9
        for T, q in zip(times, p):
            if q >= thr and T - last >= refractory and not (confirm_s and got_up(S, T - 1.0, confirm_s)):
                ev_by_clip.setdefault(c, []).append({"t": T - 1.0, "got_up_s": 1e-3, "how": "model"})
                last = T
    return ev_by_clip


def evaluate(clips, found):
    hits = falls = fa = 0
    secs = 0.0
    for c, (frames, meta, rows, _) in enumerate(clips):
        rows = [r for r in rows if r["kind"] != "none"]
        r = es.score(sorted(found.get(c, []), key=lambda e: e["t"]), rows, meta["duration"], 30)
        falls += len(rows)
        hits += len(r["hits"])
        fa += len(r["false"])
        secs += meta["duration"]
    return hits, falls, fa, fa / (secs / 3600)


def extra(folder, base):
    """labels.csv clips in tc.load's format, last frame held 10 s like Le2i (the clips stop right after the fall)."""
    out = []
    for _, frames, meta, rows, _ in tc.el.staged_clips(folder, base):
        es.hold(frames, meta, 10, base["sample_fps"])
        meta["clock_start"] = 0
        out.append((frames, meta, rows, []))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mcfd")
    ap.add_argument("le2i")
    ap.add_argument("--extra", nargs="*", default=[], help="labels.csv folders added to train only")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0, help="repeat with other seeds: one run can be luck")
    a = ap.parse_args()
    random.seed(a.seed), np.random.seed(a.seed), torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    base = pipeline.load_rules()
    base["zones"] = []
    data = tc.load(a.mcfd, a.le2i, base, zones_on=False)  # tune = MCFD 01-16 + Coffee, test = MCFD 17-24 + Home
    # split tune into train / val by scenario: MCFD clips carry no chute number here, so use the order tc.load made
    n_mcfd_tune = 16 * 8
    mcfd_tune, coffee = data["tune"][:n_mcfd_tune], data["tune"][n_mcfd_tune:]
    is_val_mcfd = [((k // 8) + 1) in VAL_MCFD for k in range(n_mcfd_tune)]
    coffee2 = [i for i, (_, meta, rows, _) in enumerate(coffee) if i >= 48]  # Coffee_room_02 = after room 01's 48
    split = {"train": [c for c, v in zip(mcfd_tune, is_val_mcfd) if not v] + [c for i, c in enumerate(coffee) if i not in coffee2],
             "val": [c for c, v in zip(mcfd_tune, is_val_mcfd) if v] + [coffee[i] for i in coffee2],
             "test": data["test"]}
    for f in a.extra:
        split["train"] += extra(Path(f), base)
    series = {k: build(v, base) for k, v in split.items()}
    tr_idx = windows(series["train"], stride=3)
    va_idx = windows(series["val"], stride=3)
    pos = sum(y for *_, y in tr_idx)
    print(f"train windows {len(tr_idx)} ({pos} fall), val {len(va_idx)} ({sum(y for *_, y in va_idx)} fall)")

    model = FallGRU().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(tr_idx) - pos) / max(pos, 1), device=dev))
    vx, vy = batch(series["val"], va_idx)
    best, best_state, patience = 1e9, None, 0
    for epoch in range(a.epochs):
        model.train()
        random.shuffle(tr_idx)
        total = 0.0
        for i in range(0, len(tr_idx), 256):
            x, y = batch(series["train"], tr_idx[i:i + 256])
            x = x + 0.02 * torch.randn_like(x)  # small noise = augmentation: keypoints are noisy in real life
            flip = torch.rand(len(x)) < 0.5     # mirror half the batch: a fall to the left is still a fall
            x[flip] = mirror(x[flip])
            loss = loss_fn(model(x.to(dev)), y.to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        model.eval()
        with torch.no_grad():
            vloss = sum(loss_fn(model(vx[i:i + 2048].to(dev)), vy[i:i + 2048].to(dev)).item() * len(vy[i:i + 2048])
                        for i in range(0, len(vy), 2048)) / len(vy)
        print(f"epoch {epoch + 1:2d}  train loss {total / len(tr_idx):.3f}  val loss {vloss:.3f}", flush=True)
        if vloss < best - 1e-3:
            best, best_state, patience = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        elif (patience := patience + 1) >= 6:
            print("val loss stopped improving -> early stop (more epochs would only memorise the train set)")
            break
    model.load_state_dict(best_state)

    # pick the threshold on VAL (with and without the stay-down check) with the same score the rules were tuned with
    vp = predict(model, series["val"], dev)
    tp = predict(model, series["test"], dev)
    min_down = base["fall"]["min_down_s"]
    chosen = {}
    for confirm in (0, min_down):
        scored = []
        for thr in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98):
            h, f, fa, fa_h = evaluate(split["val"], events(vp, thr, confirm_s=confirm))
            scored.append((h / f - 0.5 * fa_h / 10, thr))
            print(f"VAL stay-down {confirm:.0f}s thr {thr:.2f}: recall {h}/{f} = {h / f:.0%}, false alarms {fa} = {fa_h:.1f}/h")
        chosen[confirm] = max(scored)[1]
    torch.save({"state": model.state_dict(), "thr": chosen[min_down], "confirm_s": min_down, "win": WIN, "fps": FPS},
               BASE / "models" / "fall_gru.pt")

    print()
    rules = {c: ev.detect(frames, meta, base) for c, (frames, meta, _, _) in enumerate(split["test"])}
    runs = [("rules (rules.yaml, no zones)", rules)]
    for confirm, thr in chosen.items():
        runs.append((f"model thr {thr} stay-down {confirm:.0f}s", events(tp, thr, confirm_s=confirm)))
    m = events(tp, chosen[min_down], confirm_s=min_down)
    runs.append(("rules OR model+stay-down", {c: rules.get(c, []) + [e for e in m.get(c, []) if not any(
        abs(e["t"] - r["t"]) < 5 for r in rules.get(c, []))] for c in range(len(split["test"]))}))
    for name, found in runs:
        h, f, fa, fa_h = evaluate(split["test"], found)
        print(f"TEST {name}: recall {h}/{f} = {h / f:.0%}, false alarms {fa} = {fa_h:.1f}/h")


if __name__ == "__main__":
    main()
