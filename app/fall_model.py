"""The learned fall signal (a small GRU over each person's last 3 s of keypoints), trained by
scripts/train_fall_model.py (learning/TRAINING.md). Live uses it beside the rules when rules.yaml live.fall_model is on."""
from pathlib import Path

import numpy as np
import torch
from torch import nn

BASE = Path(__file__).resolve().parent.parent
MODEL = BASE / "models" / "fall_gru.pt"
FPS = 6
WIN = 18            # 3 s window
TAIL = 4 * FPS      # keep 4 s after a track's last detection (present=0): "disappeared" is a fall signal too
N_FEAT = 17 * 3 + 5


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


class FallGRU(nn.Module):
    """A GRU reads the 18 steps in order and keeps a running memory; the last memory -> one fall score."""
    def __init__(self, hidden=64):
        super().__init__()
        self.gru = nn.GRU(N_FEAT, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, 1))

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(-1)  # a logit; sigmoid(logit) = probability of a fall



def got_up(S, t, secs):
    """The same stay-down idea as the rules: standing as tall as before the fall within `secs` = not a fall."""
    before = [s["box"][3] - s["box"][1] for s in S if t - 3 <= s["t"] <= t - 0.5]
    if not before:
        return False
    H = max(before)
    return any(s["posture"] == "standing" and s["box"][3] - s["box"][1] >= 0.85 * H
               for s in S if t + 1 <= s["t"] <= t + secs)


class Scorer:
    """p(fall started 1-3 s ago) for one track at time t, from the trained checkpoint (its own threshold)."""
    def __init__(self, path=MODEL):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.model = FallGRU()
        self.model.load_state_dict(ck["state"])
        self.model.eval()
        self.thr, self.confirm_s = ck["thr"], ck["confirm_s"]

    @torch.no_grad()
    def p(self, tr, t):
        X, grid = track_series(tr, t)
        if len(grid) < WIN:
            return 0.0
        return float(torch.sigmoid(self.model(torch.from_numpy(X[-WIN:][None]))))
