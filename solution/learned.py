"""Learned per-frame evidence (replaces the hand-weighted emissions), trained on EXTERNAL DFL matches.

A small multinomial logistic regression on features that exist in both DFL and the competition
tracking (no Smart Tagging). To make it robust to our tracker, DFL training data is DEGRADED first:
the ball is deleted in long blocks (~41% of frames, like ours) and given position noise. Speeds
are expressed relative to the match's own median, so tracker scale cancels. Training adds a
team-mirrored copy of every frame (H<->A swapped), so the model cannot learn a home bias.
The HMM decoder is unchanged; this only supplies its emission scores.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from solution.features import (clean_ball, team_dists, gap_length, width_spread_diff,  # noqa: E402
                               _mean_speed, SPEED_WINDOW, PITCH, NEAR_M, CONTROL_SCALE_M)

FPS = 25
# columns that flip sign when the two teams are swapped (home <-> away)
SIGNED = ["adv", "shape"]
COLS = ["adv", "near", "seen", "out", "loggap", "spd_rel", "spd_rel_1s", "spd_rel_20s", "shape",
        "bspeed", "still", "edge"]


def degrade(d, rng, frac=0.41, noise_m=0.3):
    """Copy of d with the ball deleted in long blocks (per half) and noised."""
    ball = d["ball"].copy()
    starts = list(d["half_start"]) + [len(ball)]
    for a, b in zip(starts[:-1], starts[1:]):
        n = b - a
        gone = np.zeros(n, bool)
        while gone.mean() < frac:
            ln = int(np.clip(rng.lognormal(np.log(30 * FPS), 0.8), 3 * FPS, 200 * FPS))
            s = int(rng.integers(0, max(1, n - ln)))
            gone[s:s + ln] = True
        seg = ball[a:b]
        seg[gone] = np.nan
        ok = ~np.isnan(seg[:, 0])
        seg[ok, :2] += rng.normal(0, noise_m, (ok.sum(), 2)).astype(np.float32)
    out = dict(d)
    out["ball"] = ball
    return out


def _half_features(ball, players, team, mask=None):
    n = len(ball)
    xy = clean_ball(ball)
    seen = ~np.isnan(xy[:, 0])
    dH, dA = team_dists(xy, players, team)
    near = np.fmin(dH, dA)
    adv = np.where(np.isnan(dH) | np.isnan(dA), 0.0, np.tanh((dA - dH) / CONTROL_SCALE_M))
    adv = np.where(near <= NEAR_M, adv, 0.0)
    shape = np.nan_to_num(width_spread_diff(players, team))
    ps = _mean_speed(players)
    med = np.nanmedian(ps)

    def sm(w):
        return pd.Series(ps).rolling(w, center=True, min_periods=1).mean().to_numpy() / med

    x, y = xy[:, 0], xy[:, 1]
    out = seen & ((x < -0.5) | (x > PITCH[0] + 0.5) | (y < -0.5) | (y > PITCH[1] + 0.5))
    edge = np.where(seen, np.minimum.reduce([x, PITCH[0] - x, y, PITCH[1] - y]), 0.0)
    bsp = np.nan_to_num(ball[:, 2]) if ball.shape[1] > 2 else np.zeros(n)
    f = {
        "adv": adv,
        "near": np.where(np.isnan(near), 3.0, np.minimum(near, 15.0) / 5.0),
        "seen": seen.astype(float),
        "out": out.astype(float),
        "loggap": np.where(seen, 0.0, np.log1p(gap_length(seen)) / np.log1p(1500)),
        "spd_rel": np.clip(np.nan_to_num(sm(SPEED_WINDOW), nan=1.0), 0, 4),
        "spd_rel_1s": np.clip(np.nan_to_num(sm(25), nan=1.0), 0, 4),
        "spd_rel_20s": np.clip(np.nan_to_num(sm(500), nan=1.0), 0, 4),
        "shape": shape,
        "bspeed": np.where(seen, np.minimum(bsp, 30.0) / 10.0, 0.0),
        "still": (seen & (near <= 2.0) & (bsp < 1.0)).astype(float),
        "edge": np.clip(edge, -5, 20) / 10.0,
    }
    return np.stack([f[c] for c in COLS], axis=1).astype(np.float32)


def features(d):
    """(N, len(COLS)) per-frame features, computed per half so smoothing never crosses halftime."""
    starts = list(d["half_start"]) + [len(d["ball"])]
    return np.concatenate([_half_features(d["ball"][a:b], d["players"][a:b], d["team"])
                           for a, b in zip(starts[:-1], starts[1:])])


def mirror(X, y):
    """Swap the teams: flip signed features, swap H<->A labels (D unchanged)."""
    Xm = X.copy()
    for c in SIGNED:
        Xm[:, COLS.index(c)] *= -1
    ym = y.copy()
    ym[y == 0], ym[y == 1] = 1, 0
    return Xm, ym
