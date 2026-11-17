"""Stage B: per-frame evidence for H / A / D.

Everything here is derived from the provided tracking and Smart Tagging by fixed rules. The
weights are football-prior choices written down once -- they are NOT fitted to or tuned on any
ground truth (public or private). See the WEIGHTS block.

Output: emission scores E (N, 3) for states [H, A, D]; higher = more plausible. A frame with no
information scores 0 for all three, i.e. the HMM then relies on its transition costs alone.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from solution.tracking import load  # noqa: E402
import load_smart_tagging as lst  # noqa: E402

FPS = 25
PITCH = (105.3, 68.0)

# ---- WEIGHTS (football priors, not tuned) ----------------------------------------------------
MAX_BALL_SPEED = 40.0     # m/s; no football travels faster (hardest shots ~35 m/s)
NEAR_M = 5.0              # a player farther than this from the ball is not "on" it
CONTROL_SCALE_M = 3.0     # a 3 m advantage in nearest-player distance is a strong signal
W_CONTROL = 2.0           # weight of nearest-player evidence (log-odds H vs A)
W_TAG = 0.7               # weight of Smart Tagging phase team (log-odds H vs A)
W_OUT = 3.0               # ball visible outside the pitch: strong dead cue
W_NOBALL_LONG = 1.0       # ball absent in a long gap: mild dead cue
LONG_GAP = 75             # frames (3 s)
# Player-speed cue. 5 s-smoothed mean tracked-player speed separates live from dead play well (AUC
# ~0.90 against algorithmic pseudo-labels: fast in-pitch ball = live, ball off the pitch or lying
# alone = dead; no human or ground-truth labels). The pseudo-labels are biased toward fast play, so
# they overstate the crossover (~1.9 m/s) and give a 57% dead share. SPEED_MID_MS is instead set
# so the decoded dead share is ~40%, matching the generic football prior that the ball is in play
# ~60% of a match (typical effective playing time). It is NOT tuned on any ground truth.
W_SPEED = 2.5             # max weight of the player-speed cue (dead if slow, live if fast)
SPEED_MID_MS = 1.4
SPEED_SPAN_MS = 0.7
SPEED_WINDOW = 125        # frames (5 s)
W_SP_NEXT = 0.8           # a set piece about to start: dead cue, fading over 5 s
SP_HORIZON = 125          # frames
W_BALL_STILL = 1.0        # ball visible and nearly stationary with a player near: live control
# -----------------------------------------------------------------------------------------------


def clean_ball(ball):
    """Mask single-frame spikes (implied speed > MAX in and out). Returns (N,2) with NaN."""
    xy = ball[:, :2].copy()
    seen = np.flatnonzero(~np.isnan(xy[:, 0]))
    if len(seen) < 3:
        return xy
    p = xy[seen]
    dt = np.diff(seen)
    sp = np.hypot(*np.diff(p, axis=0).T) / dt * FPS
    bad_in = np.r_[False, (sp > MAX_BALL_SPEED) & (dt <= FPS)]
    bad_out = np.r_[(sp > MAX_BALL_SPEED) & (dt <= FPS), False]
    xy[seen[bad_in & bad_out]] = np.nan
    return xy


def team_dists(xy, players, team):
    n = len(xy)
    out = []
    for t in (0, 1):
        P = players[:, team == t, :2]
        d = np.hypot(P[:, :, 0] - xy[:, None, 0], P[:, :, 1] - xy[:, None, 1])
        with np.errstate(all="ignore"):
            m = np.where(np.all(np.isnan(d), axis=1), np.nan, np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1))
        m[np.isinf(m)] = np.nan
        out.append(m)
    return out


def tag_team_signal(n):
    """+1 where the active phase's team is home, -1 away, 0 none/N-A. Set pieces excluded."""
    s = np.zeros(n)
    for e in lst.load():
        if e["code"].lower() == "set pieces":
            continue
        team = (e["tags"].get("Team") or "").lower()
        v = 1.0 if team.startswith("barcelona") else -1.0 if team.startswith("atletic") else 0.0
        s[e["start_frame"]:e["end_frame"]] += v
    return np.clip(s, -1, 1)


def frames_to_next_set_piece(n):
    starts = sorted(e["start_frame"] for e in lst.load() if e["code"].lower() == "set pieces")
    nxt = np.full(n, 10 ** 6, dtype=np.int64)
    j = len(starts) - 1
    last = 10 ** 6
    ptr = 0
    st = np.array(starts)
    idx = np.searchsorted(st, np.arange(n), side="left")
    ok = idx < len(st)
    nxt[ok] = st[idx[ok]] - np.arange(n)[ok]
    return nxt


def gap_length(seen):
    n = len(seen)
    out = np.zeros(n, dtype=np.int64)
    idx = np.flatnonzero(np.diff(np.r_[0, (~seen).astype(np.int8), 0]))
    for s, e in zip(idx[0::2], idx[1::2]):
        out[s:e] = e - s
    return out


def width_spread_diff(players, team):
    """Home minus away spread of players across the pitch width, smoothed over 5 s.

    A team in possession spreads wide to stretch the defence; the defending team stays compact.
    Needs no ball, so it is available where the ball is missing.
    """
    out = []
    for t in (0, 1):
        y = players[:, team == t, 1]
        ok = (~np.isnan(y)).sum(1) >= 5
        with np.errstate(all="ignore"):
            sd = np.where(ok, np.nanstd(y, axis=1), np.nan)
        out.append(sd)
    return pd.Series(out[0] - out[1]).rolling(SPEED_WINDOW, center=True, min_periods=25).mean().to_numpy()


def fit_shape_weight(shape, label_home, mask):
    """Log-odds weight (no intercept) of the shape signal, fitted by Newton's method on algorithmic
    pseudo-labels only: frames where the ball is visible and a player is within 2 m, labelled by
    the nearest player's team. No human or ground-truth labels are involved."""
    x, y = shape[mask], label_home[mask].astype(float)
    w = 0.0
    for _ in range(50):
        p = 1 / (1 + np.exp(-w * x))
        step = np.sum((p - y) * x) / max(np.sum(p * (1 - p) * x * x), 1e-9)
        w -= step
        if abs(step) < 1e-9:
            break
    return w


def _smooth(x):
    return pd.Series(x).rolling(SPEED_WINDOW, center=True, min_periods=25).mean().to_numpy()


def compact_diff(players, team):
    out = []
    for t in (0, 1):
        X, Y = players[:, team == t, 0], players[:, team == t, 1]
        with np.errstate(all="ignore"):
            out.append(np.nanmean(np.hypot(X - np.nanmean(X, 1, keepdims=True), Y - np.nanmean(Y, 1, keepdims=True)), 1))
    return _smooth(out[0] - out[1])


def speed_gap(players, team):
    with np.errstate(all="ignore"):
        a = np.nanmean(players[:, team == 0, 2], 1) - np.nanmean(players[:, team == 1, 2], 1)
    return _smooth(a)


def fit_shape_weights(Z, y, iters=50):
    """Multivariate no-intercept logistic weights by Newton (pseudo-labels only)."""
    w = np.zeros(Z.shape[1]); y = y.astype(float)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (p - y)
        H = (Z * (p * (1 - p))[:, None]).T @ Z + 1e-6 * np.eye(len(w))
        step = np.linalg.solve(H, g); w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def build(d=None, tags=True, speed_q=None, span_frac=0.5, mask=None, extra_shape=True):
    """tags=False drops Smart Tagging (for external matches, which have no equivalent)."""
    d = d or load()
    ball, players, team = d["ball"], d["players"], d["team"]
    n = len(ball)
    xy = clean_ball(ball)
    seen = ~np.isnan(xy[:, 0])

    dH, dA = team_dists(xy, players, team)
    near = np.fmin(dH, dA)

    # control evidence: only when someone is close to a visible ball
    adv = np.where(np.isnan(dH) | np.isnan(dA), 0.0, np.tanh((dA - dH) / CONTROL_SCALE_M))
    adv = np.where(near <= NEAR_M, adv, 0.0)
    c = W_CONTROL * adv + (W_TAG * tag_team_signal(n) if tags else 0.0)

    # team-shape evidence, used only where the distance evidence is silent (ball missing / far)
    shape = width_spread_diff(players, team)
    fit_mask = seen & (near <= 2.0) & ~np.isnan(shape)
    w_shape = fit_shape_weight(np.nan_to_num(shape), dH < dA, fit_mask)
    silent = (near > NEAR_M) | np.isnan(near)
    if extra_shape:  # TRIAL: add team-compactness and team-speed gaps, weights fitted on the same pseudo-labels
        Z = np.stack([np.nan_to_num(shape), np.nan_to_num(compact_diff(players, team)),
                      np.nan_to_num(speed_gap(players, team))], axis=1)
        Z = Z / np.maximum(Z[fit_mask].std(0), 1e-6)
        w = fit_shape_weights(Z[fit_mask], (dH < dA)[fit_mask])
        c = c + np.where(silent, Z @ w, 0.0)
    else:
        c = c + np.where(silent, w_shape * np.nan_to_num(shape), 0.0)

    # dead evidence
    x, y = xy[:, 0], xy[:, 1]
    out = seen & ((x < -0.5) | (x > PITCH[0] + 0.5) | (y < -0.5) | (y > PITCH[1] + 0.5))
    glen = gap_length(seen)
    pspeed = pd.Series(_mean_speed(players)).rolling(SPEED_WINDOW, center=True, min_periods=1).mean().to_numpy()
    nxt = frames_to_next_set_piece(n) if tags else np.full(n, 10 ** 6)
    sp_cue = np.clip(1 - nxt / SP_HORIZON, 0, 1)

    mid, span = SPEED_MID_MS, SPEED_SPAN_MS
    if speed_q is not None:  # per-match crossover: a percentile of this match's own speed distribution
        ref = pspeed if mask is None else pspeed[mask]
        mid = float(np.nanpercentile(ref, 100 * speed_q))
        span = span_frac * mid

    s_dead = (W_OUT * out
              + W_NOBALL_LONG * ((~seen) & (glen >= LONG_GAP))
              + W_SPEED * np.clip((mid - np.nan_to_num(pspeed, nan=mid)) / span, -1, 1)
              + W_SP_NEXT * sp_cue
              - W_BALL_STILL * (seen & (near <= 2.0) & (ball[:, 2] < 1.0)))

    # states [H, A, D]; an uninformative frame (c=0, s_dead=0) scores 0 everywhere
    E = np.stack([-s_dead / 2 + c / 2, -s_dead / 2 - c / 2, s_dead / 2], axis=1).astype(np.float32)
    feats = dict(seen=seen, out=out, dH=dH, dA=dA, near=near, c=c, s_dead=s_dead, pspeed=pspeed, glen=glen)
    return E, feats


def _mean_speed(players):
    sp = players[:, :, 2]
    cnt = (~np.isnan(sp)).sum(1)
    tot = np.nansum(sp, axis=1)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


if __name__ == "__main__":
    from solution.tracking import graded_mask
    E, f = build()
    g = graded_mask()
    print("graded frames:", g.sum())
    print("ball usable: %.1f%% | someone within %.0f m: %.1f%%" % (
        100 * f["seen"][g].mean(), NEAR_M, 100 * (f["near"][g] <= NEAR_M).mean()))
    print("control evidence nonzero: %.1f%% of graded" % (100 * (np.abs(f["c"][g]) > 0).mean()))
    print("dead evidence > 0: %.1f%% | > 1.5: %.1f%%" % (100 * (f["s_dead"][g] > 0).mean(), 100 * (f["s_dead"][g] > 1.5).mean()))
    print("argmax of emissions alone  H %.1f%%  A %.1f%%  D %.1f%%" % tuple(
        100 * np.mean(np.argmax(E[g], axis=1) == k) for k in range(3)))
    os.makedirs(os.path.join(ROOT, "data", "cache"), exist_ok=True)
    np.save(os.path.join(ROOT, "data", "cache", "emissions.npy"), E)
