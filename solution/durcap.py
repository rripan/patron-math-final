"""TRIAL: cap on dead-ball run length. Where the decoded D run exceeds `cap` frames, re-decode that
stretch with the D emission lowered by the smallest delta that makes every D run <= cap. External DFL
data shows real dead runs reach ~110-150 s (90th pct ~50 s), so the cap is set well above that."""
import numpy as np
from solution.decode import viterbi


def _drun(path):
    i = np.flatnonzero(np.r_[True, path[1:] != path[:-1]])
    n = np.diff(np.r_[i, len(path)])
    return [(a, a + l) for a, l in zip(i, n) if path[a] == 2]


def cap_dead(E, T, cap):
    path = viterbi(E, T)
    for a, b in _drun(path):
        if b - a <= cap:
            continue
        pad = 250
        lo, hi = max(0, a - pad), min(len(E), b + pad)
        seg = E[lo:hi].copy()
        for delta in np.arange(0.1, 6.0, 0.1):
            seg[:, 2] = E[lo:hi, 2] - delta
            p = viterbi(seg, T)
            if all(y - x <= cap for x, y in _drun(p)):
                path[lo:hi] = p
                break
    return path
