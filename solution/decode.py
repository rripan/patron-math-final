"""Stage C: Viterbi decoding of the emission scores into H / A / D, one half at a time.

Transition costs are football priors, not fitted: a hand-over or a stoppage should need a few
seconds of consistent evidence, and the restart after a stoppage likewise.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, half_video_blocks, source_frames  # noqa: E402
from solution.features import build  # noqa: E402

LABELS = np.array(["H", "A", "D"])
# ---- TRANSITION COSTS (priors) ---------------------------------------------------------------
COST_HA = 30.0   # possession hand-over H<->A
COST_D = 40.0    # live<->dead in either direction
# -----------------------------------------------------------------------------------------------


def transition_matrix():
    T = np.array([[0, COST_HA, COST_D],
                  [COST_HA, 0, COST_D],
                  [COST_D, COST_D, 0]], dtype=np.float64)
    return T


def viterbi(E, T):
    n = len(E)
    score = E[0].astype(np.float64)
    back = np.zeros((n, 3), dtype=np.int8)
    for t in range(1, n):
        cand = score[:, None] - T          # cand[i, j] = score[i] - cost(i -> j)
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(3)] + E[t]
    path = np.zeros(n, dtype=np.int8)
    path[-1] = np.argmax(score)
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(E, game=GAME):
    n = source_frames("tactical", game)
    out = np.full(n, "X", dtype="<U1")
    T = transition_matrix()
    for _, a, b in half_video_blocks(game):
        out[a:b + 1] = LABELS[viterbi(E[a:b + 1], T)]
    return out


def write(path, labels):
    with open(path, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))


def stats(labels):
    live = labels[labels != "X"]
    runs = []
    cur, ln = None, 0
    for l in labels:
        if l in ("H", "A") and l == cur:
            ln += 1
        else:
            if cur in ("H", "A"):
                runs.append(ln)
            cur, ln = (l, 1) if l in ("H", "A") else (None, 0)
    if cur in ("H", "A"):
        runs.append(ln)
    runs = np.array(runs)
    share = {k: round(100 * float(np.mean(live == k)), 1) for k in "HAD"}
    return "possessions %d, mean %.1f s, median %.1f s | share %s" % (
        len(runs), runs.mean() / 25, np.median(runs) / 25, share)


if __name__ == "__main__":
    E, _ = build()
    labels = decode(E)
    path = os.path.join(ROOT, "evaluation", "results", "hmm_v1.csv")
    write(path, labels)
    print(stats(labels))
    print("wrote", os.path.relpath(path, ROOT))
