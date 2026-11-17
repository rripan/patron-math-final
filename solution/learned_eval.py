"""Leave-one-match-out test of the learned emissions vs the hand-set priors, on DFL (external).
Held-out matches are scored both clean and degraded (ball deleted in blocks, ~41%, like ours)."""
import itertools
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from solution.dfl_loader import load, MATCHES  # noqa: E402
from solution import learned  # noqa: E402
from solution.features import build  # noqa: E402
from solution.decode import viterbi, LABELS  # noqa: E402
import possession_metrics as pm  # noqa: E402

CODE = {"H": 0, "A": 1, "D": 2}
GRID = [(ha, dd) for ha in (3, 6, 10, 20, 30) for dd in (6, 10, 20, 40)]


def codes(d):
    return np.array([CODE.get(l, 2) for l in d["labels"]])


def decode(E, d, ha, dd):
    T = np.array([[0, ha, dd], [ha, 0, dd], [dd, dd, 0]], float)
    s = list(d["half_start"]) + [len(E)]
    return np.concatenate([LABELS[viterbi(E[a:b], T)] for a, b in zip(s[:-1], s[1:])])


def score(E, d, ha, dd):
    pred = decode(E, d, ha, dd)
    s = list(d["half_start"]) + [len(E)]
    g, p = [], []
    for a, b in zip(s[:-1], s[1:]):
        g += [d["labels"][a:b], np.array(["X"])]; p += [pred[a:b], np.array(["X"])]
    r = pm.evaluate(pm.label_codes(np.concatenate(g)), pm.label_codes(np.concatenate(p)), bootstrap_n=0)
    return r


def logp(model, X):
    return np.log(np.clip(model.predict_proba(X), 1e-4, 1)).astype(np.float32)


def fit_model(ds, seeds=(0, 1), C=1.0):
    Xs, ys = [], []
    for d in ds:
        for sd in seeds:
            dd_ = learned.degrade(d, np.random.default_rng(100 + sd))
            X, y = learned.features(dd_), codes(d)
            Xm, ym = learned.mirror(X, y)
            Xs += [X, Xm]; ys += [y, ym]
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=400))
    m.fit(np.concatenate(Xs), np.concatenate(ys))
    return m


if __name__ == "__main__":
    avail = [m for m in MATCHES if os.path.exists(learned.os.path.join(ROOT, "external", "dfl", f"DFL_04_03_positions_raw_observed_{m}.xml"))]
    data = {m: load(m) for m in avail}
    print("matches:", [m[-6:] for m in avail])
    rows = []
    for test in avail:
        train = [data[m] for m in avail if m != test]
        model = fit_model(train)
        # choose switching costs on the TRAIN matches (degraded, fresh seed), never on the test one
        best = None
        for ha, dd in GRID:
            sc = []
            for d in train:
                dg = learned.degrade(d, np.random.default_rng(999))
                r = score(logp(model, learned.features(dg)), d, ha, dd)
                sc.append((r["macro_f1"] + r["spell_f1_mean"]) / 2)
            if best is None or np.mean(sc) > best[0]:
                best = (np.mean(sc), ha, dd)
        _, ha, dd = best
        d = data[test]
        dg = learned.degrade(d, np.random.default_rng(7))
        for name, dx in (("clean", d), ("degraded", dg)):
            rl = score(logp(model, learned.features(dx)), d, ha, dd)
            rp = score(build(dx, tags=False)[0], d, 30, 40)
            print(f"{test[-6:]} {name:8s} LEARNED(ha={ha},d={dd}) macro {rl['macro_f1']:.3f} spell {rl['spell_f1_mean']:.3f} "
                  f"a/d {rl['alive_dead_f1']:.3f} team {rl['team_acc_both_alive']:.3f} spells {int(rl['n_pred_spells'])}/{int(rl['n_gt_spells'])}"
                  f" | PRIOR macro {rp['macro_f1']:.3f} spell {rp['spell_f1_mean']:.3f} a/d {rp['alive_dead_f1']:.3f} team {rp['team_acc_both_alive']:.3f}")
