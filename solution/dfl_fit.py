"""Small DFL fit with a held-out half. Fits 4 numbers on the FIRST half only, scores the SECOND:
  speed_q   percentile of the match's own 5 s player-speed distribution used as the live/dead
            crossover (replaces the absolute 1.4 m/s)
  span_frac width of the crossover as a fraction of that speed
  cost_ha, cost_d   Viterbi switching costs
Objective = mean(macroF1, spellF1 mean). Then repeated with halves swapped. External data only."""
import itertools
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from solution.dfl_loader import load  # noqa: E402
from solution.features import build  # noqa: E402
from solution.decode import viterbi, LABELS  # noqa: E402
import possession_metrics as pm  # noqa: E402

GRID = dict(q=[0.30, 0.35, 0.40, 0.45, 0.50], span=[0.35, 0.5, 0.7], ha=[10, 20, 30, 50], dd=[20, 40, 60])


def score(E, lab, ha, dd):
    T = np.array([[0, ha, dd], [ha, 0, dd], [dd, dd, 0]], float)
    pred = LABELS[viterbi(E, T)]
    r = pm.evaluate(pm.label_codes(lab), pm.label_codes(pred), bootstrap_n=0)
    return r["macro_f1"], r["spell_f1_mean"], r


def fit(d, Efull_by_cfg, fit_sl, test_sl):
    best = None
    for (q, sp), E in Efull_by_cfg.items():
        for ha, dd in itertools.product(GRID["ha"], GRID["dd"]):
            m, s, _ = score(E[fit_sl], d["labels"][fit_sl], ha, dd)
            if best is None or (m + s) / 2 > best[0]:
                best = ((m + s) / 2, q, sp, ha, dd)
    _, q, sp, ha, dd = best
    mt, st, rt = score(Efull_by_cfg[(q, sp)][test_sl], d["labels"][test_sl], ha, dd)
    return best, mt, st, rt


if __name__ == "__main__":
    d = load()
    h = int(d["half_start"][1]); n = len(d["labels"])
    halves = {"H1": slice(0, h), "H2": slice(h, n)}
    # percentile crossover is computed per HALF (match-own distribution), so build per half
    Es = {}
    for name, sl in halves.items():
        sub = {k: (v[sl] if k in ("ball", "players", "labels") else v) for k, v in d.items()}
        Es[name] = {(q, sp): build(sub, tags=False, speed_q=q, span_frac=sp)[0]
                    for q in GRID["q"] for sp in GRID["span"]}
    # baseline: hand-set priors, per half
    for name, sl in halves.items():
        sub = {k: (v[sl] if k in ("ball", "players", "labels") else v) for k, v in d.items()}
        m, s, _ = score(build(sub, tags=False)[0], d["labels"][sl], 30, 40)
        print(f"PRIOR  {name}: macroF1 {m:.3f} spellF1 {s:.3f}")
    for fit_h, test_h in (("H1", "H2"), ("H2", "H1")):
        best, mt, st, rt = fit({"labels": d["labels"][halves[fit_h]]}, Es[fit_h],
                               slice(None), slice(None)) if False else (None, None, None, None)
        # fit on fit_h, test on test_h (emissions of the two halves are built separately)
        b = None
        for (q, sp), E in Es[fit_h].items():
            for ha, dd in itertools.product(GRID["ha"], GRID["dd"]):
                m, s, _ = score(E, d["labels"][halves[fit_h]], ha, dd)
                if b is None or (m + s) / 2 > b[0]:
                    b = ((m + s) / 2, q, sp, ha, dd, m, s)
        _, q, sp, ha, dd, mf, sf = b
        mt, st, rt = score(Es[test_h][(q, sp)], d["labels"][halves[test_h]], ha, dd)
        print(f"FIT on {fit_h}: q={q} span_frac={sp} cost_ha={ha} cost_d={dd} | fit-half macroF1 {mf:.3f} spellF1 {sf:.3f}")
        print(f"  held-out {test_h}: macroF1 {mt:.3f} spellF1 {st:.3f} alive/dead F1 {rt['alive_dead_f1']:.3f} "
              f"spells {int(rt['n_pred_spells'])} vs {int(rt['n_gt_spells'])}")
