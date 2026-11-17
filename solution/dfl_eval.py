"""Run the unchanged pipeline (tags off, hand-set priors) on an external DFL match and score its
agreement with the provider's labels. Agreement with DFL is NOT competition spellF1."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from solution.dfl_loader import load  # noqa: E402
from solution.features import build  # noqa: E402
from solution.decode import viterbi, transition_matrix, LABELS  # noqa: E402
import possession_metrics as pm  # noqa: E402


def run(d=None, E=None):
    d = d or load()
    E = build(d, tags=False)[0] if E is None else E
    starts = list(d["half_start"]) + [len(E)]
    pred, gt = [], []
    T = transition_matrix()
    for a, b in zip(starts[:-1], starts[1:]):
        pred.append(LABELS[viterbi(E[a:b], T)]); gt.append(d["labels"][a:b])
        pred.append(np.array(["X"])); gt.append(np.array(["X"]))  # halftime separator
    g = pm.label_codes(np.concatenate(gt)); p = pm.label_codes(np.concatenate(pred))
    return pm.evaluate(g, p, bootstrap_n=0), np.concatenate(pred)[:-1]


if __name__ == "__main__":
    r, pred = run()
    for k in ("macro_f1", "spell_f1_mean", "n_gt_spells", "n_pred_spells", "mean_spell_duration_s",
              "gt_mean_spell_duration_s", "alive_dead_f1", "team_acc_both_alive", "gt_home_share", "pred_home_share"):
        print(k, round(float(r[k]), 3))
