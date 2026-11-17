"""Summary statistics of the ball track (no raw rows printed)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solution.tracking import load, graded_mask  # noqa: E402

d = load()
ball = d["ball"]
g = graded_mask()
seen = ~np.isnan(ball[:, 0])
print("frames", len(ball), "graded", g.sum())
print("ball seen: all %.1f%%  graded %.1f%%" % (100 * seen.mean(), 100 * seen[g].mean()))

# gap lengths inside graded frames
def runs(mask):
    idx = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return idx[0::2], idx[1::2] - idx[0::2]

for name, m in (("graded & missing", g & ~seen),):
    st, ln = runs(m)
    print(name, "gaps:", len(ln))
    for lo, hi in ((1, 5), (6, 25), (26, 75), (76, 250), (251, 10**9)):
        k = (ln >= lo) & (ln <= hi)
        print("  gap %4d-%-9d n=%5d frames=%6d (%.1f%% of missing)" %
              (lo, hi, k.sum(), ln[k].sum(), 100 * ln[k].sum() / ln.sum()))

# speed / jumps between consecutive seen frames
both = seen[1:] & seen[:-1] & g[1:]
step = np.hypot(*(np.diff(ball[:, :2], axis=0)[both].T)) * 25  # m/s
print("implied ball speed m/s percentiles 50/90/99/99.9/max:",
      np.round(np.percentile(step, [50, 90, 99, 99.9, 100]), 1))
print("steps > 40 m/s (physically implausible): %d (%.3f%%)" % ((step > 40).sum(), 100 * (step > 40).mean()))

# on-pitch
x, y = ball[seen & g, 0], ball[seen & g, 1]
out = (x < 0) | (x > 105.3) | (y < 0) | (y > 68)
print("seen frames outside pitch: %.1f%%" % (100 * out.mean()))

# players per frame
cnt = (~np.isnan(d["players"][g][:, :, 0])).sum(1)
print("tracked players per graded frame: mean %.1f  p5 %d  p95 %d" % (cnt.mean(), *np.percentile(cnt, [5, 95])))
