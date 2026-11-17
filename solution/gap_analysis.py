"""What do the long ball gaps line up with? Uses Smart Tagging only (no ground truth)."""
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from solution.tracking import load, graded_mask  # noqa: E402
import load_smart_tagging as lst  # noqa: E402

d = load(); ball = d["ball"]; g = graded_mask()
seen = ~np.isnan(ball[:, 0])
ev = lst.load()
N = len(ball)

sp = np.zeros(N, bool)
types = {}
for e in ev:
    if e["code"].lower() == "set pieces":
        sp[e["start_frame"]:e["end_frame"]] = True
print("set pieces:", Counter(e["tags"].get("Type") for e in ev if e["code"].lower() == "set pieces"))
print("fraction of graded frames inside a set-piece interval: %.1f%%" % (100 * sp[g].mean()))
print("ball missing rate: inside set piece %.1f%% | elsewhere %.1f%%" % (
    100 * (~seen[g & sp]).mean(), 100 * (~seen[g & ~sp]).mean()))

# long gaps
m = g & ~seen
idx = np.flatnonzero(np.diff(np.r_[0, m.astype(np.int8), 0]))
st, ln = idx[0::2], idx[1::2] - idx[0::2]
long_ = ln >= 76
print("long gaps: %d" % long_.sum())
hit_end = hit_any = 0
for s, l in zip(st[long_], ln[long_]):
    e = s + l
    if sp[s:e].any(): hit_any += 1
    if sp[e:e + 125].any(): hit_end += 1      # a set piece starts within 5 s after the gap ends
print("  overlap a set-piece interval: %d / %d" % (hit_any, long_.sum()))
print("  a set piece begins within 5 s after the gap ends: %d / %d" % (hit_end, long_.sum()))

# what the ball did just before each long gap started
before = []
for s in st[long_]:
    k = np.flatnonzero(seen[max(0, s - 50):s])
    if len(k):
        j = max(0, s - 50) + k[-1]
        x, y = ball[j, 0], ball[j, 1]
        before.append(not (0 <= x <= 105.3 and 0 <= y <= 68))
before = np.array(before)
print("  last seen ball position was outside the pitch before gap: %d / %d" % (before.sum(), len(before)))

# gap length when overlapping set piece vs not
ov = np.array([sp[s:s + l].any() for s, l in zip(st, ln)])
print("median gap length (frames): with set piece %d | without %d" % (
    np.median(ln[ov]) if ov.any() else 0, np.median(ln[~ov])))

# players: do they stand still / disperse during gaps? speed of tracked players in gap vs seen
spd = d["players"][:, :, 2]
mean_spd = np.nanmean(spd, axis=1)
print("mean player speed m/s: ball missing %.2f | ball seen %.2f" % (
    np.nanmean(mean_spd[g & ~seen]), np.nanmean(mean_spd[g & seen])))
