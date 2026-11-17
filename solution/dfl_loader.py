"""Load a Sportec Open DFL match (Figshare 28196177, CC BY 4.0) into the same array layout as
solution/tracking.py, plus algorithmic H/A/D labels from the provider's BallStatus/BallPossession.

EXTERNAL data only: used to check/fit the pipeline, never the competition match. Output is cached
under external/dfl/cache/ (gitignored).

Layout: ball (N,3) x,y,speed in metres (origin = pitch corner); players (N,S,3); team (S,) 0=home
1=away. Halves are concatenated; `half_start` gives the first row of each half. Provider
frames are 25 Hz. Label mapping (status takes precedence, per the audit notes):
  BallStatus=0 -> D ; BallStatus=1 & BallPossession=1 -> H ; =2 -> A ; anything else -> X.
"""
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DFL_DIR = os.path.join(ROOT, "external", "dfl")
MATCH = "DFL-COM-000001_DFL-MAT-J03WMX"
MATCHES = ["DFL-COM-000001_DFL-MAT-J03WMX", "DFL-COM-000001_DFL-MAT-J03WN1", "DFL-COM-000002_DFL-MAT-J03WOH"]
PITCH = (105.0, 68.0)
KMH = 3.6  # provider S is km/h (median player speed 6.3 = 1.76 m/s x 3.6); we want m/s
_FS = re.compile(r'<FrameSet GameSection="(\w+)" MatchId="[^"]*" TeamId="([^"]*)" PersonId="([^"]*)"')
_FR = re.compile(r'<Frame N="(\d+)" T="[^"]*" X="([-\d.]+)" Y="([-\d.]+)"(?: Z="([-\d.]+)")? '
                 r'D="[-\d.]+" S="([-\d.]+)"(?:[^>]*BallPossession="(\d)" BallStatus="(\d)")?')


def _teams(match):
    txt = open(os.path.join(DFL_DIR, f"DFL_02_01_matchinformation_{match}.xml"), encoding="utf-8").read()
    return dict(re.findall(r'<Team TeamId="([^"]+)"[^>]*Role="(home|guest)"', txt))


def load(match=MATCH):
    cache = os.path.join(DFL_DIR, "cache", "dfl.npz" if match == MATCH else f"dfl_{match[-6:]}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    role = _teams(match)
    path = os.path.join(DFL_DIR, f"DFL_04_03_positions_raw_observed_{match}.xml")
    persons, ball, rows = {}, {}, {}  # rows[section][(person)] = {N: (x,y,s)}
    sec = who = None
    ball = {"firstHalf": {}, "secondHalf": {}}
    pl = {"firstHalf": {}, "secondHalf": {}}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("<FrameSet"):
                m = _FS.match(line)
                sec, team, who = m.group(1), m.group(2), m.group(3)
                is_ball = team == "BALL"
                skip = team == "referee"
                if not is_ball and not skip:
                    persons.setdefault(who, 0 if role[team] == "home" else 1)
                    pl[sec][who] = {}
            elif line.startswith("<Frame ") and not skip:
                m = _FR.match(line)
                n, x, y, z, s, bp, bs = m.groups()
                n = int(n)
                if is_ball:
                    ball[sec][n] = (float(x), float(y), float(s), int(bp) if bp else -1, int(bs) if bs else -1)
                else:
                    pl[sec][who][n] = (float(x), float(y), float(s))
    ids = sorted(persons)
    team = np.array([persons[i] for i in ids], dtype=np.int8)
    B, P, L, starts = [], [], [], []
    n_total = 0
    for sec in ("firstHalf", "secondHalf"):
        fr = np.array(sorted(ball[sec]))
        idx = {n: i for i, n in enumerate(fr)}
        b = np.full((len(fr), 3), np.nan, dtype=np.float32)
        lab = np.full(len(fr), "X", dtype="<U1")
        for n, (x, y, s, bp, bs) in ball[sec].items():
            i = idx[n]
            b[i] = (x + PITCH[0] / 2, y + PITCH[1] / 2, s / KMH)
            lab[i] = "D" if bs == 0 else ("H" if (bs == 1 and bp == 1) else ("A" if (bs == 1 and bp == 2) else "X"))
        p = np.full((len(fr), len(ids), 3), np.nan, dtype=np.float32)
        for j, who in enumerate(ids):
            for n, (x, y, s) in pl[sec].get(who, {}).items():
                if n in idx:
                    p[idx[n], j] = (x + PITCH[0] / 2, y + PITCH[1] / 2, s / KMH)
        starts.append(n_total)
        n_total += len(fr)
        B.append(b); P.append(p); L.append(lab)
    out = {"ball": np.concatenate(B), "players": np.concatenate(P), "team": team,
           "labels": np.concatenate(L), "half_start": np.array(starts)}
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


if __name__ == "__main__":
    d = load()
    lab = d["labels"]
    n = len(lab)
    print("frames", n, "halves start", d["half_start"], "slots", len(d["team"]),
          "home slots", int((d["team"] == 0).sum()))
    print("label share", {k: round(float((lab == k).mean()) * 100, 1) for k in "HADX"})
    print("ball seen %", round(float(np.isfinite(d["ball"][:, 0]).mean()) * 100, 1))
    print("players seen per frame", round(float(np.isfinite(d["players"][..., 0]).sum(1).mean()), 1))
    run = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1]])
    ln = np.diff(np.r_[run, n]) / 25
    for k in "HAD":
        s = np.array([ln[i] for i, r in enumerate(run) if lab[r] == k])
        print(k, "runs", len(s), "mean s", round(float(s.mean()), 1), "median", round(float(np.median(s)), 1),
              "runs<1s", int((s < 1).sum()))
