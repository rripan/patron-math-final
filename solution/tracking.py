"""Load the raw ATD into numpy arrays (metres), cached under data/cache/ (gitignored)."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, require, source_frames, half_video_blocks  # noqa: E402
from baseline import team_of_each_track  # noqa: E402


def _parse(group):
    if group[0] == "N":
        return np.nan, np.nan, np.nan
    a, b, c = group.split(",")
    return float(a), float(b), float(c)


def load(game=GAME, data_dir=None):
    """Returns dict: ball (N,3) x,y,speed in metres; players (N,S,3); team (S,) 0=H 1=A."""
    meta, raw = require(["atd_meta", "atd_raw"], game, data_dir)
    cache = os.path.join(os.path.dirname(raw), "..", "cache", "atd.npz")
    cache = os.path.abspath(cache)
    if os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}

    n = source_frames("tactical", game)
    length, width = GAMES[game]["pitch"]
    team = team_of_each_track(meta)
    slots = len(team)
    ball = np.full((n, 3), np.nan, dtype=np.float32)
    players = np.full((n, slots, 3), np.nan, dtype=np.float32)
    with open(raw) as fh:
        for line in fh:
            p = line.split(":")
            if len(p) != 3:
                continue
            f = int(p[0]) - 1
            ball[f] = _parse(p[2].strip())
            for s, g in enumerate(p[1].split(";")):
                if s < slots:
                    players[f, s] = _parse(g)
    for arr in (ball, players):
        arr[..., 0] *= length
        arr[..., 1] *= width
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, ball=ball, players=players, team=team)
    return {"ball": ball, "players": players, "team": team}


def graded_mask(game=GAME):
    n = source_frames("tactical", game)
    m = np.zeros(n, dtype=bool)
    for _, a, b in half_video_blocks(game):
        m[a:b + 1] = True
    return m
