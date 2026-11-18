# Technical document: frame-level ball possession (More than a Hack 2027, Phase 1)

Team: **Patron**

## 1. Approach in one paragraph

We treat the task as **temporal segmentation with a hidden state**. The state at each frame is one of
`H` (home in possession), `A` (away in possession) or `D` (dead ball). It is never observed, but it
is *persistent*: possessions last seconds, not frames. So the pipeline has two parts:

1. **Evidence.** From the provided tracking and Smart Tagging, compute for every frame a score for
   each of the three states.
2. **Decoding.** A three-state hidden Markov model (HMM) with a switching cost, decoded by Viterbi
   once per half, turns the noisy per-frame scores into a few clean possessions.

We chose this over per-frame classification plus smoothing (the provided baseline) because the
ranking metric (`spellF1`) rewards recovering whole possessions, and a decoder that pays a cost per
switch controls the number of possessions directly. We chose it over a learned model because the
rules forbid training on hand labels from the provided game and we had no labelled data from other
games. No parameter of the pipeline is trained on, or fitted to, any possession labels. External labelled matches (section 3) were used only to *validate* it, and to decide which cues were worth adding.

## 2. Assets used, and dropped

| Asset | Used? | How / why not |
|---|---|---|
| Tactical ATD (tracking incl. ball) | **Yes** | Main input: player and ball positions and speeds, converted to metres. |
| Smart Tagging | **Yes** | (a) Team of the active phase gives a weak home/away prior. (b) Time until the next set piece gives a dead-ball cue. Throw-ins (team `N/A`) contribute nothing to (a). Set pieces are excluded from (a). |
| Tactical homography | No | Tracking already is in pitch coordinates; no pixel-space reasoning needed. |
| Tactical / broadcast video | No | Not needed; the broadcast is also not frame-aligned to the reference. |
| Broadcast ATD / homography | No | No mapping to the tactical timeline is provided, and the added complexity did not fit the time available. |
| Public 5-minute ground truth | **Only to check the pipeline** | Used to confirm frame alignment, file format and label meaning. Not used for training, and not used to choose any parameter or between versions. |

## 3. Training data

**None for fitting.** No labelled data was used to fit any parameter: no hand labels on the provided
game, no event data. Several constants (below) were read off **algorithmic pseudo-labels**
computed from the tracking itself, with no person deciding any label.

**External validation (not training).** We ran the pipeline on three matches of the Sportec Open DFL
dataset (Figshare 28196177, CC BY 4.0; matches J03WMX, J03WN1, J03WOH; 25 Hz tracking; the provider's
`BallStatus`/`BallPossession` mapped to D/H/A, status taking precedence). The competition match is not
part of it. These labels are provider-generated and provisional (they contain many sub-2-second
spells), so scores against them measure *agreement with DFL*, not competition spellF1. Results are in
section 6. DFL labels informed two design decisions, stated openly: (a) the live/dead and team-shape
cues were inspected against them, and (b) the two extra team cues in 4.2 were added because they
helped on all three DFL matches. We also tried, and rejected because they gave no reliable gain on
held-out DFL halves or matches: refitting the speed crossover and switching costs, a learned
logistic-regression emission model trained on ball-degraded DFL, and a dead-ball duration cap. DFL
data is not redistributed (gitignored).

## 4. Pipeline in detail

### 4.1 Cleaning
- Tracking coordinates (0-1) are scaled to metres using the pitch of 105.3 x 68.0 m.
- Single-frame ball spikes (implied speed above 40 m/s into and out of a frame) are removed.
- Ball dropouts are **not interpolated**. Measured on the data, 85% of missing-ball frames lie in
  gaps longer than 10 s and only about 0.1% in gaps under 1 s, so interpolation would help almost
  nothing. A missing ball is handled as missing evidence, not as a dead ball.

### 4.2 Per-frame evidence (`solution/features.py`)
Each frame gets a score vector `E = [H, A, D]`; a frame with no information scores 0 for all three.

*Who controls the ball (H vs A):*
- Nearest-player distance by team, used only when a player is within 5 m of a visible ball
  (`tanh` of the distance advantage over a 3 m scale). If nobody is near, the ball is treated as
  in flight or loose and the evidence is zero, so the hidden state simply persists.
- Smart Tagging phase team, as a weaker prior.
- **Team shape (three cues, home minus away, smoothed over 5 s):** width spread, compactness (mean
  distance of players from their team centre) and the gap in mean player speed (the team without
  the ball runs more). Compactness and the speed gap were added last; DFL AUCs 0.76 and 0.73.
- **Team width spread** (home minus away standard deviation of players across the pitch width,
  smoothed over 5 s). A team in possession stretches the defence; the defending team stays
  compact. It needs no ball, so it is used where the distance evidence is silent (ball missing or
  far). Its weight is fitted by logistic regression, with no intercept, on algorithmic
  pseudo-labels (frames with a visible ball and a player within 2 m, labelled by that player's
  team). Against those pseudo-labels it had an AUC of about 0.80 on our match (0.85 against DFL labels). The three shape cues are combined by the same no-intercept logistic regression on the same pseudo-labels, fitted per match without any possession labels. Two other shape signals (block
  position, depth spread) were tested and dropped as uninformative or weak (AUC 0.53, 0.47, 0.58).

*Is the ball dead (D vs live):*
- Ball visible outside the pitch (strong cue).
- Ball missing inside a gap longer than 3 s (mild cue).
- **5 s-smoothed mean player speed**: slow means standing or walking (dead), fast means live.
- A set piece about to start within 5 s (fading cue).
- A visible, nearly stationary ball with a player within 2 m counts slightly *against* dead.

All weights are written once at the top of `features.py` as football priors. They were chosen by
reasoning, not fitted, except the width-spread weight (pseudo-label regression, above).

**The player-speed crossover (1.4 m/s) is a judgement call and should be read as one.** Pseudo-labels
(fast in-pitch ball = live; ball off the pitch or lying alone = dead) separated live from dead play
with an AUC of about 0.90, but they are biased towards fast play and put the crossover near
1.9 m/s, which gave a 57% dead share. We set the crossover to 1.4 m/s so that the decoded dead share
is about 40%, matching the generic football prior that the ball is in play about 60% of a match.
It was not tuned on any ground truth. On DFL the real dead share is 41.5%, 1.4 m/s sits at the 37th percentile of its player-speed distribution (43rd on ours), and alive/dead F1 is about 0.9, which supports it.

### 4.3 Decoding (`solution/decode.py`)
A three-state HMM. Emission scores are the vectors above. Transition costs: **30** for a possession
hand-over (`H<->A`) and **40** for entering or leaving a dead ball (`live<->D`), zero to stay. These
are priors meant to require a few seconds of consistent evidence before a switch, and were not tuned.
Viterbi runs separately on each graded half; frames outside the two halves are written as `X`.

## 5. Possession rules we implemented

- A possession is a maximal run of one team with the ball alive; a dead-ball stretch always splits
  possessions (this follows from how `H`/`A`/`D` are decoded).
- **Loose or contested ball:** possession stays with the last team (no evidence, so no switch).
- **Controlled possession:** approximated by the switching cost. A short touch by an opponent
  does not flip the state because it cannot outweigh the hand-over cost; sustained opposing control
  can.
- **Dead ball:** no owner is ever assigned; the output is `D`.
- **Restarts:** only through a soft cue (a set piece about to start pushes towards dead ball). We do
  not force a particular team to restart, and throw-ins (team `N/A`) are not handled specially.
- We never write `X` on graded frames. The scorer fills `X` with the previous label on the ranking
  metric, so abstaining would not have helped.

## 6. Results and what we can and cannot claim

On the full match, our file contains **229 possessions of mean 16.3 s (median 10.4 s)**, with a label
share of about `H` 36%, `A` 23%, `D` 41%. The provided best baseline produces 669 possessions of
mean 7.6 s on the same match.

On the public 5 minutes the file scored macroF1 0.78 and spellF1 0.44 with 17 possessions against 17
in the ground truth. **We do not rely on that number:** 17 possessions cannot rank anything, and we
only used it to check that the pipeline is correct. We do not know our full-match score.

**External validation on DFL** (three matches, ball additionally deleted in long blocks to about 41%
missing to mimic our tracker, Smart Tagging off, no fitting): macroF1 0.74-0.80, alive/dead F1
about 0.89-0.91, correct team when both agree the ball is live about 75-84%. spellF1 against DFL is
only 0.26-0.34 because our decoder finds about 40% as many spells as DFL's fragmented labels.
Adding the compactness and speed-gap cues raised spellF1 by 0.015-0.027 and team accuracy by about
1 point on every one of the three matches. These are agreement numbers with provisional labels.

## 7. Limitations

- **The missing ball is the main weakness.** In long gaps we only have indirect evidence (player
  speed, team width, Smart Tagging), so both the live/dead decision and the team are weaker there.
  A few dead-ball runs still exceed 60 s (9 in the match), which is rare in real football, so some
  live play with a missing ball is probably still labelled `D`.
- The team-shape cues were validated on frames where the ball is visible, so it may be weaker
  on the ball-missing frames where it is actually used. The signal also has a small bias towards
  home (home tends to be wider), which may add a slight home lean there.
- DFL tracking is cleaner than ours and always has the ball, so validation on it may overstate performance; we mimicked our gaps and noise only approximately. The 1.4 m/s crossover and the transition costs are priors, not measured optima, and fitting them on DFL gave no reliable gain. A different
  reasonable choice would change the possession count and the dead share.
- No restart modelling: we do not use Smart Tagging to decide which team restarts.
- We did not use the video, homography or broadcast feed.

**With more time:** estimate the transition costs and weights from other games with labels (allowed
if shared); model dead-ball duration explicitly (a simple cap did not help on DFL, where real stoppages reach 2+ minutes);
use set-piece types to choose the restarting team; use the broadcast feed to recover the ball in
long gaps.

## 8. Reproduction

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r solution/requirements.txt      # numpy 2.5.4, pandas 3.1.1 (Python 3.13.11)
# unpack the provided bundle into data/ (or set MTAH_DATA_DIR)
python solution/run.py --no-cache --out Patron.csv
python evaluation/validate_submission.py Patron.csv
```

`--no-cache` rebuilds from the raw files; without it the parsed tracking is cached under
`data/cache/` (gitignored, must never be published). The run is deterministic: two clean runs gave a
byte-identical file (SHA-256 `278c3acf915d035d96e2891c74a8f19a180c4230fdf249c5945d82b57f76546c`).
The DFL validation scripts (`solution/dfl_*.py`, `learned*.py`, `durcap.py`) are optional and need
the DFL files in `external/dfl/` and scikit-learn; the submission does not. The match data is not in this repository.

## 9. Use of LLMs

Claude was used to debug code for the Viterbi algorithm. The raw data files, videos and
tracking rows were not uploaded or pasted to it. The assistant worked with snippets of code locally and saw only
aggregate summary statistics (for example gap-length histograms, label shares, correlations and
possession counts) and the scorer output. All of it was reviewed by the team.
