# STEP1-PREREG: do naked mole-rat colony "dialects" carry over from one recording day to another?

molerat-dialect-test, step 1. MV2, Munim, Inc. Made with AI agents working in MV2's name; no
human expert has checked it yet. Registered 2026-11-21, before any acoustic feature or classifier
was computed on the real data by us, and before the step-0 structure counts were seen (the step-0
job was still queued). Text: CC BY 4.0; code: Apache-2.0.

## 1. Why a step 1 before step 0's counts

DESCRIBE-PLAN.md (step 0) asks whether colonies share recording dates, so that a within-date test
is possible. Writing step 1 we saw a stronger design that works whatever step 0 finds: **train on
some days, test on another day, with no rat in both.** A classifier trained that way cannot use
anything specific to the test day (room, microphone placement, gain, temperature, background), and
cannot use the test rats' own voices. If it still names the colony, the colony signature is stable
across days. Step 1 therefore runs the leave-one-date-out test as the primary test and keeps the
within-date test from DESCRIBE-PLAN as a secondary one (it runs only if shared dates exist). This
is a change of plan made before any data were seen, recorded here; DESCRIBE-PLAN.md is unchanged.

The claim under test: Barker, Veviurko, Bennett, Hart, Mograby & Lewin (2021), Science 371:503
(doi 10.1126/science.abc6588): soft chirps carry a colony-specific dialect (a Random Forest names
the colony "with high accuracy" from 8 soft chirp features, with cross-validation folds grouped by
rat), and colony cohesion falls in queenless ("anarchy") periods. Data: Zenodo 5032855 (CC BY 4.0).

## 2. Features: the authors' own code

`molerat_step1.py` downloads the authors' repository at commit
d0442129bb8cfeff81e177ad09105ede9bf45579 (github.com/veviurko/Naked-mole-rats-paper; it has no
licence file, so nothing of it is copied into this repository; the job fetches it) and calls their
functions: each soft chirp is cut from its recording at the times in its trace name (22050 Hz),
high-passed at 3 kHz (`butter_highpass_filter`), its trace image is cleaned (`clear_trace`), and
`compute_all_features(traces, wavs, freq_mult)` gives the features with freq_mult = 22050/2/128, as
in their notebook. librosa 0.8.1 (theirs: 0.8.0). Traces are matched to recordings as in step 0
(the authors' key, recording name minus its last 3 characters). Any trace that fails is counted by
reason, never dropped silently.

Cleaning, as in their notebook: the 8 notebook features (slope, frequency, pitch, asymmetry,
wiener_entropy, zero_crossings, duration, height); drop non-finite values and outliers outside
slope 0.01-0.2, frequency 2000-8000 Hz, duration 0.05-10 s, height 300-6000 Hz.

Main set: clean soft chirps with colony, date and rat ids; queenless epochs removed (the notebook's
windows: baratheon 2020-04-04 to 2020-06-20; stark 2017-09-25 to 2017-11-25 and 2018-02-02 to
2018-06-30); rats (colony + ratids string, as the authors group) with at least 20 chirps, rats
above 1000 randomly cut to 1000; colonies with at least 2 such rats. Features are z-scored.
Seed 20261121 for everything random.

## 3. Tests

All main tests use linear discriminant analysis (LDA; equal priors, pooled covariance, ridge 1e-3
times the mean variance), because the null and power worlds need thousands of refits. The score is
the **balanced-accuracy excess**: mean recall over the true colonies tested, minus 1/(number of
colonies in training), averaged over folds weighted by test chirps (0 = chance).

- **A (published style).** Folds grouped by rat as the authors do: per colony, shuffle its rats and
  split them into min(5, rats) groups; fold f tests group f mod (groups) of every colony.
- **B (primary): leave one date out, rat-disjoint.** For each recording date d: test on every chirp
  recorded on d; train on chirps from other dates of rats that were NOT recorded on d.
- **Day-only null world.** From the real data we estimate (method of moments, z units) four
  covariance matrices: noise within a rat-date, rat effects within a colony-date, colony-date
  ("day") effects within a colony, and colony effects. 200 Gaussian data sets on the real
  structure (same colonies, rats, dates and chirp counts) with noise + rat + day effects and NO
  colony effect. p_A, p_B = (1 + #null >= observed) / 201.
- **Power world.** 100 data sets as above plus a colony effect with the estimated colony
  covariance (the colony difference the data show if all of it were dialect). power_B = share of
  them with p_B <= 0.05 against the null world.
- **W (secondary): within date.** Only (colony, date) cells with >= 50 chirps from >= 2 rats, on
  dates where >= 2 colonies have such a cell. Leave one rat out inside each date; null: shuffle
  colony labels among the rats of each date (999 shuffles). A within-date signal can still come
  from colony-specific recording sessions on that day; B cannot.
- **Random Forest reproduction.** The authors' recipe: balance the whole set by subsampling every
  colony to the smallest colony's count, rat-grouped folds as in A, RandomForest(256 trees), plain
  accuracy averaged over folds, against chance 1/K. Also a leave-one-date-out RF (100 trees,
  training capped at 10,000 chirps, 75-minute budget; folds done are reported).
- **Queen placebo.** On all clean chirps of baratheon and stark (queenless epochs included), dates
  with >= 30 chirps; at every boundary between consecutive dates with 3 dates on each side:
  shift = squared distance between the mean feature vectors of the 3 dates before and after;
  dispersion = log(trace of covariance after / before). The queen boundaries (baratheon
  2020-04-03/04 loss; stark 2017-09-24/25 loss, 2017-11-25/26 new queen, 2018-02-01/02 loss)
  are ranked among all boundaries of their colony: p_shift = share of boundaries with shift >= the
  queen boundary's; p_disp = share with dispersion >= it (loss) or <= it (new queen). Needs >= 10
  boundaries (the smallest possible p is 1/boundaries, so with fewer than 10 the 0.10 level could
  never be reached). p <= 0.10 counts as exceptional.

## 4. Verdicts (fixed now)

Primary, B: LODO_UNTESTABLE if fewer than 200 test chirps or fewer than 2 colonies tested; else
**COLONY_SIGNATURE_TRANSFERS_ACROSS_DAYS** if p_B <= 0.05; else **NO_TRANSFER_WITH_POWER** if
power_B >= 0.80; else **TRANSFER_UNDERPOWERED**.
A: EXCEEDS_DAY_ONLY_WORLD if p_A <= 0.05, else DAY_ONLY_WORLD_CAN_MATCH (day effects of the size
seen inside colonies can produce the published-style score).
W: WITHIN_DATE_SIGNAL / WITHIN_DATE_NO_SIGNAL (p_W <= 0.05) / WITHIN_DATE_UNTESTABLE.
RF: REPRODUCED_ABOVE_CHANCE if accuracy >= 1/K + 0.10, else NOT_REPRODUCED.
Queen: per boundary, EXCEPTIONAL / TYPICAL (shift) and PREDICTED_DIRECTION_EXCEPTIONAL /
NOT_EXCEPTIONAL (dispersion), or UNTESTABLE.

Readings. TRANSFERS: the strongest day-controlled evidence so far for a stable colony vocal
signature in naked mole-rats (it still cannot separate dialect from a colony's permanent home
acoustics). NO_TRANSFER_WITH_POWER: the published colony signal does not carry over between days
although the data could have shown it; the Science headline would then rest on recording-day
structure. Queen boundaries that are typical among ordinary date-to-date changes would mean the
public data cannot single out the queen effect.

## 5. Predictions (graded in the reading)

P1 RF reproduction REPRODUCED_ABOVE_CHANCE: 0.85. P2 A = DAY_ONLY_WORLD_CAN_MATCH: 0.35.
P3 B: TRANSFERS 0.45, NO_TRANSFER_WITH_POWER 0.20, UNDERPOWERED 0.15, UNTESTABLE 0.20.
P4 at least 2 of the 3 queen-loss boundaries PREDICTED_DIRECTION_EXCEPTIONAL: 0.25.
P5 at least 90% of soft chirp traces yield features: 0.70.
P6 the day covariance has a larger trace than the colony covariance: 0.50.

## 6. Limits known now

The Gaussian worlds are models: real features are not Gaussian, and rats recorded on several
dates make day and rat effects partly entangled in the estimates. A colony's housing and home
acoustics are constant across days and cannot be separated from dialect by any test on these data.
Our subset will not match the paper's exactly (their notebook windows and filters vary by
analysis); the RF reproduction shows how close it is. librosa 0.8.1 is not their 0.8.0.

## 7. Rules

The job (`gate1.py`) refuses to run until this file and the step-1 code on GitHub main equal its
copies. A published plan is never edited; changes go in a dated AMENDMENT file. If the job fails
for a technical reason, it is fixed and re-run under a new version name, with the fix described in
an amendment. Results are published whatever they say.

Correction before first publication (2026-11-21, 18:44Z run): the first draft of this plan, never
published, required >= 6 queen-placebo boundaries and its unit test expected EXCEPTIONAL at
p = 1/7. Since 1/7 > 0.10 that test cannot pass, and the draft never reached GitHub. The minimum
is now 10 boundaries, so the 0.10 level can be reached, and the test uses 16 toy dates
(11 boundaries, p = 1/11); nothing else changed and no data had been touched.
