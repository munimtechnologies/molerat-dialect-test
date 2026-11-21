# DESCRIBE-PLAN: are naked mole-rat colony "dialects" separable from recording days?

molerat-dialect-test, step 0 (structure only). MV2, Munim, Inc. Made with AI agents working in
MV2's name; no human expert has checked it yet. Registered 2026-11-21, before any of the data
was downloaded by us. Licence of this text: CC BY 4.0; code: Apache-2.0.

## 1. The question

Barker, Veviurko, Bennett, Hart, Mograby & Lewin (2021), *Cultural transmission of vocal dialect
in the naked mole-rat*, Science 371:503-507 (doi 10.1126/science.abc6588), report that the soft
chirp of naked mole-rats carries a colony-specific "dialect": a classifier trained on chirp
acoustics names a rat's colony far above chance; cross-fostered pups take on the adoptive
colony's dialect; and dialect cohesion falls after a queen dies and comes back with a new queen.
It is one of the most cited claims of vocal culture in a non-human mammal.

The authors' public classification code (github.com/veviurko/Naked-mole-rats-paper,
`src/data_analysis/softchirps_classification.py`) builds cross-validation folds grouped by rat
(`group_by='ratids'`), so no rat is in both training and test sets. Nothing in that code groups
by recording date. If each colony was recorded on its own days, then test rats share recording
days (room, microphone placement, gain, temperature, background) with training rats of the same
colony, and a day signature can pass for a colony signature. The queen-loss result compares time
epochs of one colony, so it is a comparison between recording periods by construction.

Our question: **does a colony signature survive when the classifier can only use differences
that are present within the same recording day, and is the queen-loss change larger than the
change across arbitrary (placebo) epoch boundaries of the same colony?** The wow, if dialects
survive: the strongest, day-controlled evidence yet for learned vocal culture in a rodent. If
they do not: a Science headline rests on recording-day structure, and the public data cannot
settle it. Either answer matters to the field.

Prior art (searched 2026-11-21): news coverage and the Science Perspective (Buffenstein 2021)
only; we found no reanalysis, replication or comment that tests day or session structure.

## 2. What step 0 computes

`describe_molerat.py` downloads Zenodo record 5032855 (`data.zip`, CC BY 4.0), checks its md5
against Zenodo's, and reads only `recordings_metadata.csv`, `rat_info.csv` and the FILE NAMES of
the trace images. From the names it reads the class label exactly as the authors' loader does
(first `_[a-z]+_` match) and assigns each trace to a recording (the authors' key `name[:-3]`;
prefix match first, containment as fallback). It reports, per colony and per recording date,
recordings, rats and soft chirps. It computes no acoustic feature and trains no classifier.

## 3. Decision rule (fixed now)

A colony "counts" on a date if it has at least 50 soft chirps from at least 2 distinct rats on
that date (experiments pooled). A "shared date" is a date on which at least 2 colonies count.

- **WITHIN_DATE_FEASIBLE**: at least 3 shared dates covering at least 4 distinct colonies.
  Step 1 then registers a within-date test: colony classification trained and tested only on
  shared dates, folds grouped by rat and stratified by date, with a date-preserving label
  permutation null, a planted-effect power world and a day-only confound world.
- **WITHIN_DATE_PARTIAL**: 1 or 2 shared dates, or fewer than 4 colonies on them. Step 1 runs
  the within-date test on what exists and reports its power before any verdict, plus the
  calibration below.
- **COLONY_DATE_CONFOUNDED**: no shared date. A within-date colony test is impossible in the
  public data. Step 1 then measures how large day effects are inside a colony (rat-grouped
  classification of recording date within each colony, against a date-preserving null) and
  asks whether the published rat-grouped colony accuracy is reachable by day effects of that
  size (a simulation calibrated on the within-colony day effects).
- **LABEL_UNKNOWN** / **NO_METADATA**: the files are not as the authors' code expects; step 1
  is replaced by a reading note.

Whatever the branch, step 1 also registers the queen-epoch placebo test (the change in chirp
features across the queen-loss boundary against the same statistic at every other boundary
between recording dates of that colony) if the metadata hold dated recordings on both sides.

Our priors for step 0 (graded in the step-0 reading): FEASIBLE 0.35, PARTIAL 0.35,
CONFOUNDED 0.25, LABEL_UNKNOWN or NO_METADATA 0.05. We also predict (0.6) that at least 80% of
soft chirp traces match a recording by prefix.

## 4. Rules

- The job refuses to run until this file and the code on GitHub main equal the job's copies.
- Nothing acoustic is computed in step 0. Step 1 is registered, with code and tests, before any
  feature or classifier is computed on the real data.
- A published plan is never edited; changes go in a dated AMENDMENT file.
- Counts depend on the authors' file naming; any trace that cannot be matched is reported.
