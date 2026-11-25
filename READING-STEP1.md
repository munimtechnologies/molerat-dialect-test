# READING-STEP1: do naked mole-rat colony "dialects" carry over between recording days?

MV2, Munim, Inc. (mv2@munimtech.com), 2026-11-25. Made with AI agents working in MV2's name; no
human expert has checked it yet. Text CC BY 4.0. Every number below comes from
[results/step1/step1_results.json](results/step1/step1_results.json) and
[results/step1/RESULTS-STEP1.md](results/step1/RESULTS-STEP1.md), written unchanged by job
molerat-step1-v2 (run 2026-11-25 13:02-13:40 UTC; unit tests passed; gate1.py checked the plan and
code against GitHub main before computing). Plan: [STEP1-PREREG.md](STEP1-PREREG.md), public
2026-11-21.

Claim under test: Barker, Veviurko, Bennett, Hart, Mograby and Lewin (2021), Science 371:503,
doi:10.1126/science.abc6588: naked mole-rat soft chirps carry a colony-specific dialect. Data:
Zenodo 10.5281/zenodo.5032855 (CC BY 4.0; archive MD5 a2c5a9363e335d9849e635f78382edda). Features
were computed with the authors' own code (github.com/veviurko/Naked-mole-rats-paper at commit
d0442129), fetched by the job.

## Short answer

**The colony signature carries over to a recording day the classifier never saw, from rats it
never heard, but much more weakly than the published-style score suggests (excess over chance
0.10 against 0.40, on fewer colonies), and the authors' Random Forest does not carry over at
all.**

| test | balanced-accuracy excess over chance | day-only null world | p | reading |
|---|---|---|---|---|
| A: published style (folds grouped by rat) | 0.4004 (4 colonies, 5,816 chirps) | mean 0.0952 | 0.0050 | EXCEEDS_DAY_ONLY_WORLD |
| B (primary): leave one date out, no test rat in training | 0.1048 (2 colonies, 4,043 chirps) | mean -0.0698 | 0.0100 | COLONY_SIGNATURE_TRANSFERS_ACROSS_DAYS |
| W: within a date, leave one rat out | 0.2944 (3 colonies, 4,568 chirps) | shuffle mean -0.0014 | 0.0010 | WITHIN_DATE_SIGNAL |
| Random Forest, published recipe | accuracy 0.6647 against chance 0.25 | | | REPRODUCED_ABOVE_CHANCE |
| Random Forest, leave one date out (9 of 9 folds) | -0.0725 | | | reported, no verdict registered |

Power of test B in worlds with a colony effect of the estimated size: 0.80 (test A: 0.96).

## What it means

- **Reproduced.** With the authors' features and recipe, a Random Forest names the colony with
  accuracy 0.66 against a chance of 0.25 on our subset (4 colonies, 63 rats, 9 dates).
- **Part of the published-style score is day structure.** In simulated data with the real
  structure (same colonies, rats, dates and chirp counts) and recording-day and rat effects but
  **no** colony effect, the published-style score averages 0.095 above chance. The real score,
  0.400, is far above that (p 0.005).
- **The signature transfers across days, weakly.** Trained on other days and other rats, the
  linear classifier still beats chance on the held-out day by 0.105 (p 0.01 against the day-only
  world, where the mean is -0.070). Only 2 of the 4 colonies entered this test as test classes
  (dothrakib, for one, was recorded on a single date, so it can never be trained on another day).
- **The authors' classifier does not transfer.** The Random Forest, trained on other days, scores
  0.073 *below* chance on the held-out day. One possible reason (not tested): a flexible classifier
  fitted to the training days learns day-specific detail, while a linear one keeps only the shared
  colony direction. Either way, the published
  accuracy should therefore not be read as the accuracy of a stable dialect.
- **Day effects are smaller than colony effects** in the variance estimates (trace 0.62 against
  2.09; rat 3.40; within-rat noise 3.49).
- **The queen-loss result could not be tested.** Every queen boundary had fewer than the 10
  surrounding date boundaries the plan requires (4 for baratheon, 0 for stark), so all four are
  UNTESTABLE.

## Predictions (priors fixed in STEP1-PREREG.md)

| id | prediction | prior | outcome | Brier |
|---|---|---|---|---|
| P1 | Random Forest REPRODUCED_ABOVE_CHANCE | 0.85 | true | 0.023 |
| P2 | A = DAY_ONLY_WORLD_CAN_MATCH | 0.35 | false | 0.123 |
| P3 | B = TRANSFERS (0.45; NO_TRANSFER_WITH_POWER 0.20, UNDERPOWERED 0.15, UNTESTABLE 0.20) | 0.45 | true | 0.303 |
| P4 | at least 2 of 3 queen-loss boundaries exceptional | 0.25 | false | 0.063 |
| P5 | at least 90% of soft chirp traces yield features (7,726 of 7,823 = 98.8%) | 0.70 | true | 0.090 |
| P6 | day covariance trace larger than colony covariance trace | 0.50 | false | 0.250 |

Mean Brier score 0.142 (P3 scored on the probability given to the realised outcome; its
four-way Brier score is 0.405).

## Limits

- A colony's housing and home acoustics are constant across days; no test on these data can
  separate them from a learned dialect.
- The null and power worlds are Gaussian models built from estimated variance components; real
  features are not Gaussian, and day and rat effects are partly entangled because 49 rats were
  recorded on 2 or more dates.
- Our subset (4 colonies with at least 2 rats of 20+ chirps outside queenless periods) is smaller
  than the paper's 7 colonies; 97 soft chirp traces had no unique recording file and were skipped.
- Test B covers only 2 colonies.
- This was done with AI agents working in MV2's name and has not been checked by a human expert.
