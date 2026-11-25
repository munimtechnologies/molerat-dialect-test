# molerat-dialect-test

An open, pre-registered test of whether naked mole-rat colony "dialects" (Barker et al. 2021,
Science 371:503) survive when recording-day effects are controlled, using the authors' public
data (Zenodo 10.5281/zenodo.5032855, CC BY 4.0).

By MV2, Munim, Inc. (mv2@munimtech.com). Made with AI agents working in MV2's name; no human
expert has checked it yet. Code: Apache-2.0. Text and derived tables: CC BY 4.0.

## Result (2026-11-25): [READING-STEP1.md](READING-STEP1.md)

- **Transfers across days, weakly**: trained on other recording days and other rats, a linear
  classifier still names the colony on the held-out day (balanced-accuracy excess 0.105, p 0.01
  against a day-only null world; 2 colonies testable; power 0.80).
- The published-style score (0.400) far exceeds what day and rat effects alone produce (0.095).
- **The authors' Random Forest reproduces (accuracy 0.66, chance 0.25) but does not transfer**
  to an unseen day (excess -0.073).
- Queen-loss boundaries: untestable on the public dates.

## Status

| step | what | state |
|---|---|---|
| 0 | structure of the data: colonies x recording dates x rats x soft chirps (no acoustics) | registered 2026-11-21 ([DESCRIBE-PLAN.md](DESCRIBE-PLAN.md)); graded LABEL_UNKNOWN for a matching bug ([READING-DESCRIBE.md](READING-DESCRIBE.md)) |
| 1 | leave-one-date-out colony test (primary), day-only null and power worlds, within-date test, queen-boundary placebo, Random Forest reproduction | registered 2026-11-21 ([STEP1-PREREG.md](STEP1-PREREG.md)); **graded 2026-11-25** ([READING-STEP1.md](READING-STEP1.md), [results/step1/](results/step1/)) |

## Files

- `DESCRIBE-PLAN.md`: the question, prior art, what step 0 computes and the decision rule.
- `describe_molerat.py`: step 0 (stdlib only). `python describe_molerat.py --data-dir D --out-dir O`.
- `test_describe_molerat.py`: unit tests on synthetic files (`python -m unittest -v`).
- `gate.py`: the step-0 job computes nothing until these files on GitHub equal the job's copies.
- `STEP1-PREREG.md`: step 1's design, verdict rules and predictions.
- `molerat_step1.py`: step 1. Computes soft chirp features with the authors' own functions
  (their repository is fetched at a pinned commit at run time, not copied here), then runs the tests.
- `step1_stats.py`: step 1's statistics (numpy only); `test_step1_stats.py`: its unit tests.
- `gate1.py`: the step-1 job computes nothing until its plan and code on GitHub equal the job's copies.
- `results/step1/`: the step-1 job's output, unchanged.

Part of MV2's ledger of calibrated re-tests of animal-communication claims:
https://github.com/munimtechnologies/calibrated-animal-claims

## Credit

Data and the original analysis: Alison J. Barker, Grigorii Veviurko, Nigel C. Bennett, Daniel
W. Hart, Lina Mograby and Gary R. Lewin. Their code: github.com/veviurko/Naked-mole-rats-paper.
This project only re-examines their public data; any error here is ours.
