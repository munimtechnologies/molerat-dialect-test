# molerat-dialect-test

An open, pre-registered test of whether naked mole-rat colony "dialects" (Barker et al. 2021,
Science 371:503) survive when recording-day effects are controlled, using the authors' public
data (Zenodo 10.5281/zenodo.5032855, CC BY 4.0).

By MV2, Munim, Inc. (mv2@munimtech.com). Made with AI agents working in MV2's name; no human
expert has checked it yet. Code: Apache-2.0. Text and derived tables: CC BY 4.0.

## Status

| step | what | state |
|---|---|---|
| 0 | structure of the data: colonies x recording dates x rats x soft chirps (no acoustics) | registered 2026-11-21 ([DESCRIBE-PLAN.md](DESCRIBE-PLAN.md)); job queued |
| 1 | within-date colony test and queen-epoch placebo test | registered only after step 0 |

## Files

- `DESCRIBE-PLAN.md`: the question, prior art, what step 0 computes and the decision rule.
- `describe_molerat.py`: step 0 (stdlib only). `python describe_molerat.py --data-dir D --out-dir O`.
- `test_describe_molerat.py`: unit tests on synthetic files (`python -m unittest -v`).
- `gate.py`: the job computes nothing until these files on GitHub equal the job's copies.

## Credit

Data and the original analysis: Alison J. Barker, Grigorii Veviurko, Nigel C. Bennett, Daniel
W. Hart, Lina Mograby and Gary R. Lewin. Their code: github.com/veviurko/Naked-mole-rats-paper.
This project only re-examines their public data; any error here is ours.
