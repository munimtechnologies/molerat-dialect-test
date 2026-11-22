# READING-DESCRIBE: step 0 returned LABEL_UNKNOWN because of our matcher, not the data

Reading of `results/describe/` (job `molerat-describe-v1`, run 2026-11-22 13:24-13:47 UTC), the
structure-only step 0 registered in `DESCRIBE-PLAN.md`. MV2, Munim, Inc. (mv2@munimtech.com).
Made with AI agents; no human expert has checked it yet. Text CC BY 4.0.

## What happened

- `data.zip` from Zenodo 5032855 was read in full; its md5 (a2c5a9363e335d9849e635f78382edda)
  matches Zenodo's. It holds 25,098 files: 951 recording files, 24,145 trace images (7,823
  labelled `softchirp`, 16,322 `sound`), `recordings_metadata.csv` (641 rows, 5 colonies) and
  `rat_info.csv` (96 rats).
- **No trace was assigned to a recording**, so the registered decision is `LABEL_UNKNOWN`
  (soft chirps matched: 0).
- The cause is in our step-0 matcher. It tried prefix matching first (a recording key at the
  start of the trace name). Trace names begin with `freshtrace_<label>_<start>_<end>_`, so no key
  is a prefix. Its containment fallback was written with a safety cap (`FALLBACK_CAP = 20000`)
  and was skipped because 24,145 traces were unmatched. The authors' own loader
  (`load_data.py` in github.com/veviurko/Naked-mole-rats-paper) matches by containment:
  `rec_name[:-3] in trace_name`.
- Trace names carry colony, date and rat directly, for example
  `freshtrace_softchirp_0.0129_0.169389_baratheon_15-05-20_2097_0000006.png`.

## Predictions registered for step 0

| prediction | prior | outcome |
|---|---|---|
| WITHIN_DATE_FEASIBLE | 0.35 | false |
| WITHIN_DATE_PARTIAL | 0.35 | false |
| COLONY_DATE_CONFOUNDED | 0.25 | false |
| LABEL_UNKNOWN or NO_METADATA | 0.05 | true (for a technical reason, above) |
| at least 80% of soft chirp traces match by prefix | 0.60 | false (0%) |

The step-0 question (how many colonies share recording dates) is therefore still open.

## What it means for step 1

`STEP1-PREREG.md` and `molerat_step1.py` were registered before this result. Step 1 calls the
same matcher with containment switched on and no cap, which is the authors' rule, so this
failure should not repeat there. Its own counts of skipped traces (by cause) and of shared
dates will answer the step-0 question. If step 1 also fails to match, the fix goes into a dated
amendment and a new job version, as its registration requires.

## Files

`results/describe/DESCRIBE-MOLERAT.md`, `describe.json` and `cells.csv` are the job's output,
unchanged (counts only; no acoustic feature and no classifier were computed).
