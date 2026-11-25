# RESULTS-STEP1: do naked mole-rat colony dialects carry over between recording days?

Rendered by molerat_step1.py from step1_results.json (pre-registration: STEP1-PREREG.md).
Made with AI agents working in MV2's name; no human expert has checked it yet.

## Verdicts

- primary (B, leave-one-date-out, rat-disjoint): **COLONY_SIGNATURE_TRANSFERS_ACROSS_DAYS**
- published-style CV vs day-only world (A): **EXCEEDS_DAY_ONLY_WORLD**
- within-date test (W): **WITHIN_DATE_SIGNAL**
- Random Forest reproduction: **REPRODUCED_ABOVE_CHANCE**

## Numbers

| quantity | value |
|---|---|
| soft chirp traces | 7823 |
| with features | 7726 |
| clean (in bounds, with metadata) | 7486 |
| main set (no queenless epochs, rats >= 20 chirps, colonies >= 2 rats) | 5816 |
| colonies / rats / dates / rat-date units | 4 / 63 / 9 / 204 |
| rats recorded on 2+ dates | 49 |
| A: published-style LDA balanced-accuracy excess | 0.4004 (n 5816) |
| A: day-only world mean / p_A / power_A | 0.0952 / 0.0050 / 0.960 |
| B: leave-one-date-out LDA excess | 0.1048 (n 4043, colonies 2) |
| B: day-only world mean / p_B / power_B | -0.0698 / 0.0100 / 0.800 |
| W: within-date LDA excess / p_W (shared dates 5) | 0.2944 / 0.0010 |
| RF published-style accuracy (chance) | 0.6647 (0.2500) |
| RF leave-one-date-out excess (folds done/total) | -0.0725 (9/9) |
| variance traces noise / rat / day / colony | 3.488 / 3.403 / 0.623 / 2.086 |

## Per colony (main set)

| colony | chirps | rats | dates | first | last |
|---|---|---|---|---|---|
| baratheon | 2866 | 16 | 6 | 2019-04-04 | 2019-11-08 |
| dothrakib | 391 | 14 | 1 | 2019-08-22 | 2019-08-22 |
| martell | 1027 | 15 | 3 | 2019-10-31 | 2019-11-08 |
| targaryen | 1532 | 18 | 4 | 2019-04-20 | 2019-06-21 |

## Queen boundaries (placebo against every other date boundary of the colony)

| colony | kind | boundary | boundaries | p_shift | shift verdict | p_disp | dispersion verdict |
|---|---|---|---|---|---|---|---|
| baratheon | loss | 2020-04-03 / 2020-04-04 | 4 | nan | UNTESTABLE | nan | UNTESTABLE |
| stark | loss | 2017-09-24 / 2017-09-25 | 0 | nan | UNTESTABLE | nan | UNTESTABLE |
| stark | new_queen | 2017-11-25 / 2017-11-26 | 0 | nan | UNTESTABLE | nan | UNTESTABLE |
| stark | loss | 2018-02-01 / 2018-02-02 | 0 | nan | UNTESTABLE | nan | UNTESTABLE |

Skipped traces: {"no_unique_recording_file": 97}
