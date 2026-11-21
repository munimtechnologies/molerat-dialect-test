# Copyright 2026 MV2, Munim, Inc. Licensed under the Apache License, Version 2.0 (Apache-2.0).
# Made with AI agents working in MV2's name.
"""Unit tests for step1_stats.py (numpy only; synthetic data)."""
import unittest

import numpy as np

import step1_stats as st


def queen_toy(n_dates, jump_at=6):
    """n_dates dates of 30 rows each; both features jump by 5 from date index jump_at on."""
    dates = ["2017-01-%02d" % (i + 1) for i in range(n_dates)]
    X, col, dt = [], [], []
    for i, d in enumerate(dates):
        for j in range(30):
            jit = (j % 7) / 10.0
            base = 5.0 if i >= jump_at else 0.0
            X.append([base + jit, base + jit])
            col.append("toy")
            dt.append(d)
    return np.array(X), np.array(col), np.array(dt)


class TestParsing(unittest.TestCase):
    def test_dates(self):
        self.assertEqual(st.parse_date("24-03-19"), "2019-03-24")
        self.assertEqual(st.parse_date("2017-09-25"), "2017-09-25")
        self.assertEqual(st.parse_date("1.2.18"), "2018-02-01")
        self.assertEqual(st.parse_date("03.11.2019"), "2019-11-03")
        self.assertEqual(st.parse_date("2019-03-24 00:00:00"), "2019-03-24")
        self.assertEqual(st.parse_date("bad"), "")
        self.assertEqual(st.parse_date("31-02-19"), "")
        self.assertEqual(st.parse_date(None), "")

    def test_anarchy(self):
        self.assertTrue(st.in_anarchy("stark", "2017-10-01"))
        self.assertTrue(st.in_anarchy("stark", "2017-09-25"))
        self.assertFalse(st.in_anarchy("stark", "2017-11-26"))
        self.assertTrue(st.in_anarchy("baratheon", "2020-06-20"))
        self.assertFalse(st.in_anarchy("martell", "2017-10-01"))

    def test_keep_mask(self):
        good = [0.05, 4000, 4000, 100, 1.0, 0.1, 0.2, 1000]
        steep = [0.5, 4000, 4000, 100, 1.0, 0.1, 0.2, 1000]
        nanp = [0.05, 4000, float("nan"), 100, 1.0, 0.1, 0.2, 1000]
        edge = [0.05, 4000, 4000, 100, 1.0, 0.1, 0.2, 300]
        low = [0.05, 1999, 4000, 100, 1.0, 0.1, 0.2, 1000]
        got = st.keep_mask(np.array([good, steep, nanp, edge, low]), st.FEATURES)
        self.assertEqual(got.tolist(), [True, False, False, True, False])

    def test_rat_filter(self):
        rat = np.array(["a"] * 25 + ["b"] * 5 + ["c"] * 1500)
        kept = st.rat_filter(rat, np.random.default_rng(0))
        self.assertEqual(int(np.sum(rat[kept] == "a")), 25)
        self.assertEqual(int(np.sum(rat[kept] == "b")), 0)
        self.assertEqual(int(np.sum(rat[kept] == "c")), 1000)
        self.assertTrue(np.all(np.diff(kept) > 0))


class TestLDA(unittest.TestCase):
    def test_balanced_excess(self):
        self.assertAlmostEqual(st.balanced_excess([0, 0, 1, 1], [0, 1, 1, 1], 2), 0.25)
        self.assertAlmostEqual(st.balanced_excess([2, 2], [2, 2], 4), 0.75)

    def test_lda_separates(self):
        x0 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        X = np.vstack([x0, x0 + 10.0])
        unit = np.arange(8)
        N, S, Q = st.unit_stats(X, unit, 8)
        self.assertEqual(N.tolist(), [1.0] * 8)
        self.assertTrue(np.allclose(S, X))
        self.assertTrue(np.allclose(Q[1], [1.0, 0.0, 0.0, 0.0]))
        y = np.array([0] * 4 + [1] * 4)
        G = np.zeros((8, 2))
        G[np.arange(8), y] = 1.0
        cls, A, b = st.lda_from_sums(G.T @ N, G.T @ S, G.T @ Q)
        self.assertEqual(cls.tolist(), [0, 1])
        test = np.array([[0.5, 0.2], [9.0, 11.0]])
        self.assertEqual(np.argmax(test @ A + b, axis=1).tolist(), [0, 1])

    def test_lda_needs_two_classes(self):
        self.assertIsNone(st.lda_from_sums(np.array([3.0, 0.0]), np.zeros((2, 2)), np.zeros((2, 4))))

    def test_units(self):
        unit, ur, ud = st.make_units([5, 5, 6, 5], [1, 1, 1, 2])
        self.assertEqual(unit.tolist(), [0, 0, 1, 2])
        self.assertEqual(ur.tolist(), [5, 6, 5])
        self.assertEqual(ud.tolist(), [1, 1, 2])
        order, starts, ends = st.unit_items(unit, 3)
        self.assertEqual(sorted(order[starts[0]:ends[0]].tolist()), [0, 1])
        self.assertEqual(order[starts[2]:ends[2]].tolist(), [3])


class TestFolds(unittest.TestCase):
    def test_rat_folds(self):
        unit_rat = np.array([0, 1, 2, 3, 4, 5, 0])
        unit_y = np.array([0, 0, 0, 1, 1, 1, 0])
        folds = st.rat_folds(unit_rat, unit_y, np.random.default_rng(1))
        self.assertEqual(len(folds), 5)
        seen = set()
        for tr, te in folds:
            self.assertFalse(np.any(tr & te))
            self.assertTrue(np.all(tr | te))
            test_rats = set(unit_rat[te].tolist())
            self.assertFalse(test_rats & set(unit_rat[tr].tolist()))
            self.assertEqual(len(set(unit_y[te].tolist())), 2)
            seen |= test_rats
        self.assertEqual(seen, {0, 1, 2, 3, 4, 5})

    def test_lodo_folds(self):
        folds = st.lodo_folds(np.array([0, 0, 1, 2]), np.array([0, 1, 1, 0]))
        self.assertEqual(len(folds), 2)
        self.assertEqual(folds[0][1].tolist(), [True, False, False, True])
        self.assertEqual(folds[0][0].tolist(), [False, False, True, False])
        self.assertEqual(folds[1][1].tolist(), [False, True, True, False])
        self.assertEqual(folds[1][0].tolist(), [False, False, False, True])

    def test_within_date_folds_and_shared(self):
        # units: (colony, date, n): c0 d0 30, c0 d0 30, c1 d0 30, c1 d0 30, c0 d1 60, c1 d1 60
        unit_y = np.array([0, 0, 1, 1, 0, 1])
        unit_date = np.array([0, 0, 0, 0, 1, 1])
        unit_rat = np.array([0, 1, 2, 3, 0, 2])
        N = np.array([30.0, 30.0, 30.0, 30.0, 60.0, 60.0])
        ok, shared = st.shared_cells(unit_y, unit_date, N)
        self.assertEqual(shared, [0])
        self.assertEqual(ok.tolist(), [True, True, True, True, False, False])
        folds = st.within_date_folds(unit_rat, unit_date, ok)
        self.assertEqual(len(folds), 4)
        self.assertEqual(folds[0][1].tolist(), [True, False, False, False, False, False])
        self.assertEqual(folds[0][0].tolist(), [False, True, True, True, False, False])

    def test_permute_within_date(self):
        unit_y = np.array([0, 0, 1, 1, 0, 1])
        unit_date = np.array([0, 0, 0, 0, 1, 1])
        ok = np.array([True, True, True, True, False, False])
        rng = np.random.default_rng(3)
        for _ in range(20):
            y = st.permute_within_date(unit_y, unit_date, ok, rng)
            self.assertEqual(sorted(y[:4].tolist()), [0, 0, 1, 1])
            self.assertEqual(y[4:].tolist(), [0, 1])

    def test_evaluate_separable(self):
        # 3 colonies x 2 rats x 2 dates, 5 chirps per unit; colony c sits at 10*c on both features
        rng = np.random.default_rng(4)
        rows, rat, date, y = [], [], [], []
        for c in range(3):
            for r in range(2):
                for d in range(2):
                    for _ in range(5):
                        rows.append(10.0 * c + rng.normal(0, 0.1, size=2))
                        rat.append(2 * c + r)
                        date.append(d)
                        y.append(c)
        X = np.array(rows)
        unit, ur, ud = st.make_units(rat, date)
        uy = np.zeros(ur.size, dtype=int)
        uy[unit] = np.array(y)
        N, S, Q = st.unit_stats(X, unit, ur.size)
        res = st.evaluate_folds(X, unit, uy, N, S, Q, st.rat_folds(ur, uy, rng), 3)
        self.assertAlmostEqual(res["excess"], 1.0 - 1.0 / 3.0)
        self.assertEqual(res["classes_tested"], 3)
        # every rat is on both dates, so leave-one-date-out (rat-disjoint) has no training data
        res_b = st.evaluate_folds(X, unit, uy, N, S, Q, st.lodo_folds(ur, ud), 3)
        self.assertEqual(res_b["n_test"], 0)
        self.assertTrue(np.isnan(res_b["excess"]))


class TestComponents(unittest.TestCase):
    def test_recovers_noise_rat_day(self):
        # 4 colonies x 30 dates x 5 rats (new rats on every date) x 20 chirps, p = 2
        rng = np.random.default_rng(5)
        comps = {"noise": np.eye(2) * 1.0, "rat": np.eye(2) * 0.5, "day": np.eye(2) * 0.8,
                 "colony": np.zeros((2, 2))}
        colony, rat, cell, date = [], [], [], []
        r = 0
        for c in range(4):
            for d in range(30):
                for _ in range(5):
                    colony += [c] * 20
                    rat += [r] * 20
                    cell += [c * 30 + d] * 20
                    date += [d] * 20
                    r += 1
        colony, rat, cell, date = map(np.array, (colony, rat, cell, date))
        struct = {"colony": colony, "rat": rat, "cell": cell, "K": 4, "n_rats": r, "n_cells": 120}
        X = st.simulate(struct, comps, rng)
        unit, ur, ud = st.make_units(rat, date)
        uy = np.zeros(ur.size, dtype=int)
        uy[unit] = colony
        est = st.components(X, unit, ur, ud, uy)
        self.assertTrue(0.9 < np.trace(est["noise"]) / 2.0 < 1.1)
        self.assertTrue(0.25 < np.trace(est["rat"]) / 2.0 < 0.75)
        self.assertTrue(0.4 < np.trace(est["day"]) / 2.0 < 1.2)


class TestQueenAndVerdicts(unittest.TestCase):
    def test_queen_jump(self):
        # 16 dates, k = 3: boundaries after date indexes 2..12 (11 of them); the jump sits
        # between indexes 5 and 6, so it has the single largest shift and p = 1/11 <= 0.10.
        X, col, dt = queen_toy(16)
        res = st.queen_placebo(X, col, dt,
                               boundaries=[("toy", "2017-01-06", "2017-01-07", "loss"),
                                           ("toy", "2017-01-02", "2017-01-03", "loss"),
                                           ("other", "2017-01-06", "2017-01-07", "loss")])
        self.assertEqual(res[0]["n_boundaries"], 11)
        self.assertEqual(res[0]["boundary_dates"], ["2017-01-06", "2017-01-07"])
        self.assertAlmostEqual(res[0]["p_shift"], 1.0 / 11.0)
        self.assertEqual(res[0]["verdict_shift"], "EXCEPTIONAL")
        self.assertEqual(res[1]["verdict_shift"], "UNTESTABLE")
        self.assertEqual(res[2]["verdict_shift"], "UNTESTABLE")

    def test_queen_needs_ten_boundaries(self):
        # 14 dates give 9 boundaries: below QUEEN_MIN_BOUNDARIES, so no verdict is given.
        X, col, dt = queen_toy(14)
        res = st.queen_placebo(X, col, dt,
                               boundaries=[("toy", "2017-01-06", "2017-01-07", "loss")])
        self.assertEqual(res[0]["n_boundaries"], 9)
        self.assertEqual(res[0]["verdict_shift"], "UNTESTABLE")
        self.assertEqual(st.QUEEN_MIN_BOUNDARIES, 10)

    def test_p_and_power(self):
        null = [0.0] * 19 + [float("nan")]
        self.assertAlmostEqual(st.p_upper(0.5, null), 1.0 / 20.0)
        self.assertAlmostEqual(st.p_upper(0.0, null), 1.0)
        self.assertTrue(np.isnan(st.p_upper(float("nan"), null)))
        self.assertAlmostEqual(st.power_upper([0.5, 0.5, 0.0, 0.0], null), 0.5)

    def test_verdicts(self):
        self.assertEqual(st.verdict_transfer(0.2, 1000, 3, 0.01, 0.9),
                         "COLONY_SIGNATURE_TRANSFERS_ACROSS_DAYS")
        self.assertEqual(st.verdict_transfer(0.0, 1000, 3, 0.4, 0.9), "NO_TRANSFER_WITH_POWER")
        self.assertEqual(st.verdict_transfer(0.0, 1000, 3, 0.4, 0.5), "TRANSFER_UNDERPOWERED")
        self.assertEqual(st.verdict_transfer(0.3, 150, 3, 0.01, 0.9), "LODO_UNTESTABLE")
        self.assertEqual(st.verdict_transfer(0.3, 1000, 1, 0.01, 0.9), "LODO_UNTESTABLE")
        self.assertEqual(st.verdict_transfer(float("nan"), 0, 0, float("nan"), 0.9),
                         "LODO_UNTESTABLE")
        self.assertEqual(st.verdict_day_world(0.3, 0.2), "DAY_ONLY_WORLD_CAN_MATCH")
        self.assertEqual(st.verdict_day_world(0.3, 0.005), "EXCEEDS_DAY_ONLY_WORLD")
        self.assertEqual(st.verdict_within(0, 0.1, 0.01), "WITHIN_DATE_UNTESTABLE")
        self.assertEqual(st.verdict_within(2, 0.1, 0.01), "WITHIN_DATE_SIGNAL")
        self.assertEqual(st.verdict_within(2, 0.0, 0.5), "WITHIN_DATE_NO_SIGNAL")
        self.assertEqual(st.verdict_reproduction(0.45, 5), "REPRODUCED_ABOVE_CHANCE")
        self.assertEqual(st.verdict_reproduction(0.25, 5), "NOT_REPRODUCED")
        self.assertEqual(st.verdict_reproduction(float("nan"), 5), "NOT_RUN")


if __name__ == "__main__":
    unittest.main()
