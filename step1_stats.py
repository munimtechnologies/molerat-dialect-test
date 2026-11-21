# Copyright 2026 MV2, Munim, Inc. Licensed under the Apache License, Version 2.0 (Apache-2.0).
# Made with AI agents working in MV2's name.
"""Statistics for step 1 of molerat-dialect-test (numpy and stdlib only). See STEP1-PREREG.md.

Data model: every soft chirp belongs to one rat and one recording date. A "unit" is one rat on one
date. Colony labels live on units (unit_y), so a label permutation is a permutation of unit_y.
All classifiers here are linear discriminant analysis (LDA, equal priors, pooled covariance with a
small ridge) fitted from per-unit sums, so thousands of refits stay cheap.
"""
import datetime
import re

import numpy as np

FEATURES = ["slope", "frequency", "pitch", "asymmetry", "wiener_entropy", "zero_crossings",
            "duration", "height"]                      # the authors' notebook feature list
OUTLIER_BOUNDS = {"slope": (0.01, 0.2), "frequency": (2000.0, 8000.0),
                  "duration": (0.05, 10.0), "height": (300.0, 6000.0)}   # the authors' notebook
MIN_PER_RAT = 20
MAX_PER_RAT = 1000
RIDGE = 1e-3
N_FOLDS = 5
R_NULL = 200
R_POWER = 100
B_PERM = 999
ALPHA = 0.05
POWER_OK = 0.80
MIN_CELL_SC = 50
MIN_CELL_RATS = 2
MIN_LODO_TEST = 200
REPRO_MARGIN = 0.10
QUEEN_MIN_SC = 30
QUEEN_K = 3
QUEEN_P = 0.10
# The smallest possible p_shift is 1 / boundaries, so p <= QUEEN_P needs >= 1 / QUEEN_P boundaries.
QUEEN_MIN_BOUNDARIES = 10
# Queenless ("anarchy") epochs, from the authors' notebook (inclusive date windows).
ANARCHY = {"baratheon": [("2020-04-04", "2020-06-20")],
           "stark": [("2017-09-25", "2017-11-25"), ("2018-02-02", "2018-06-30")]}
# (colony, last date of the earlier epoch, first date of the later epoch, kind)
QUEEN_BOUNDARIES = [("baratheon", "2020-04-03", "2020-04-04", "loss"),
                    ("stark", "2017-09-24", "2017-09-25", "loss"),
                    ("stark", "2017-11-25", "2017-11-26", "new_queen"),
                    ("stark", "2018-02-01", "2018-02-02", "loss")]


def parse_date(s):
    """'dd-mm-yy' (the authors' format), 'dd.mm.yyyy' or 'yyyy-mm-dd' -> 'yyyy-mm-dd'; else ''."""
    s = (s or "").strip().split(" ")[0].split("T")[0]
    parts = [p for p in re.split(r"[-./_]+", s) if p]
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        return ""
    a, b, c = (int(p) for p in parts)
    if len(parts[0]) == 4:
        y, m, d = a, b, c
    else:
        d, m, y = a, b, c
        if y < 100:
            y += 2000
    try:
        return datetime.date(y, m, d).isoformat()
    except ValueError:
        return ""


def in_anarchy(colony, date_iso):
    for lo, hi in ANARCHY.get(colony, []):
        if lo <= date_iso <= hi:
            return True
    return False


def keep_mask(F, names):
    """True where every FEATURES column is finite and every bounded column is inside its bounds."""
    F = np.asarray(F, dtype=float)
    ok = np.ones(F.shape[0], dtype=bool)
    for j, nm in enumerate(names):
        col = F[:, j]
        if nm in FEATURES:
            ok &= np.isfinite(col)
        if nm in OUTLIER_BOUNDS:
            lo, hi = OUTLIER_BOUNDS[nm]
            ok &= (col >= lo) & (col <= hi)
    return ok


def rat_filter(rat, rng, min_n=MIN_PER_RAT, max_n=MAX_PER_RAT):
    """Sorted indexes kept: rats with >= min_n items; rats above max_n are subsampled to max_n."""
    rat = np.asarray(rat)
    keep = []
    for r in np.unique(rat):
        idx = np.flatnonzero(rat == r)
        if idx.size < min_n:
            continue
        if idx.size > max_n:
            idx = rng.choice(idx, size=max_n, replace=False)
        keep.append(idx)
    return np.sort(np.concatenate(keep)) if keep else np.zeros(0, dtype=int)


def zscore(X):
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def make_units(rat, date):
    """Integer codes of (rat, date) pairs. Returns unit (n,), unit_rat (U,), unit_date (U,)."""
    index = {}
    rat = np.asarray(rat).tolist()
    date = np.asarray(date).tolist()
    unit = np.empty(len(rat), dtype=int)
    for i, pair in enumerate(zip(rat, date)):
        unit[i] = index.setdefault(pair, len(index))
    unit_rat = np.empty(len(index), dtype=int)
    unit_date = np.empty(len(index), dtype=int)
    for (r, d), u in index.items():
        unit_rat[u] = r
        unit_date[u] = d
    return unit, unit_rat, unit_date


def unit_stats(X, unit, n_units):
    """Per-unit count N (U,), sum S (U,p) and sum of outer products Q (U,p*p)."""
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    N = np.bincount(unit, minlength=n_units).astype(float)
    S = np.stack([np.bincount(unit, weights=X[:, j], minlength=n_units) for j in range(p)], axis=1)
    XX = (X[:, :, None] * X[:, None, :]).reshape(n, p * p)
    Q = np.stack([np.bincount(unit, weights=XX[:, j], minlength=n_units) for j in range(p * p)],
                 axis=1)
    return N, S, Q


def unit_items(unit, n_units):
    """Item indexes grouped by unit: items of unit u are order[starts[u]:ends[u]]."""
    order = np.argsort(unit, kind="stable")
    counts = np.bincount(unit, minlength=n_units)
    ends = np.cumsum(counts)
    return order, ends - counts, ends


def lda_from_sums(Nc, Sc, Qc, ridge=RIDGE):
    """Class sums -> (classes, A, b) for classes with Nc > 0, or None if fewer than 2 classes.
    The score of class k for a row x is x @ A[:, k] + b[k] (equal priors)."""
    cls = np.flatnonzero(Nc > 0)
    if cls.size < 2:
        return None
    p = Sc.shape[1]
    n = Nc[cls]
    mu = Sc[cls] / n[:, None]
    W = Qc[cls].sum(axis=0).reshape(p, p) - (mu.T * n) @ mu
    cov = W / max(float(n.sum()) - cls.size, 1.0)
    cov = (cov + cov.T) / 2.0
    tr = float(np.trace(cov)) / p
    cov = cov + ridge * (tr if tr > 0 else 1.0) * np.eye(p)
    A = np.linalg.solve(cov, mu.T)
    b = -0.5 * np.sum(mu.T * A, axis=0)
    return cls, A, b


def balanced_excess(y_true, y_pred, n_classes):
    """Mean recall over the true classes present, minus 1/n_classes (0 = chance)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rec = [float(np.mean(y_pred[y_true == c] == c)) for c in np.unique(y_true)]
    return float(np.mean(rec)) - 1.0 / n_classes


def evaluate_folds(X, unit, unit_y, N, S, Q, folds, K, items=None):
    """Fit LDA on each fold's training units, predict its test units' items.
    folds: list of (train_units bool (U,), test_units bool (U,)). Test items whose class is absent
    from training are dropped. Returns the item-weighted mean balanced-accuracy excess."""
    if items is None:
        items = unit_items(unit, unit_y.size)
    order, starts, ends = items
    tot = 0.0
    tot_w = 0.0
    used = 0
    tested = set()
    for train_u, test_u in folds:
        idx = np.flatnonzero(train_u)
        if idx.size == 0:
            continue
        G = np.zeros((idx.size, K))
        G[np.arange(idx.size), unit_y[idx]] = 1.0
        fit = lda_from_sums(G.T @ N[idx], G.T @ S[idx], G.T @ Q[idx])
        if fit is None:
            continue
        cls, A, b = fit
        tu = np.flatnonzero(test_u)
        if tu.size == 0:
            continue
        ti = np.concatenate([order[starts[u]:ends[u]] for u in tu])
        yt = unit_y[unit[ti]]
        keep = np.isin(yt, cls)
        if not keep.any():
            continue
        ti = ti[keep]
        yt = yt[keep]
        pred = cls[np.argmax(X[ti] @ A + b, axis=1)]
        e = balanced_excess(yt, pred, cls.size)
        tot += e * ti.size
        tot_w += ti.size
        used += 1
        tested.update(np.unique(yt).tolist())
    return {"excess": (tot / tot_w) if tot_w > 0 else float("nan"), "n_test": int(tot_w),
            "n_folds": used, "classes_tested": len(tested)}


def rat_folds(unit_rat, unit_y, rng, n_folds=N_FOLDS):
    """The authors' scheme: per colony, shuffle its rats and split them into min(n_folds, rats)
    groups; test fold f takes group f % len(groups) of every colony. Returns (train, test) masks."""
    rat_col = {}
    for r, c in zip(np.asarray(unit_rat).tolist(), np.asarray(unit_y).tolist()):
        rat_col[r] = c
    groups = []
    for c in sorted(set(rat_col.values())):
        rs = np.array(sorted(r for r, cc in rat_col.items() if cc == c))
        rs = rs[rng.permutation(rs.size)]
        groups.append(np.array_split(rs, min(n_folds, rs.size)))
    folds = []
    for f in range(n_folds):
        test_rats = np.concatenate([g[f % len(g)] for g in groups])
        test = np.isin(unit_rat, test_rats)
        folds.append((~test, test))
    return folds


def lodo_folds(unit_rat, unit_date):
    """Leave one date out, rat-disjoint: test = all units on date d; train = units on other dates
    of rats NOT recorded on d."""
    folds = []
    for d in np.unique(unit_date):
        test = unit_date == d
        excl = np.isin(unit_rat, np.unique(unit_rat[test]))
        folds.append((~excl, test))
    return folds


def within_date_folds(unit_rat, unit_date, unit_ok):
    """Leave one rat out inside each date, using only units flagged in unit_ok."""
    folds = []
    for d in np.unique(unit_date[unit_ok]):
        on = unit_ok & (unit_date == d)
        for u in np.flatnonzero(on):
            test = np.zeros(on.size, dtype=bool)
            test[u] = True
            folds.append((on & ~test, test))
    return folds


def shared_cells(unit_y, unit_date, N, min_sc=MIN_CELL_SC, min_rats=MIN_CELL_RATS):
    """Units in (colony, date) cells with >= min_sc items from >= min_rats rats, on dates where at
    least 2 colonies have such a cell. Returns (bool (U,), sorted shared date codes)."""
    cells = {}
    for u, key in enumerate(zip(np.asarray(unit_y).tolist(), np.asarray(unit_date).tolist())):
        cells.setdefault(key, []).append(u)
    good = {k: v for k, v in cells.items() if N[v].sum() >= min_sc and len(v) >= min_rats}
    by_date = {}
    for (c, d) in good:
        by_date.setdefault(d, []).append(c)
    shared = sorted(d for d, cs in by_date.items() if len(cs) >= 2)
    ok = np.zeros(len(unit_y), dtype=bool)
    for (c, d), v in good.items():
        if d in shared:
            ok[v] = True
    return ok, shared


def permute_within_date(unit_y, unit_date, unit_ok, rng):
    """Shuffle colony labels among the flagged units of each date; other units keep theirs."""
    y = np.array(unit_y, copy=True)
    for d in np.unique(unit_date[unit_ok]):
        idx = np.flatnonzero(unit_ok & (unit_date == d))
        y[idx] = y[idx[rng.permutation(idx.size)]]
    return y


def psd(M):
    M = (M + M.T) / 2.0
    w, V = np.linalg.eigh(M)
    return (V * np.clip(w, 0.0, None)) @ V.T


def components(X, unit, unit_rat, unit_date, unit_y):
    """Method-of-moments covariances (p x p, clipped to PSD) of: noise (within a rat-date unit),
    rat (between rats inside one colony-date cell), day (between colony-date cells inside a colony)
    and colony (between colonies)."""
    X = np.asarray(X, dtype=float)
    p = X.shape[1]
    U = unit_rat.size
    N, S, Q = unit_stats(X, unit, U)
    m = S / N[:, None]
    W = Q.sum(axis=0).reshape(p, p) - (m.T * N) @ m
    Se = psd(W / max(float(N.sum()) - U, 1.0))
    cells = {}
    for u, key in enumerate(zip(unit_y.tolist(), unit_date.tolist())):
        cells.setdefault(key, []).append(u)
    acc = np.zeros((p, p))
    wsum = 0.0
    for v in cells.values():
        k = len(v)
        if k < 2:
            continue
        D = m[v] - m[v].mean(axis=0)
        acc += (D.T @ D / (k - 1) - Se * float(np.mean(1.0 / N[v]))) * (k - 1)
        wsum += k - 1
    Sr = psd(acc / wsum) if wsum > 0 else np.zeros((p, p))
    cell_mean, cell_var, cols = {}, {}, {}
    for key, v in cells.items():
        cell_mean[key] = m[v].mean(axis=0)
        cell_var[key] = (Sr + Se * float(np.mean(1.0 / N[v]))) / len(v)
        cols.setdefault(key[0], []).append(key)
    acc = np.zeros((p, p))
    wsum = 0.0
    for keys in cols.values():
        mm = len(keys)
        if mm < 2:
            continue
        cm = np.array([cell_mean[k] for k in keys])
        D = cm - cm.mean(axis=0)
        acc += (D.T @ D / (mm - 1) - np.mean([cell_var[k] for k in keys], axis=0)) * (mm - 1)
        wsum += mm - 1
    Sd = psd(acc / wsum) if wsum > 0 else np.zeros((p, p))
    gm, gv = [], []
    for keys in cols.values():
        cm = np.array([cell_mean[k] for k in keys])
        gm.append(cm.mean(axis=0))
        gv.append((Sd + np.mean([cell_var[k] for k in keys], axis=0)) / len(keys))
    if len(gm) >= 2:
        G = np.array(gm)
        D = G - G.mean(axis=0)
        Sc = psd(D.T @ D / (len(gm) - 1) - np.mean(gv, axis=0))
    else:
        Sc = np.zeros((p, p))
    return {"noise": Se, "rat": Sr, "day": Sd, "colony": Sc}


def sqrt_psd(M):
    w, V = np.linalg.eigh((M + M.T) / 2.0)
    return V * np.sqrt(np.clip(w, 0.0, None))


def simulate(struct, comps, rng, colony_scale=0.0):
    """Gaussian data on the real structure: noise + rat effect + colony-date ('day') effect, plus
    colony_scale times a colony effect. struct: colony, rat, cell (item codes), K, n_rats, n_cells."""
    p = comps["noise"].shape[0]

    def draw(M, k):
        return rng.standard_normal((k, p)) @ sqrt_psd(M).T

    X = draw(comps["noise"], struct["colony"].size)
    X += draw(comps["rat"], struct["n_rats"])[struct["rat"]]
    X += draw(comps["day"], struct["n_cells"])[struct["cell"]]
    if colony_scale > 0:
        X += colony_scale * draw(comps["colony"], struct["K"])[struct["colony"]]
    return X


def stat_AB(X, unit, unit_rat, unit_date, unit_y, K, rng, lodo=None, items=None):
    """A: published-style rat-grouped CV; B: leave-one-date-out, rat-disjoint CV (both LDA)."""
    N, S, Q = unit_stats(X, unit, unit_rat.size)
    if items is None:
        items = unit_items(unit, unit_rat.size)
    if lodo is None:
        lodo = lodo_folds(unit_rat, unit_date)
    a = evaluate_folds(X, unit, unit_y, N, S, Q, rat_folds(unit_rat, unit_y, rng), K, items)
    b = evaluate_folds(X, unit, unit_y, N, S, Q, lodo, K, items)
    return a, b


def p_upper(obs, null):
    """(1 + #null >= obs) / (1 + #null), finite null values only."""
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if not np.isfinite(obs):
        return float("nan")
    return float((1.0 + np.sum(null >= obs)) / (null.size + 1.0))


def power_upper(alt, null, alpha=ALPHA):
    alt = [v for v in alt if np.isfinite(v)]
    if not alt:
        return float("nan")
    return float(np.mean([p_upper(v, null) <= alpha for v in alt]))


def queen_placebo(X, colony, date, boundaries=None, min_sc=QUEEN_MIN_SC, k=QUEEN_K):
    """For each queen boundary: compare the change across it with the change across every other
    boundary between eligible dates of the same colony (k dates each side). shift = squared
    distance between the window means; disp = log(trace cov after / trace cov before)."""
    boundaries = QUEEN_BOUNDARIES if boundaries is None else boundaries
    X = np.asarray(X, dtype=float)
    colony = np.asarray(colony)
    date = np.asarray(date)
    out = []
    for col, end, start, kind in boundaries:
        res = {"colony": col, "before_end": end, "after_start": start, "kind": kind}
        m = colony == col
        ds, cs = np.unique(date[m], return_counts=True)
        dates = [d for d, c in zip(ds.tolist(), cs.tolist()) if c >= min_sc]
        stats = []
        for i in range(k - 1, len(dates) - k):
            xb = X[m & np.isin(date, dates[i - k + 1:i + 1])]
            xa = X[m & np.isin(date, dates[i + 1:i + 1 + k])]
            shift = float(np.sum((xa.mean(axis=0) - xb.mean(axis=0)) ** 2))
            tb = float(np.trace(np.atleast_2d(np.cov(xb.T))))
            ta = float(np.trace(np.atleast_2d(np.cov(xa.T))))
            disp = float(np.log(ta / tb)) if ta > 0 and tb > 0 else float("nan")
            stats.append((dates[i], dates[i + 1], shift, disp))
        res["n_eligible_dates"] = len(dates)
        res["n_boundaries"] = len(stats)
        q = [s for s in stats if s[0] <= end and s[1] >= start]
        if not q or len(stats) < QUEEN_MIN_BOUNDARIES:
            res["verdict_shift"] = "UNTESTABLE"
            res["verdict_disp"] = "UNTESTABLE"
            out.append(res)
            continue
        _, _, sq, dq = q[0]
        shifts = np.array([s[2] for s in stats])
        disps = np.array([s[3] for s in stats])
        res["boundary_dates"] = [q[0][0], q[0][1]]
        res["shift"] = sq
        res["disp"] = dq
        res["p_shift"] = float(np.mean(shifts >= sq))
        if kind == "loss":
            res["p_disp"] = float(np.mean(disps[np.isfinite(disps)] >= dq))
        else:
            res["p_disp"] = float(np.mean(disps[np.isfinite(disps)] <= dq))
        res["verdict_shift"] = "EXCEPTIONAL" if res["p_shift"] <= QUEEN_P else "TYPICAL"
        if not np.isfinite(dq):
            res["p_disp"] = float("nan")
            res["verdict_disp"] = "UNTESTABLE"
        else:
            res["verdict_disp"] = ("PREDICTED_DIRECTION_EXCEPTIONAL" if res["p_disp"] <= QUEEN_P
                                   else "NOT_EXCEPTIONAL")
        out.append(res)
    return out


def verdict_transfer(b_obs, n_test, n_classes_tested, p_b, power_b):
    if not np.isfinite(b_obs) or n_test < MIN_LODO_TEST or n_classes_tested < 2:
        return "LODO_UNTESTABLE"
    if p_b <= ALPHA:
        return "COLONY_SIGNATURE_TRANSFERS_ACROSS_DAYS"
    if np.isfinite(power_b) and power_b >= POWER_OK:
        return "NO_TRANSFER_WITH_POWER"
    return "TRANSFER_UNDERPOWERED"


def verdict_day_world(a_obs, p_a):
    if not np.isfinite(a_obs) or not np.isfinite(p_a):
        return "UNTESTABLE"
    return "EXCEEDS_DAY_ONLY_WORLD" if p_a <= ALPHA else "DAY_ONLY_WORLD_CAN_MATCH"


def verdict_within(n_shared, w_obs, p_w):
    if n_shared == 0 or not np.isfinite(w_obs) or not np.isfinite(p_w):
        return "WITHIN_DATE_UNTESTABLE"
    return "WITHIN_DATE_SIGNAL" if p_w <= ALPHA else "WITHIN_DATE_NO_SIGNAL"


def verdict_reproduction(acc, K):
    if acc is None or not np.isfinite(acc) or K < 2:
        return "NOT_RUN"
    return "REPRODUCED_ABOVE_CHANCE" if acc >= 1.0 / K + REPRO_MARGIN else "NOT_REPRODUCED"
