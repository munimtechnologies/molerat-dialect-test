# Copyright 2026 MV2, Munim, Inc. Licensed under the Apache License, Version 2.0 (Apache-2.0).
# Made with AI agents working in MV2's name.
"""Step 1 of molerat-dialect-test (STEP1-PREREG.md): do naked mole-rat colony "dialects"
(Barker et al. 2021, Science 371:503) carry over from one recording day to another?

1. Soft chirp features are computed with the AUTHORS' OWN functions (github.com/veviurko/
   Naked-mole-rats-paper at a pinned commit, fetched at run time; their repo has no licence, so
   nothing of it is copied here): 3 kHz high-pass, clear_trace, compute_all_features.
2. The authors' notebook outlier bounds and feature list are applied.
3. Tests (step1_stats.py): A published-style rat-grouped CV; B leave-one-date-out rat-disjoint
   CV (primary); a day-only null world and a planted-colony power world for both; W within-date
   leave-one-rat-out with a within-date label permutation null; queen-boundary placebo; and a
   Random Forest reproduction of the published-style accuracy.

Run:  python molerat_step1.py --data-dir /data/molerat --out-dir /out
"""
import argparse
import csv
import io
import json
import os
import sys
import tarfile
import time
import urllib.request
import zipfile
from collections import Counter

import numpy as np

sys.path.append("/tmp/mr")          # gate1.py puts describe_molerat.py here if /work is read-only
import describe_molerat as dm  # noqa: E402  (step-0 helper, fetched from main by gate1.py)
import step1_stats as st  # noqa: E402

AUTHORS_SHA = "d0442129bb8cfeff81e177ad09105ede9bf45579"
AUTHORS_TGZ = "https://codeload.github.com/veviurko/Naked-mole-rats-paper/tar.gz/" + AUTHORS_SHA
SR = 22050
FREQ_MULT = SR / 2 / 128            # the authors' notebook: Hz per trace-image row
HIGHPASS_HZ = 3000                  # the authors' loader: butter_highpass_filter(wav, 3000, 22050)
EXTRACT_VERSION = 1
ALL_FEATURES = st.FEATURES + ["amplitude"]
META_COLS = ["trace", "recording", "colony", "date_raw", "date", "ratids", "experiment"]
SEED = 20261121
RF_BUDGET_S = 75 * 60
RF_LODO_MAX_TRAIN = 10000


def nan():
    return float("nan")


def get_zip(data_dir):
    """Wait while another job of the same data volume is still downloading, then download (if
    needed) and check the md5 against Zenodo's."""
    path = os.path.join(data_dir, dm.ZIP_NAME)
    part = path + ".part"
    last, still = -1, 0
    while not os.path.exists(path) and os.path.exists(part) and still < 10:
        size = os.path.getsize(part)
        still = still + 1 if size == last else 0
        last = size
        print("data.zip.part exists (%d bytes): waiting for the other download" % size, flush=True)
        time.sleep(60)
    exp = dm.expected_md5()
    path = dm.download(data_dir)
    md5 = dm.md5_file(path)
    print("data md5 %s expected %s" % (md5, exp), flush=True)
    if exp and md5 != exp:
        raise SystemExit("md5 MISMATCH: nothing computed")
    return path, md5, exp


def fetch_authors(dest):
    os.makedirs(dest, exist_ok=True)
    roots = [d for d in os.listdir(dest) if d.startswith("Naked-mole-rats-paper")]
    if not roots:
        tgz = os.path.join(dest, "authors.tar.gz")
        for attempt in range(3):
            try:
                with urllib.request.urlopen(AUTHORS_TGZ, timeout=300) as r, open(tgz, "wb") as fh:
                    fh.write(r.read())
                break
            except Exception as exc:  # noqa: BLE001
                print("authors' code download attempt %d failed: %s" % (attempt + 1, exc), flush=True)
                time.sleep(30)
        else:
            raise SystemExit("could not download the authors' code; nothing computed")
        with tarfile.open(tgz, "r:gz") as tf:
            safe = [mb for mb in tf.getmembers()
                    if (mb.isfile() or mb.isdir()) and not mb.name.startswith("/")
                    and ".." not in mb.name.split("/")]
            tf.extractall(dest, members=safe)
        roots = [d for d in os.listdir(dest) if d.startswith("Naked-mole-rats-paper")]
    root = os.path.join(dest, sorted(roots)[0])
    sys.path.insert(0, root)
    from src.data_representations import softchirp_features as sf
    from src.data_representations.process_traces import clear_trace
    from src.data_representations.process_wavs import butter_highpass_filter
    return sf, clear_trace, butter_highpass_filter


def extract(zip_path, mods):
    """Soft chirp features for every soft chirp trace that maps to exactly one recording."""
    import matplotlib.image as mpimg
    sf, clear_trace, highpass = mods
    rows = []
    skipped = Counter()
    info = {}
    with zipfile.ZipFile(zip_path) as zf:
        names = [i.filename for i in zf.infolist() if not i.is_dir()]
        metas = sorted((n for n in names if dm.basename(n) == "recordings_metadata.csv"),
                       key=lambda n: (n.count("/"), len(n), n))
        if not metas:
            raise SystemExit("no recordings_metadata.csv in data.zip; nothing computed")
        meta_rows, _ = dm.read_csv_member(zf, metas[0])
        recs = {}
        for row in meta_rows:
            name = (row.get("name") or "").strip()
            if name and name not in recs:
                recs[name] = row
        keys = {}
        for name in recs:
            keys.setdefault(name[:-3], []).append(name)
        rec_member, dup = {}, set()
        for n in names:
            if dm.group_of(n, "recordings") is not None:
                b = dm.basename(n)
                if b in rec_member:
                    dup.add(b)
                rec_member[b] = n
        by_rec = {}
        n_sc = 0
        for n in names:
            if dm.group_of(n, "traces") is None:
                continue
            t = dm.basename(n)
            if dm.label_of(t) != dm.SC:
                continue
            n_sc += 1
            hit, _how = dm.match_trace(t, keys, allow_contains=True)
            if len(hit) != 1:
                skipped["no_unique_recording"] += 1
                continue
            by_rec.setdefault(hit[0], []).append((t, n))
        info["softchirp_traces"] = n_sc
        info["recordings_with_softchirps"] = len(by_rec)
        t0 = time.time()
        for k, rec_name in enumerate(sorted(by_rec)):
            items = sorted(by_rec[rec_name])
            member = rec_member.get(rec_name)
            if member is None or rec_name in dup:
                skipped["no_unique_recording_file"] += len(items)
                continue
            try:
                rec = np.load(io.BytesIO(zf.read(member)), allow_pickle=False)
            except Exception:  # noqa: BLE001
                skipped["recording_unreadable"] += len(items)
                continue
            row = recs[rec_name]
            colony = (row.get("colony") or "").strip().lower()
            date_raw = (row.get("date") or "").strip()
            meta = {"recording": rec_name, "colony": colony, "date_raw": date_raw,
                    "date": st.parse_date(date_raw), "ratids": (row.get("ratids") or "").strip(),
                    "experiment": (row.get("experiment") or "").strip()}
            for t, tn in items:
                tm = dm.TIMES_RE.findall(t)
                if not tm:
                    skipped["no_times"] += 1
                    continue
                a, b = tm[0].split("_")
                s, e = int(SR * float(a)), int(SR * float(b))
                try:
                    wav = highpass(rec[s:e], HIGHPASS_HZ, SR)
                    im = mpimg.imread(io.BytesIO(zf.read(tn)), format="png")
                    trace = clear_trace(im)
                    df, bad = sf.compute_all_features([trace], [wav], FREQ_MULT)
                except Exception as exc:  # noqa: BLE001
                    skipped["error_" + type(exc).__name__] += 1
                    continue
                if len(bad) or len(df) != 1:
                    skipped["features_failed"] += 1
                    continue
                out = dict(meta)
                out["trace"] = t
                for f in ALL_FEATURES:
                    out[f] = float(df.iloc[0][f])
                rows.append(out)
            if k % 100 == 0:
                print("recording %d/%d, %d chirps with features, %.0f s" % (
                    k, len(by_rec), len(rows), time.time() - t0), flush=True)
    info["skipped"] = dict(sorted(skipped.items()))
    info["with_features"] = len(rows)
    return rows, info


def write_features(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(META_COLS + ALL_FEATURES)
        for r in rows:
            w.writerow([r[c] for c in META_COLS] + [repr(r[f]) for f in ALL_FEATURES])


def read_features(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for f in ALL_FEATURES:
                r[f] = float(r[f])
            rows.append(r)
    return rows


def rf_parts(X, y, rat, unit, unit_rat, unit_date, K, rng):
    from sklearn.ensemble import RandomForestClassifier
    t0 = time.time()
    out = {}
    counts = np.bincount(y, minlength=K)
    m = int(counts[counts > 0].min())
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), m, replace=False)
                                  for c in range(K) if counts[c] > 0]))
    Xb, yb, rb = X[idx], y[idx], rat[idx]
    accs = []
    for tr, te in st.rat_folds(rb, yb, rng):
        if not tr.any() or not te.any():
            continue
        clf = RandomForestClassifier(n_estimators=256, n_jobs=2, random_state=SEED)
        clf.fit(Xb[tr], yb[tr])
        accs.append(float(np.mean(clf.predict(Xb[te]) == yb[te])))
    out["published_style_accuracy"] = float(np.mean(accs)) if accs else nan()
    out["published_style_fold_accuracies"] = accs
    out["published_style_chance"] = 1.0 / K
    out["published_style_n_per_colony"] = m
    folds = st.lodo_folds(unit_rat, unit_date)
    tot, wsum, done = 0.0, 0.0, 0
    for tr_u, te_u in folds:
        if time.time() - t0 > RF_BUDGET_S:
            break
        done += 1
        tr = np.flatnonzero(tr_u[unit])
        if tr.size > RF_LODO_MAX_TRAIN:
            tr = np.sort(rng.choice(tr, RF_LODO_MAX_TRAIN, replace=False))
        cls = np.unique(y[tr])
        if cls.size < 2:
            continue
        te = np.flatnonzero(te_u[unit])
        te = te[np.isin(y[te], cls)]
        if te.size == 0:
            continue
        clf = RandomForestClassifier(n_estimators=100, n_jobs=2, random_state=SEED)
        clf.fit(X[tr], y[tr])
        e = st.balanced_excess(y[te], clf.predict(X[te]), cls.size)
        tot += e * te.size
        wsum += te.size
    out["lodo_rf_excess"] = (tot / wsum) if wsum else nan()
    out["lodo_rf_n_test"] = int(wsum)
    out["lodo_rf_folds_done"] = done
    out["lodo_rf_folds_total"] = len(folds)
    out["seconds"] = round(time.time() - t0, 1)
    return out


def analyse(rows, rng, log=print):
    res = {"n_with_features": len(rows)}
    if not rows:
        res["verdict"] = "NO_FEATURES"
        return res
    F = np.array([[r[f] for f in st.FEATURES] for r in rows], dtype=float)
    colony = np.array([r["colony"] for r in rows])
    date = np.array([r["date"] for r in rows])
    ratids = np.array([r["ratids"] for r in rows])
    meta_ok = (colony != "") & (date != "") & (ratids != "")
    keep = st.keep_mask(F, st.FEATURES) & meta_ok
    res["n_dropped_outlier_or_nonfinite"] = int((~st.keep_mask(F, st.FEATURES)).sum())
    res["n_dropped_missing_meta"] = int((~meta_ok).sum())
    ci = np.flatnonzero(keep)
    res["n_clean"] = int(ci.size)
    res["queen"] = st.queen_placebo(st.zscore(F[ci]), colony[ci], date[ci]) if ci.size else []
    anarchy = np.array([st.in_anarchy(c, d) for c, d in zip(colony[ci], date[ci])], dtype=bool)
    res["n_anarchy_excluded"] = int(anarchy.sum())
    mi = ci[~anarchy]
    rat_key = np.array([c + "|" + r for c, r in zip(colony[mi], ratids[mi])])
    kept = st.rat_filter(rat_key, rng)
    mi, rat_key = mi[kept], rat_key[kept]
    col_rats = {}
    for c, rk in zip(colony[mi].tolist(), rat_key.tolist()):
        col_rats.setdefault(c, set()).add(rk)
    cols = sorted(c for c, s in col_rats.items() if len(s) >= 2)
    sel = np.isin(colony[mi], cols)
    mi, rat_key = mi[sel], rat_key[sel]
    K = len(cols)
    res["colonies"] = cols
    res["n_main"] = int(mi.size)
    if K < 2 or mi.size < 100:
        res["verdict"] = "TOO_FEW_DATA"
        return res
    X = st.zscore(F[mi])
    y = np.array([cols.index(c) for c in colony[mi]])
    rcode = {k: i for i, k in enumerate(sorted(set(rat_key.tolist())))}
    rat = np.array([rcode[k] for k in rat_key.tolist()])
    dlist = sorted(set(date[mi].tolist()))
    dcode = {d: i for i, d in enumerate(dlist)}
    dt = np.array([dcode[d] for d in date[mi].tolist()])
    unit, unit_rat, unit_date = st.make_units(rat, dt)
    unit_y = np.zeros(unit_rat.size, dtype=int)
    unit_y[unit] = y
    N, S, Q = st.unit_stats(X, unit, unit_rat.size)
    items = st.unit_items(unit, unit_rat.size)
    lodo = st.lodo_folds(unit_rat, unit_date)
    per = {}
    for c in range(K):
        mc = y == c
        per[cols[c]] = {"chirps": int(mc.sum()), "rats": int(np.unique(rat[mc]).size),
                        "dates": int(np.unique(dt[mc]).size),
                        "first_date": dlist[int(dt[mc].min())], "last_date": dlist[int(dt[mc].max())]}
    res["per_colony"] = per
    res["n_rats"] = len(rcode)
    res["n_dates"] = len(dlist)
    res["n_units"] = int(unit_rat.size)
    dates_per_rat = np.bincount(unit_rat)
    res["rats_on_2plus_dates"] = int(np.sum(dates_per_rat >= 2))

    comps = st.components(X, unit, unit_rat, unit_date, unit_y)
    res["components_trace"] = {k: float(np.trace(v)) for k, v in comps.items()}
    log("components (trace): %s" % json.dumps(res["components_trace"]))

    a, b = st.stat_AB(X, unit, unit_rat, unit_date, unit_y, K, rng, lodo, items)
    res["A_obs"] = a
    res["B_obs"] = b
    log("A_obs %s | B_obs %s" % (json.dumps(a), json.dumps(b)))

    cell_index = {}
    cell = np.array([cell_index.setdefault(key, len(cell_index))
                     for key in zip(y.tolist(), dt.tolist())])
    struct = {"colony": y, "rat": rat, "cell": cell, "K": K, "n_rats": len(rcode),
              "n_cells": len(cell_index)}
    t0 = time.time()
    null_a, null_b = [], []
    for i in range(st.R_NULL):
        Xs = st.simulate(struct, comps, rng, 0.0)
        sa, sb = st.stat_AB(Xs, unit, unit_rat, unit_date, unit_y, K, rng, lodo, items)
        null_a.append(sa["excess"])
        null_b.append(sb["excess"])
        if i % 50 == 0:
            log("null world %d/%d, %.0f s" % (i, st.R_NULL, time.time() - t0))
    alt_a, alt_b = [], []
    for i in range(st.R_POWER):
        Xs = st.simulate(struct, comps, rng, 1.0)
        sa, sb = st.stat_AB(Xs, unit, unit_rat, unit_date, unit_y, K, rng, lodo, items)
        alt_a.append(sa["excess"])
        alt_b.append(sb["excess"])
    res["null_A"] = null_a
    res["null_B"] = null_b
    res["power_world_A"] = alt_a
    res["power_world_B"] = alt_b
    res["p_A"] = st.p_upper(a["excess"], null_a)
    res["p_B"] = st.p_upper(b["excess"], null_b)
    res["power_A"] = st.power_upper(alt_a, null_a)
    res["power_B"] = st.power_upper(alt_b, null_b)
    log("p_A %.4f p_B %.4f power_A %.3f power_B %.3f (%.0f s)" % (
        res["p_A"], res["p_B"], res["power_A"], res["power_B"], time.time() - t0))

    ok, shared = st.shared_cells(unit_y, unit_date, N)
    res["shared_dates"] = [dlist[d] for d in shared]
    res["W_units"] = int(ok.sum())
    if shared:
        wf = st.within_date_folds(unit_rat, unit_date, ok)
        w = st.evaluate_folds(X, unit, unit_y, N, S, Q, wf, K, items)
        perm = [st.evaluate_folds(X, unit, st.permute_within_date(unit_y, unit_date, ok, rng),
                                  N, S, Q, wf, K, items)["excess"] for _ in range(st.B_PERM)]
        res["W_obs"] = w
        res["p_W"] = st.p_upper(w["excess"], perm)
        res["W_perm_mean"] = float(np.nanmean(perm))
    else:
        res["W_obs"] = {"excess": nan(), "n_test": 0, "n_folds": 0, "classes_tested": 0}
        res["p_W"] = nan()

    try:
        res["rf"] = rf_parts(X, y, rat, unit, unit_rat, unit_date, K, rng)
    except Exception as exc:  # noqa: BLE001
        res["rf"] = {"error": "%s: %s" % (type(exc).__name__, exc),
                     "published_style_accuracy": nan()}
    log("rf %s" % json.dumps({k: v for k, v in res["rf"].items()
                              if k != "published_style_fold_accuracies"}))

    res["verdict_reproduction"] = st.verdict_reproduction(res["rf"].get("published_style_accuracy"), K)
    res["verdict_day_world"] = st.verdict_day_world(a["excess"], res["p_A"])
    res["verdict_transfer"] = st.verdict_transfer(b["excess"], b["n_test"], b["classes_tested"],
                                                  res["p_B"], res["power_B"])
    res["verdict_within"] = st.verdict_within(len(shared), res["W_obs"]["excess"], res["p_W"])
    res["verdict"] = res["verdict_transfer"]
    return res


def fmt(v, nd=4):
    if isinstance(v, float):
        return "nan" if v != v else ("%." + str(nd) + "f") % v
    return str(v)


def render(res, info):
    L = ["# RESULTS-STEP1: do naked mole-rat colony dialects carry over between recording days?", "",
         "Rendered by molerat_step1.py from step1_results.json (pre-registration: STEP1-PREREG.md).",
         "Made with AI agents working in MV2's name; no human expert has checked it yet.", "",
         "## Verdicts", "",
         "- primary (B, leave-one-date-out, rat-disjoint): **%s**" % res.get("verdict_transfer"),
         "- published-style CV vs day-only world (A): **%s**" % res.get("verdict_day_world"),
         "- within-date test (W): **%s**" % res.get("verdict_within"),
         "- Random Forest reproduction: **%s**" % res.get("verdict_reproduction"), "",
         "## Numbers", ""]
    a = res.get("A_obs", {})
    b = res.get("B_obs", {})
    w = res.get("W_obs", {})
    rf = res.get("rf", {})
    L += ["| quantity | value |", "|---|---|",
          "| soft chirp traces | %s |" % info.get("softchirp_traces"),
          "| with features | %s |" % info.get("with_features"),
          "| clean (in bounds, with metadata) | %s |" % res.get("n_clean"),
          "| main set (no queenless epochs, rats >= 20 chirps, colonies >= 2 rats) | %s |" % res.get("n_main"),
          "| colonies / rats / dates / rat-date units | %d / %s / %s / %s |" % (
              len(res.get("colonies", [])), res.get("n_rats"), res.get("n_dates"), res.get("n_units")),
          "| rats recorded on 2+ dates | %s |" % res.get("rats_on_2plus_dates"),
          "| A: published-style LDA balanced-accuracy excess | %s (n %s) |" % (fmt(a.get("excess")), a.get("n_test")),
          "| A: day-only world mean / p_A / power_A | %s / %s / %s |" % (
              fmt(float(np.nanmean(res["null_A"])) if res.get("null_A") else nan()),
              fmt(res.get("p_A")), fmt(res.get("power_A"), 3)),
          "| B: leave-one-date-out LDA excess | %s (n %s, colonies %s) |" % (
              fmt(b.get("excess")), b.get("n_test"), b.get("classes_tested")),
          "| B: day-only world mean / p_B / power_B | %s / %s / %s |" % (
              fmt(float(np.nanmean(res["null_B"])) if res.get("null_B") else nan()),
              fmt(res.get("p_B")), fmt(res.get("power_B"), 3)),
          "| W: within-date LDA excess / p_W (shared dates %d) | %s / %s |" % (
              len(res.get("shared_dates", [])), fmt(w.get("excess")), fmt(res.get("p_W"))),
          "| RF published-style accuracy (chance) | %s (%s) |" % (
              fmt(rf.get("published_style_accuracy")), fmt(rf.get("published_style_chance"))),
          "| RF leave-one-date-out excess (folds done/total) | %s (%s/%s) |" % (
              fmt(rf.get("lodo_rf_excess")), rf.get("lodo_rf_folds_done"), rf.get("lodo_rf_folds_total")),
          "| variance traces noise / rat / day / colony | %s |" % " / ".join(
              fmt(res.get("components_trace", {}).get(k, nan()), 3) for k in ("noise", "rat", "day", "colony")),
          "", "## Per colony (main set)", "",
          "| colony | chirps | rats | dates | first | last |", "|---|---|---|---|---|---|"]
    for c, v in (res.get("per_colony") or {}).items():
        L.append("| %s | %d | %d | %d | %s | %s |" % (c, v["chirps"], v["rats"], v["dates"],
                                                     v["first_date"], v["last_date"]))
    L += ["", "## Queen boundaries (placebo against every other date boundary of the colony)", "",
          "| colony | kind | boundary | boundaries | p_shift | shift verdict | p_disp | dispersion verdict |",
          "|---|---|---|---|---|---|---|---|"]
    for q in res.get("queen", []):
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            q["colony"], q["kind"], " / ".join(q.get("boundary_dates", [q["before_end"], q["after_start"]])),
            q.get("n_boundaries"), fmt(q.get("p_shift", nan()), 3), q.get("verdict_shift"),
            fmt(q.get("p_disp", nan()), 3), q.get("verdict_disp")))
    L += ["", "Skipped traces: %s" % json.dumps(info.get("skipped")), ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    path, md5, exp = get_zip(a.data_dir)
    cache = os.path.join(a.data_dir, "step1-features-x%d-%s.csv" % (EXTRACT_VERSION, md5[:8]))
    info_path = cache[:-4] + ".info.json"
    if os.path.exists(cache) and os.path.exists(info_path):
        rows = read_features(cache)
        with open(info_path, encoding="utf-8") as fh:
            info = json.load(fh)
        print("features read from cache %s (%d rows)" % (cache, len(rows)), flush=True)
    else:
        mods = fetch_authors("/tmp/authors")
        rows, info = extract(path, mods)
        write_features(rows, cache)
        with open(info_path, "w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=1, sort_keys=True)
    import librosa
    info["librosa"] = librosa.__version__
    info["data_md5"] = md5
    info["authors_sha"] = AUTHORS_SHA
    write_features(rows, os.path.join(a.out_dir, "features.csv"))
    print("features: %s" % json.dumps(info), flush=True)
    rng = np.random.default_rng(SEED)
    res = analyse(rows, rng, log=lambda s: print(s, flush=True))
    res["info"] = info
    with open(os.path.join(a.out_dir, "step1_results.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    with open(os.path.join(a.out_dir, "RESULTS-STEP1.md"), "w", encoding="utf-8") as fh:
        fh.write(render(res, info))
    print(json.dumps({k: res.get(k) for k in ("verdict_transfer", "verdict_day_world",
                                              "verdict_within", "verdict_reproduction",
                                              "p_A", "p_B", "power_B", "p_W")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
