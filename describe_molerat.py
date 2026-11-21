# Copyright 2026 MV2, Munim, Inc. Licensed under the Apache License, Version 2.0 (Apache-2.0).
# Made with AI agents working in MV2's name.
"""Step 0 of molerat-dialect-test: describe the STRUCTURE of the public naked mole-rat soft chirp
data (Barker et al. 2021, Science 371:503; Zenodo 10.5281/zenodo.5032855, CC BY 4.0).

It counts recordings, rats, dates and labelled sounds per colony and per recording date. It
computes NO acoustic feature and NO classifier: nothing here can show whether colonies differ.
The decision rule in DESCRIBE-PLAN.md (section 3) is applied to these counts only.

Run:  python describe_molerat.py --data-dir /data/molerat --out-dir /out
      python describe_molerat.py --zip some_local.zip --out-dir out   (no download)
"""
import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import time
import urllib.request
import zipfile
from collections import Counter

API = "https://zenodo.org/api/records/5032855"
DATA_URL = "https://zenodo.org/api/records/5032855/files/data.zip/content"
ZIP_NAME = "data.zip"
TIMES_RE = re.compile(r"[0-9]+\.[0-9]+_[0-9]+\.[0-9]+")   # the authors' time pattern
LABEL_RE = re.compile(r"_[a-z]+_")                        # the authors' class-label pattern
SC = "softchirp"
MIN_SC_CELL = 50       # soft chirps a colony needs on one date to count on that date
MIN_RATS_CELL = 2      # distinct rats a colony needs on one date to count on that date
MIN_SHARED_DATES = 3   # shared dates needed for WITHIN_DATE_FEASIBLE
MIN_SHARED_COLONIES = 4  # distinct colonies across shared dates needed for WITHIN_DATE_FEASIBLE
FALLBACK_CAP = 20000   # containment matching is skipped if more traces than this lack a prefix


def basename(path):
    return path.rstrip("/").split("/")[-1]


def group_of(path, folder):
    """Name of the directory just above `folder` in a zip path, or None if `folder` is absent."""
    parts = path.split("/")[:-1]
    if folder not in parts:
        return None
    idx = parts.index(folder)
    return parts[idx - 1] if idx >= 1 else ""


def split_ids(raw):
    raw = (raw or "").strip()
    out = []
    for tok in re.split(r"[^0-9A-Za-z#\-]+", raw):
        if tok and tok.lower() not in ("nan", "unknown", "none", "na"):
            out.append(tok)
    return out


def label_of(trace_name):
    """Class label exactly as the authors' loader reads it: first '_[a-z]+_' match."""
    m = LABEL_RE.findall(trace_name)
    return m[0].strip("_") if m else ""


def match_trace(trace_name, keys, allow_contains=True):
    """Recording names whose authors' key (name[:-3]) the trace name starts with (longest first),
    else contains. Returns (list of recording names, how)."""
    for length in range(len(trace_name), 0, -1):
        hit = keys.get(trace_name[:length])
        if hit:
            return sorted(hit), "prefix"
    if not allow_contains:
        return [], "none"
    hits = [k for k in keys if k and k in trace_name]
    if not hits:
        return [], "none"
    best = max(hits, key=len)
    return sorted(keys[best]), ("contains" if len(hits) == 1 else "multi")


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 23)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def expected_md5():
    try:
        with urllib.request.urlopen(API, timeout=120) as r:
            rec = json.loads(r.read().decode("utf-8"))
        for f in rec.get("files", []):
            if f.get("key") == ZIP_NAME:
                ck = f.get("checksum", "")
                return ck.split(":", 1)[1] if ck.startswith("md5:") else ck
    except Exception as exc:  # noqa: BLE001
        print("could not read expected md5: %s" % exc, flush=True)
    return None


def download(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, ZIP_NAME)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    part = path + ".part"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(DATA_URL, timeout=300) as r, open(part, "wb") as fh:
                while True:
                    chunk = r.read(1 << 23)
                    if not chunk:
                        break
                    fh.write(chunk)
            os.replace(part, path)
            return path
        except Exception as exc:  # noqa: BLE001
            print("download attempt %d failed: %s" % (attempt + 1, exc), flush=True)
            time.sleep(60)
    raise SystemExit("download failed 3 times; nothing described")


def read_csv_member(zf, name):
    with zf.open(name) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
        reader = csv.DictReader(text)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def shared_dates(cells, min_sc=MIN_SC_CELL, min_rats=MIN_RATS_CELL):
    """cells: dicts with colony, date, n_sc, rats (set). Experiments are pooled per colony-date.
    Returns {date: sorted colonies meeting both thresholds} for dates with >= 2 such colonies."""
    agg = {}
    for c in cells:
        key = (c["date"], c["colony"])
        if key not in agg:
            agg[key] = [0, set()]
        agg[key][0] += c["n_sc"]
        agg[key][1] |= set(c["rats"])
    by_date = {}
    for (date, colony), (n_sc, rats) in agg.items():
        if n_sc >= min_sc and len(rats) >= min_rats:
            by_date.setdefault(date, []).append(colony)
    return {d: sorted(cols) for d, cols in by_date.items() if len(cols) >= 2}


def decide(shared, total_sc):
    if total_sc == 0:
        return "LABEL_UNKNOWN"
    cols = set()
    for v in shared.values():
        cols |= set(v)
    if len(shared) >= MIN_SHARED_DATES and len(cols) >= MIN_SHARED_COLONIES:
        return "WITHIN_DATE_FEASIBLE"
    if len(shared) >= 1:
        return "WITHIN_DATE_PARTIAL"
    return "COLONY_DATE_CONFOUNDED"


def summarise(zip_path, max_list=200):
    out = {}
    with zipfile.ZipFile(zip_path) as zf:
        names = [i.filename for i in zf.infolist() if not i.is_dir()]
        out["zip_files"] = len(names)
        layout = Counter()
        for n in names:
            parts = n.split("/")
            head = "/".join(parts[:-1][:3]) or "."
            layout[head + " *" + os.path.splitext(parts[-1])[1].lower()] += 1
        out["layout"] = dict(sorted(layout.items()))
        tables = sorted(n for n in names
                        if n.lower().endswith((".csv", ".tsv", ".json", ".xlsx", ".xls")))
        out["table_files_n"] = len(tables)
        out["table_files"] = tables[:max_list]
        metas = sorted((n for n in names if basename(n) == "recordings_metadata.csv"),
                       key=lambda n: (n.count("/"), len(n), n))
        out["metadata_files"] = metas
        infos = sorted((n for n in names if basename(n) == "rat_info.csv"),
                       key=lambda n: (n.count("/"), len(n), n))
        if infos:
            rows, cols = read_csv_member(zf, infos[0])
            out["rat_info"] = {"file": infos[0], "columns": cols, "rows": len(rows),
                               "colonies": dict(sorted(Counter(
                                   (r.get("colony") or "").strip().lower() for r in rows).items()))}
        if not metas:
            out["decision"] = "NO_METADATA"
            return out, []
        rows, cols = read_csv_member(zf, metas[0])
        out["metadata_columns"] = cols
        out["metadata_rows"] = len(rows)
        recs, dup = {}, 0
        for row in rows:
            name = (row.get("name") or "").strip()
            if not name:
                continue
            if name in recs:
                dup += 1
                continue
            recs[name] = {"colony": (row.get("colony") or "").strip().lower(),
                          "date": (row.get("date") or "").strip(),
                          "ratids": (row.get("ratids") or "").strip(),
                          "experiment": (row.get("experiment") or "").strip(),
                          "labels": Counter()}
        out["metadata_duplicate_names"] = dup
        keys = {}
        for name in recs:
            keys.setdefault(name[:-3], []).append(name)
        rec_files = set(basename(n) for n in names if group_of(n, "recordings") is not None)
        out["metadata_names_with_recording_file"] = sum(1 for n in recs if n in rec_files)
        out["recording_files_in_zip"] = len(rec_files)
        traces = [basename(n) for n in names if group_of(n, "traces") is not None]
        out["trace_files"] = len(traces)
        labels_all = Counter(label_of(t) for t in traces)
        out["trace_labels_all"] = dict(labels_all.most_common())
        out["traces_with_times"] = sum(1 for t in traces if TIMES_RE.search(t))
        how = Counter()
        unmatched = []
        for t in traces:
            hit, h = match_trace(t, keys, allow_contains=False)
            if hit:
                recs[hit[0]]["labels"][label_of(t)] += 1
                how[h] += 1
            else:
                unmatched.append(t)
        if len(unmatched) <= FALLBACK_CAP:
            for t in unmatched:
                hit, h = match_trace(t, keys, allow_contains=True)
                how[h] += 1
                if hit:
                    recs[hit[0]]["labels"][label_of(t)] += 1
            out["contains_fallback"] = "run"
        else:
            how["none"] += len(unmatched)
            out["contains_fallback"] = "skipped (%d unmatched by prefix)" % len(unmatched)
        out["trace_match"] = dict(sorted(how.items()))
        out["unmatched_examples"] = unmatched[:10]
        out["experiments"] = dict(Counter(r["experiment"] for r in recs.values()).most_common())
        out["colonies_recordings"] = dict(sorted(Counter(r["colony"] for r in recs.values()).items()))
        out["date_formats"] = dict(Counter(re.sub(r"[0-9]", "9", r["date"])
                                           for r in recs.values()).most_common())
        out["date_examples"] = sorted(set(r["date"] for r in recs.values()))[:10]
        out["ratids_examples"] = sorted(set(r["ratids"] for r in recs.values()))[:10]
        cellmap = {}
        rat_dates = {}
        for r in recs.values():
            key = (r["colony"], r["date"], r["experiment"])
            c = cellmap.setdefault(key, {"colony": key[0], "date": key[1], "experiment": key[2],
                                         "n_recs": 0, "rats": set(), "n_traces": 0, "n_sc": 0})
            c["n_recs"] += 1
            n_sc = r["labels"].get(SC, 0)
            c["n_sc"] += n_sc
            c["n_traces"] += sum(r["labels"].values())
            if n_sc > 0:
                for rat in split_ids(r["ratids"]):
                    c["rats"].add(rat)
                    rat_dates.setdefault((r["colony"], rat), set()).add(r["date"])
        cells = [cellmap[k] for k in sorted(cellmap)]
        total_sc = sum(c["n_sc"] for c in cells)
        out["soft_chirps_matched"] = total_sc
        per_col = {}
        for c in cells:
            p = per_col.setdefault(c["colony"], {"n_recs": 0, "n_sc": 0, "dates": set(),
                                                 "rats": set()})
            p["n_recs"] += c["n_recs"]
            p["n_sc"] += c["n_sc"]
            if c["n_sc"] > 0:
                p["dates"].add(c["date"])
            p["rats"] |= c["rats"]
        out["per_colony"] = {k: {"n_recs": v["n_recs"], "n_sc": v["n_sc"],
                                 "n_dates_with_sc": len(v["dates"]), "n_rats_with_sc": len(v["rats"])}
                             for k, v in sorted(per_col.items())}
        out["rats_on_2plus_dates"] = sum(1 for v in rat_dates.values() if len(v) >= 2)
        out["rats_with_sc"] = len(rat_dates)
        shared = shared_dates(cells)
        out["shared_dates"] = dict(sorted(shared.items()))
        sc_on_shared = sum(c["n_sc"] for c in cells
                           if c["date"] in shared and c["colony"] in shared[c["date"]])
        out["soft_chirps_on_shared_dates"] = sc_on_shared
        dates_by_n = Counter()
        date_cols = {}
        for c in cells:
            if c["n_sc"] > 0:
                date_cols.setdefault(c["date"], set()).add(c["colony"])
        for d, cs in date_cols.items():
            dates_by_n[len(cs)] += 1
        out["dates_by_number_of_colonies_with_sc"] = {str(k): v for k, v in sorted(dates_by_n.items())}
        out["thresholds"] = {"MIN_SC_CELL": MIN_SC_CELL, "MIN_RATS_CELL": MIN_RATS_CELL,
                             "MIN_SHARED_DATES": MIN_SHARED_DATES,
                             "MIN_SHARED_COLONIES": MIN_SHARED_COLONIES}
        out["decision"] = decide(shared, total_sc)
    return out, cells


def write_cells(cells, path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["colony", "date", "experiment", "n_recs", "n_rats_with_sc", "n_traces",
                    "n_softchirp", "rat_ids"])
        for c in cells:
            w.writerow([c["colony"], c["date"], c["experiment"], c["n_recs"], len(c["rats"]),
                        c["n_traces"], c["n_sc"], " ".join(sorted(c["rats"]))])


def render(summary):
    lines = ["# DESCRIBE-MOLERAT: structure of the public naked mole-rat soft chirp data", "",
             "Rendered by describe_molerat.py from describe.json. Counts only: no acoustic feature",
             "and no classifier was computed.", "",
             "- decision: **%s**" % summary.get("decision"),
             "- data md5: %s (Zenodo says %s)" % (summary.get("data_md5"), summary.get("expected_md5")),
             "- zip files: %s; recording files: %s; trace files: %s" % (
                 summary.get("zip_files"), summary.get("recording_files_in_zip"),
                 summary.get("trace_files")),
             "- metadata rows: %s (duplicate names %s); with a recording file: %s" % (
                 summary.get("metadata_rows"), summary.get("metadata_duplicate_names"),
                 summary.get("metadata_names_with_recording_file")),
             "- trace matching: %s" % json.dumps(summary.get("trace_match")),
             "- soft chirps matched to a recording: %s; on shared dates: %s" % (
                 summary.get("soft_chirps_matched"), summary.get("soft_chirps_on_shared_dates")),
             "- rats with soft chirps: %s; recorded on 2+ dates: %s" % (
                 summary.get("rats_with_sc"), summary.get("rats_on_2plus_dates")),
             "- dates by number of colonies with soft chirps: %s" % json.dumps(
                 summary.get("dates_by_number_of_colonies_with_sc")),
             "- shared dates (>= %d soft chirps from >= %d rats for each of >= 2 colonies): %s" % (
                 MIN_SC_CELL, MIN_RATS_CELL, json.dumps(summary.get("shared_dates"))),
             "", "## Per colony", "", "| colony | recordings | soft chirps | dates | rats |",
             "|---|---|---|---|---|"]
    for k, v in (summary.get("per_colony") or {}).items():
        lines.append("| %s | %d | %d | %d | %d |" % (k, v["n_recs"], v["n_sc"], v["n_dates_with_sc"],
                                                     v["n_rats_with_sc"]))
    lines += ["", "Full colony x date x experiment table: cells.csv. Everything else: describe.json.", ""]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--zip", default=None, help="use this local zip instead of downloading")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    if a.zip:
        path, exp = a.zip, None
    else:
        if not a.data_dir:
            ap.error("--data-dir or --zip is required")
        exp = expected_md5()
        path = download(a.data_dir)
    md5 = md5_file(path)
    print("data %s md5 %s bytes %d expected %s" % (ZIP_NAME, md5, os.path.getsize(path), exp),
          flush=True)
    if exp and exp != md5:
        print("md5 MISMATCH: the downloaded file is not Zenodo's file; nothing described", flush=True)
        return 2
    summary, cells = summarise(path)
    summary["data_md5"] = md5
    summary["expected_md5"] = exp
    with open(os.path.join(a.out_dir, "describe.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
    write_cells(cells, os.path.join(a.out_dir, "cells.csv"))
    with open(os.path.join(a.out_dir, "DESCRIBE-MOLERAT.md"), "w", encoding="utf-8") as fh:
        fh.write(render(summary))
    print(json.dumps({"decision": summary.get("decision"),
                      "soft_chirps_matched": summary.get("soft_chirps_matched"),
                      "shared_dates": len(summary.get("shared_dates") or {}),
                      "trace_match": summary.get("trace_match")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
