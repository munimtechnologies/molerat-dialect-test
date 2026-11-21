# Copyright 2026 MV2, Munim, Inc. Licensed under the Apache License, Version 2.0 (Apache-2.0).
# Made with AI agents working in MV2's name.
"""Unit tests for describe_molerat.py (stdlib only; synthetic files, no real data)."""
import json
import os
import shutil
import tempfile
import unittest
import zipfile

import describe_molerat as dm


META = ("name,path,colony,date,ratids,experiment\n"
        "r1.npy,/x/stark/recordings/,Stark,2019-01-01,11,single\n"
        "r2.npy,/x/stark/recordings/,stark,2019-01-01,12,single\n"
        "r3.npy,/x/baratheon/recordings/,Baratheon,2019-01-01,21,single\n"
        "r4.npy,/x/baratheon/recordings/,baratheon,2019-02-02,21;22,pair\n")


class Helpers(unittest.TestCase):
    def test_split_ids(self):
        self.assertEqual(dm.split_ids("1234"), ["1234"])
        self.assertEqual(dm.split_ids("12_34; 56"), ["12", "34", "56"])
        self.assertEqual(dm.split_ids("nan"), [])
        self.assertEqual(dm.split_ids(""), [])
        self.assertEqual(dm.split_ids(None), [])

    def test_label_of(self):
        self.assertEqual(dm.label_of("r1._1.23_1.45_softchirp_x.png"), "softchirp")
        self.assertEqual(dm.label_of("r1._1.23_1.45.png"), "")

    def test_group_of(self):
        self.assertEqual(dm.group_of("data/stark/traces/a.png", "traces"), "stark")
        self.assertIsNone(dm.group_of("data/stark/recordings/a.npy", "traces"))
        self.assertEqual(dm.group_of("data/stark/recordings/a.npy", "recordings"), "stark")

    def test_match_trace(self):
        keys = {"a.": ["a.npy"], "ab.": ["ab.npy"]}
        self.assertEqual(dm.match_trace("ab._1.0_2.0_softchirp_.png", keys), (["ab.npy"], "prefix"))
        self.assertEqual(dm.match_trace("a._1.0_2.0_noise_.png", keys), (["a.npy"], "prefix"))
        self.assertEqual(dm.match_trace("x_ab._1.0_2.0_softchirp_.png", keys),
                         (["ab.npy"], "contains"))
        self.assertEqual(dm.match_trace("x_ab._1.0_2.0_softchirp_.png", keys, allow_contains=False),
                         ([], "none"))
        self.assertEqual(dm.match_trace("zzz.png", keys), ([], "none"))

    def test_shared_dates_pools_experiments(self):
        cells = [
            {"colony": "a", "date": "d1", "n_sc": 60, "rats": {"1", "2"}},
            {"colony": "b", "date": "d1", "n_sc": 60, "rats": {"3", "4"}},
            {"colony": "c", "date": "d1", "n_sc": 40, "rats": {"5", "6"}},
            {"colony": "a", "date": "d2", "n_sc": 100, "rats": {"1"}},
            {"colony": "b", "date": "d2", "n_sc": 100, "rats": {"3", "4"}},
        ]
        self.assertEqual(dm.shared_dates(cells), {"d1": ["a", "b"]})
        self.assertEqual(dm.decide(dm.shared_dates(cells), 360), "WITHIN_DATE_PARTIAL")
        cells.append({"colony": "a", "date": "d2", "n_sc": 1, "rats": {"9"}})
        self.assertEqual(dm.shared_dates(cells), {"d1": ["a", "b"], "d2": ["a", "b"]})

    def test_decide_table(self):
        self.assertEqual(dm.decide({}, 0), "LABEL_UNKNOWN")
        self.assertEqual(dm.decide({}, 10), "COLONY_DATE_CONFOUNDED")
        self.assertEqual(dm.decide({"d1": ["a", "b"], "d2": ["c", "d"], "d3": ["a", "c"]}, 10),
                         "WITHIN_DATE_FEASIBLE")
        self.assertEqual(dm.decide({"d1": ["a", "b"], "d2": ["a", "b"], "d3": ["a", "b"]}, 10),
                         "WITHIN_DATE_PARTIAL")


class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.zip = os.path.join(self.tmp, "data.zip")
        with zipfile.ZipFile(self.zip, "w") as zf:
            zf.writestr("data/recordings_metadata.csv", META)
            zf.writestr("data/rat_info.csv", "id,colony\n11,Stark\n21,baratheon\n")
            for r, col in (("r1", "stark"), ("r2", "stark"), ("r3", "baratheon"),
                           ("r4", "baratheon")):
                zf.writestr("data/%s/recordings/%s.npy" % (col, r), b"0")
            zf.writestr("data/stark/traces/r1._1.0_2.0_softchirp_.png", b"0")
            zf.writestr("data/stark/traces/r1._3.0_4.0_softchirp_.png", b"0")
            zf.writestr("data/stark/traces/r2._1.0_2.0_noise_.png", b"0")
            zf.writestr("data/baratheon/traces/r3._1.0_2.0_softchirp_.png", b"0")
            zf.writestr("data/baratheon/traces/r4._1.0_2.0_softchirp_.png", b"0")
            zf.writestr("data/baratheon/traces/orphan_1.0_2.0_softchirp_.png", b"0")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_summary(self):
        s, cells = dm.summarise(self.zip)
        self.assertEqual(s["metadata_rows"], 4)
        self.assertEqual(s["metadata_names_with_recording_file"], 4)
        self.assertEqual(s["trace_files"], 6)
        self.assertEqual(s["trace_labels_all"], {"softchirp": 5, "noise": 1})
        self.assertEqual(s["trace_match"], {"none": 1, "prefix": 5})
        self.assertEqual(s["soft_chirps_matched"], 4)
        self.assertEqual(s["colonies_recordings"], {"baratheon": 2, "stark": 2})
        self.assertEqual(s["per_colony"]["stark"],
                         {"n_recs": 2, "n_sc": 2, "n_dates_with_sc": 1, "n_rats_with_sc": 1})
        self.assertEqual(s["per_colony"]["baratheon"],
                         {"n_recs": 2, "n_sc": 2, "n_dates_with_sc": 2, "n_rats_with_sc": 2})
        self.assertEqual(s["rats_on_2plus_dates"], 1)
        self.assertEqual(s["dates_by_number_of_colonies_with_sc"], {"1": 1, "2": 1})
        self.assertEqual(s["shared_dates"], {})
        self.assertEqual(s["decision"], "COLONY_DATE_CONFOUNDED")
        self.assertEqual(s["rat_info"]["colonies"], {"baratheon": 1, "stark": 1})
        self.assertEqual(len(cells), 3)

    def test_main_writes_outputs(self):
        out = os.path.join(self.tmp, "out")
        self.assertEqual(dm.main(["--zip", self.zip, "--out-dir", out]), 0)
        with open(os.path.join(out, "describe.json"), encoding="utf-8") as fh:
            s = json.load(fh)
        self.assertEqual(s["decision"], "COLONY_DATE_CONFOUNDED")
        self.assertTrue(os.path.exists(os.path.join(out, "cells.csv")))
        with open(os.path.join(out, "DESCRIBE-MOLERAT.md"), encoding="utf-8") as fh:
            self.assertIn("COLONY_DATE_CONFOUNDED", fh.read())

    def test_md5(self):
        p = os.path.join(self.tmp, "abc.bin")
        with open(p, "wb") as fh:
            fh.write(b"abc")
        self.assertEqual(dm.md5_file(p), "900150983cd24fb0d6963f7d28e17f72")


if __name__ == "__main__":
    unittest.main()
