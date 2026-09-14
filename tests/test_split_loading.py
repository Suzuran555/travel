"""Regression checks for split isolation and oracle-field stripping.

Run with: python tests/test_split_loading.py
All fixtures live in a temporary project root; actual benchmark data is untouched.
"""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chinatravel.data import load_datasets


ORACLE_FIELDS = ("hard_logic", "hard_logic_py", "hard_logic_nl", "hard_logic_py_nl")


class SplitLoadingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="tpc-split-loading-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data_root = self.root / "chinatravel" / "data"
        self.split_root = self.root / "chinatravel" / "evaluation" / "default_splits"
        self.split_root.mkdir(parents=True)
        self.root_patch = patch.object(load_datasets, "project_root_path", str(self.root))
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def write_split(self, name, uids=("shared",)):
        (self.split_root / f"{name}.txt").write_text("\n".join(uids) + "\n")

    def write_query(self, directory, source, *, lang="en", oracle=False, uid="shared"):
        base = self.data_root / "en" if lang == "en" else self.data_root
        folder = base / directory
        folder.mkdir(parents=True, exist_ok=True)
        query = {"uid": uid, "source": source, "nature_language": "Plan a trip."}
        if oracle:
            query.update({field: ["reference constraint"] for field in ORACLE_FIELDS})
        (folder / f"{uid}.json").write_text(json.dumps(query))

    def load(self, split, *, lang="en", oracle=True):
        return load_datasets.load_query_local(SimpleNamespace(
            splits=split, lang=lang, oracle_translation=oracle,
        ))

    def test_same_uid_datasets_are_isolated_in_either_directory_order(self):
        for split in ("phase2_familiar", "phase2_heldout_sim"):
            self.write_split(split)
        self.write_query("phase2_familiar_EN", "references", oracle=True)
        self.write_query("phase2_heldout_sim_EN", "simulation")

        original_listdir = os.listdir
        data_dir = self.data_root / "en"
        for order in (
            ["phase2_familiar_EN", "phase2_heldout_sim_EN"],
            ["phase2_heldout_sim_EN", "phase2_familiar_EN"],
        ):
            def ordered_listdir(path):
                return order if Path(path) == data_dir else original_listdir(path)

            with self.subTest(order=order), patch.object(
                load_datasets.os, "listdir", side_effect=ordered_listdir,
            ):
                _, familiar = self.load("phase2_familiar")
                _, simulation = self.load("phase2_heldout_sim")
                self.assertEqual(familiar["shared"]["source"], "references")
                self.assertTrue(all(field in familiar["shared"] for field in ORACLE_FIELDS))
                self.assertEqual(simulation["shared"]["source"], "simulation")
                self.assertTrue(all(field not in simulation["shared"] for field in ORACLE_FIELDS))

    def test_english_exact_directory_without_suffix_is_supported(self):
        self.write_split("custom")
        self.write_query("custom", "selected")
        self.write_query("unrelated_EN", "wrong sibling", oracle=True)
        _, queries = self.load("custom")
        self.assertEqual(queries["shared"]["source"], "selected")

    def test_chinese_exact_directory_is_selected(self):
        self.write_split("custom_zh")
        self.write_query("custom_zh", "selected", lang="zh")
        self.write_query("unrelated", "wrong sibling", lang="zh", oracle=True)
        _, queries = self.load("custom_zh", lang="zh")
        self.assertEqual(queries["shared"]["source"], "selected")

    def test_shard_without_exact_directory_keeps_legacy_lookup(self):
        self.write_split("historical_shard", ("first", "second"))
        self.write_query("older_dataset_EN", "first dataset", uid="first")
        self.write_query("another_dataset_EN", "second dataset", uid="second")
        ids, queries = self.load("historical_shard")
        self.assertEqual(ids, ["first", "second"])
        self.assertEqual(set(queries), set(ids))
        self.assertEqual(queries["first"]["source"], "first dataset")
        self.assertEqual(queries["second"]["source"], "second dataset")

    def test_oracle_disabled_strips_all_four_fields_without_changing_source(self):
        self.write_split("phase2_familiar")
        self.write_query("phase2_familiar_EN", "references", oracle=True)
        _, visible = self.load("phase2_familiar", oracle=False)
        self.assertTrue(all(field not in visible["shared"] for field in ORACLE_FIELDS))
        _, references = self.load("phase2_familiar", oracle=True)
        self.assertTrue(all(field in references["shared"] for field in ORACLE_FIELDS))


if __name__ == "__main__":
    unittest.main()
