from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))

from data.dataset import apply_row_filter, prepare_labels


class TestDataFilter(unittest.TestCase):
	def test_prepare_labels_requires_label_column(self) -> None:
		df = pd.DataFrame({"emotion": [1, 2]})
		cfg = OmegaConf.create({"data": {}, "experiment": {}})
		with self.assertRaises(ValueError):
			prepare_labels(df, cfg)

	def test_prepare_labels_passes_when_label_present(self) -> None:
		df = pd.DataFrame({"emotion": [1, 2], "label": [0, 1]})
		cfg = OmegaConf.create({"data": {}, "experiment": {}})
		out = prepare_labels(df, cfg)
		self.assertTrue((out["label"] == df["label"]).all())

	def test_no_filter_when_column_missing(self) -> None:
		df = pd.DataFrame({"emotion": [1, 2], "label": [0, 1]})
		cfg = OmegaConf.create({"data": {"filter": {"column": None, "value": 1}}})
		out, did = apply_row_filter(df, cfg)
		self.assertFalse(did)
		self.assertEqual(len(out), 2)

	def test_filter_scalar(self) -> None:
		df = pd.DataFrame({"emotion": [1, 2, 1], "label": [0, 1, 0]})
		cfg = OmegaConf.create({"data": {"filter": {"column": "emotion", "value": 1}}})
		out, did = apply_row_filter(df, cfg)
		self.assertTrue(did)
		self.assertEqual(len(out), 2)

	def test_filter_list(self) -> None:
		df = pd.DataFrame({"emotion": [1, 2, 3], "label": [0, 1, 1]})
		cfg = OmegaConf.create({"data": {"filter": {"column": "emotion", "value": [1, 3]}}})
		out, did = apply_row_filter(df, cfg)
		self.assertTrue(did)
		self.assertEqual(len(out), 2)

	def test_filter_value_zero(self) -> None:
		df = pd.DataFrame({"emotion": [0, 1], "label": [0, 1]})
		cfg = OmegaConf.create({"data": {"filter": {"column": "emotion", "value": 0}}})
		out, did = apply_row_filter(df, cfg)
		self.assertTrue(did)
		self.assertEqual(len(out), 1)


if __name__ == "__main__":
	unittest.main()
