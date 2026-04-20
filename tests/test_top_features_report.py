from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))

from evaluation.feature_importance import feature_importance_percent
from evaluation.reports import report_top


class TestFeatureImportancePercent(unittest.TestCase):
	def test_tree_importances_sum_to_100(self) -> None:
		n = 5
		rf = RandomForestClassifier(n_estimators=20, random_state=0)
		rng = np.random.default_rng(0)
		X = rng.standard_normal((80, n))
		y = (X[:, 0] + 0.25 * X[:, 1] > 0).astype(int)
		rf.fit(X, y)
		packed = feature_importance_percent(rf, n)
		self.assertIsNotNone(packed)
		pct, source = packed
		self.assertEqual(source, "feature_importances_")
		self.assertAlmostEqual(float(np.sum(pct)), 100.0, places=5)
		self.assertTrue(np.all(pct >= 0))

	def test_linear_coef_abs_sum_to_100(self) -> None:
		n = 4
		lr = LogisticRegression(random_state=0)
		rng = np.random.default_rng(1)
		X = rng.standard_normal((60, n))
		y = (X[:, 2] > 0).astype(int)
		lr.fit(X, y)
		packed = feature_importance_percent(lr, n)
		self.assertIsNotNone(packed)
		pct, source = packed
		self.assertEqual(source, "|coef_| (normalized)")
		self.assertAlmostEqual(float(np.sum(pct)), 100.0, places=5)


class TestReportTop(unittest.TestCase):
	def test_tsv_sorted_most_important_first_and_png_written(self) -> None:
		n = 6
		rf = RandomForestClassifier(n_estimators=30, random_state=2)
		rng = np.random.default_rng(2)
		X = rng.standard_normal((100, n))
		y = (3.0 * X[:, 3] + X[:, 1] > 0).astype(int)
		rf.fit(X, y)
		names = [f"feat_{i}" for i in range(n)]
		cfg = OmegaConf.create({"report": {"top": {"n": 3}}})
		with tempfile.TemporaryDirectory() as td:
			out = Path(td)
			report_top(cfg, {"model": rf, "feature_names": names}, out)
			tsv = out / "top_features.tsv"
			png = out / "top_features.png"
			self.assertTrue(tsv.is_file())
			self.assertTrue(png.is_file())
			lines = tsv.read_text(encoding="utf-8").strip().split("\n")
			self.assertEqual(lines[0], "name\tpercent")
			self.assertEqual(len(lines), 4)
			pcts = [float(row.split("\t")[1]) for row in lines[1:]]
			self.assertEqual(pcts, sorted(pcts, reverse=True))

	def test_png_skipped_when_write_png_false(self) -> None:
		n = 4
		rf = RandomForestClassifier(n_estimators=20, random_state=3)
		rng = np.random.default_rng(3)
		X = rng.standard_normal((80, n))
		y = (X[:, 0] > 0).astype(int)
		rf.fit(X, y)
		names = [f"f{i}" for i in range(n)]
		cfg = OmegaConf.create({"report": {"top": {"n": 3, "write_png": False}}})
		with tempfile.TemporaryDirectory() as td:
			out = Path(td)
			report_top(cfg, {"model": rf, "feature_names": names}, out)
			self.assertTrue((out / "top_features.tsv").is_file())
			self.assertFalse((out / "top_features.png").exists())

	def test_skips_when_model_has_no_signal(self) -> None:
		class _NoImp:
			pass

		cfg = OmegaConf.create({"report": {"top": {"n": 5}}})
		with tempfile.TemporaryDirectory() as td:
			out = Path(td)
			report_top(cfg, {"model": _NoImp(), "feature_names": ["a", "b"]}, out)
			self.assertFalse((out / "top_features.tsv").exists())


if __name__ == "__main__":
	unittest.main()
