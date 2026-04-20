from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))

from experiments.anomaly_detection import _apply_actor_zscore


class TestActorZscore(unittest.TestCase):
	def test_uses_train_normal_stats_per_actor(self) -> None:
		train_df = pd.DataFrame(
			{
				"actor": [1, 1, 1, 2, 2],
				"emotion": [1, 1, 2, 1, 2],
				"label": [0, 0, 1, 0, 1],
				"f1": [1.0, 3.0, 5.0, 2.0, 8.0],
				"f2": [10.0, 14.0, 18.0, 20.0, 28.0],
			}
		)
		test_df = pd.DataFrame(
			{
				"actor": [1, 2],
				"emotion": [2, 2],
				"label": [1, 1],
				"f1": [5.0, 8.0],
				"f2": [18.0, 28.0],
			}
		)

		train_out, test_out = _apply_actor_zscore(
			train_df=train_df,
			test_df=test_df,
			feature_cols=["f1", "f2"],
		)

		# Actor 1 train-normal rows for f1 are [1, 3] -> mean=2, std=sqrt(2).
		self.assertAlmostEqual(float(test_out.loc[0, "f1"]), 2.121320, places=5)
		# Actor 2 has only one normal row -> std fallback to eps, finite output expected.
		self.assertTrue(pd.notna(test_out.loc[1, "f1"]))
		self.assertTrue(pd.notna(train_out.loc[0, "f2"]))

	def test_unknown_actor_uses_global_normal_fallback(self) -> None:
		train_df = pd.DataFrame(
			{
				"actor": [1, 1, 2],
				"emotion": [1, 1, 1],
				"label": [0, 0, 0],
				"f1": [0.0, 2.0, 4.0],
			}
		)
		test_df = pd.DataFrame({"actor": [99], "emotion": [2], "label": [1], "f1": [2.0]})

		_, test_out = _apply_actor_zscore(
			train_df=train_df,
			test_df=test_df,
			feature_cols=["f1"],
		)

		self.assertAlmostEqual(float(test_out.loc[0, "f1"]), 0.0, places=7)


if __name__ == "__main__":
	unittest.main()

