from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))

from data.dataset import is_pandas_index_artifact_column
from features.factory import resolve_feature_columns


class TestIndexArtifactColumns(unittest.TestCase):
	def test_is_pandas_index_artifact_column(self) -> None:
		self.assertTrue(is_pandas_index_artifact_column("Unnamed: 0"))
		self.assertTrue(is_pandas_index_artifact_column("Unnamed: 12"))
		self.assertFalse(is_pandas_index_artifact_column("F0_mean"))
		self.assertFalse(is_pandas_index_artifact_column("Unnamed"))

	def test_resolve_feature_columns_excludes_unnamed_index(self) -> None:
		df = pd.DataFrame(
			{
				"Unnamed: 0": range(8),
				"F0_mean": [0.1] * 8,
				"label": [0, 0, 0, 0, 1, 1, 1, 1],
				"actor": [1] * 8,
			}
		)
		cfg = OmegaConf.create({"features": {"mode": "all"}})
		cols = resolve_feature_columns(df, cfg)
		self.assertIn("F0_mean", cols)
		self.assertNotIn("Unnamed: 0", cols)


if __name__ == "__main__":
	unittest.main()
