from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))

from evaluation.reports import write_all_reports


class TestDataDebugReport(unittest.TestCase):
	def test_writes_tsv_with_expected_columns(self) -> None:
		test_debug = pd.DataFrame(
			{
				"file": ["a.wav", "b.wav"],
				"emotion": [1, 2],
				"actor": [3, 4],
				"true_label": [0, 1],
				"pred_label": [0, 1],
				"score": [0.2, 0.8],
				"outcome": ["TN", "TP"],
			}
		)
		with tempfile.TemporaryDirectory() as td:
			out = Path(td)
			cfg = OmegaConf.create(
				{"report": {"include": ["metrics"], "data_debug": True}, "run": {"output_dir": str(out)}}
			)
			result = {
				"y_true": np.array([0, 1]),
				"y_pred": np.array([0, 1]),
				"y_score": np.array([0.2, 0.8]),
				"test_debug": test_debug,
			}
			write_all_reports(cfg, result)
			path = out / "reports" / "data_debug.tsv"
			self.assertTrue(path.is_file())
			text = path.read_text(encoding="utf-8")
			self.assertIn("emotion", text)
			self.assertIn("outcome", text)
			self.assertIn("TN", text)


if __name__ == "__main__":
	unittest.main()
