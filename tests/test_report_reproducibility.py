from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from omegaconf import OmegaConf

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))

from evaluation.reports import write_all_reports
from experiments.anomaly_detection import AnomalyExperiment


def _synthetic_df() -> pd.DataFrame:
	rows: list[dict] = []
	for actor in [1, 2, 3, 4]:
		for i in range(20):
			emotion = 1 if i < 12 else 2
			rows.append(
				{
					"duration": float(i) / 10.0,
					"file": f"actor{actor}_{i}.wav",
					"emotion": emotion,
					"statement": 1,
					"actor": actor,
					"label": 0 if emotion == 1 else 1,
					"f1": float(actor * 10 + i),
					"f2": float(actor * 5 + (i % 7)),
				}
			)
	return pd.DataFrame(rows)


def _base_cfg():
	return OmegaConf.create(
		{
			"data": {"split": {"test_size": 0.25, "random_state": 42, "stratify": True}},
			"experiment": {
				"name": "anomaly",
				"scaler": "robust",
				"actor_zscore": {
					"enabled": False,
					"actor_column": "actor",
					"label_column": "label",
					"eps": 1e-6,
				},
			},
			"features": {"mode": "explicit", "columns": ["f1", "f2"]},
			"model": {"name": "one_class_svm", "params": {"nu": 0.1, "kernel": "rbf", "gamma": "scale"}},
			"report": {"include": ["metrics"]},
			"run": {"output_dir": ""},
		}
	)


class TestClassificationReportReproducibility(unittest.TestCase):
	def test_report_is_identical_for_same_data_and_config(self) -> None:
		df = _synthetic_df()
		config_variants = [
			{},
			{
				"experiment": {"scaler": "standard"},
				"model": {
					"name": "isolation_forest",
					"params": {
						"n_estimators": 50,
						"contamination": "auto",
						"random_state": 123,
						"n_jobs": 1,
					},
				},
			},
			{"experiment": {"actor_zscore": {"enabled": True}}},
		]

		for variant in config_variants:
			with self.subTest(variant=variant):
				cfg_template = _base_cfg()
				variant_cfg = OmegaConf.create(variant)
				if "model" in variant_cfg:
					# Replace model block entirely to avoid mixing params between model families.
					cfg_template.model = variant_cfg.model
					del variant_cfg["model"]
				cfg_template = OmegaConf.merge(cfg_template, variant_cfg)
				with tempfile.TemporaryDirectory() as td:
					root = Path(td)
					reports: list[str] = []
					predictions: list[list[int]] = []

					for run_id in (1, 2):
						cfg = OmegaConf.create(OmegaConf.to_container(cfg_template, resolve=True))
						cfg.run.output_dir = str(root / f"run{run_id}")

						with patch("experiments.anomaly_detection.load_tabular", return_value=df):
							result = AnomalyExperiment().run(cfg)
						write_all_reports(cfg, result)

						report_path = Path(cfg.run.output_dir) / "reports" / "classification_report.txt"
						reports.append(report_path.read_text(encoding="utf-8"))
						predictions.append(result["y_pred"].tolist())

					self.assertEqual(reports[0], reports[1])
					self.assertEqual(predictions[0], predictions[1])


if __name__ == "__main__":
	unittest.main()

