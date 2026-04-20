from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from omegaconf import OmegaConf

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))

from data import dataset as dataset_module
from evaluation.reports import write_all_reports
from experiments.anomaly_detection import AnomalyExperiment


def _prepare_labels_for_real_data_test(df: pd.DataFrame, cfg: OmegaConf) -> pd.DataFrame:
	"""Production requires `label`; legacy `data/*.tsv` may only have `emotion` (normal id=1)."""
	if "label" in df.columns:
		return dataset_module.prepare_labels(df, cfg)
	if "emotion" in df.columns:
		out = df.copy()
		out["label"] = (~out["emotion"].isin([1])).astype(int)
		return out
	raise ValueError("Real-data test expects 'label' or 'emotion' in the table.")


def _base_cfg():
	return OmegaConf.create(
		{
			"data": {"path": "", "savee_actor_map": None, "split": {"test_size": 0.2, "random_state": 42, "stratify": True}},
			"experiment": {
				"name": "anomaly",
				"scaler": "robust",
				"actor_zscore": {"enabled": False, "actor_column": "actor", "label_column": "label", "eps": 1e-6},
			},
			"features": {"mode": "all"},
			"model": {"name": "one_class_svm", "params": {"nu": 0.1, "kernel": "rbf", "gamma": "scale"}},
			"report": {"include": ["metrics"]},
			"logging": {"level": "WARNING", "mute": ["matplotlib", "PIL"], "mute_level": "WARNING"},
			"run": {"output_dir": ""},
		}
	)


class TestRealDataReportReproducibility(unittest.TestCase):
	def test_real_data_classification_report_stable(self) -> None:
		ravdess_path = _root / "data" / "ravdess.tsv"
		savee_path = _root / "data" / "savee.tsv"
		if not ravdess_path.exists() and not savee_path.exists():
			self.skipTest("No real datasets found in data/(ravdess|savee).tsv")

		cases = []
		if ravdess_path.exists():
			cases.append(
				(
					"ravdess_ocsvm",
					{
						"data": {"path": "data/ravdess.tsv", "savee_actor_map": None},
					},
				)
			)
			cases.append(
				(
					"ravdess_iforest",
					{
						"data": {"path": "data/ravdess.tsv", "savee_actor_map": None},
						"model": {
							"name": "isolation_forest",
							"params": {"n_estimators": 50, "contamination": "auto", "random_state": 42, "n_jobs": 1},
						},
					},
				)
			)
		if savee_path.exists():
			cases.append(
				(
					"savee_ocsvm_actor_z",
					{
						"data": {
							"path": "data/savee.tsv",
							"savee_actor_map": {"KL": 25, "JK": 26, "JE": 27, "DC": 28},
						},
						"experiment": {"actor_zscore": {"enabled": True}},
					},
				)
			)

		with tempfile.TemporaryDirectory() as td:
			tmp_root = Path(td)
			matrix_rows = []

			for case_name, overrides in cases:
				cfg_template = _base_cfg()
				override_cfg = OmegaConf.create(overrides)
				if "model" in override_cfg:
					cfg_template.model = override_cfg.model
					del override_cfg["model"]
				cfg_template = OmegaConf.merge(cfg_template, override_cfg)

				reports = []
				for run_id in (1, 2):
					cfg = OmegaConf.create(OmegaConf.to_container(cfg_template, resolve=True))
					cfg.run.output_dir = str(tmp_root / case_name / f"run{run_id}")
					with patch("data.loader.prepare_labels", side_effect=_prepare_labels_for_real_data_test):
						result = AnomalyExperiment().run(cfg)
					write_all_reports(cfg, result)
					rpt = Path(cfg.run.output_dir) / "reports" / "classification_report.txt"
					reports.append(rpt.read_text(encoding="utf-8"))

				self.assertEqual(reports[0], reports[1], f"Classification report differs for case '{case_name}'")
				matrix_rows.append(
					{
						"case": case_name,
						"dataset": str(cfg_template.data.path),
						"model": str(cfg_template.model.name),
						"actor_zscore": bool(cfg_template.experiment.actor_zscore.enabled),
						"classification_report": reports[0].replace("\n", "\\n"),
					}
				)

			matrix_path = tmp_root / "classification_report_matrix.csv"
			with matrix_path.open("w", newline="", encoding="utf-8") as f:
				writer = csv.DictWriter(
					f,
					fieldnames=["case", "dataset", "model", "actor_zscore", "classification_report"],
				)
				writer.writeheader()
				writer.writerows(matrix_rows)

			self.assertTrue(matrix_path.exists())


if __name__ == "__main__":
	unittest.main()

