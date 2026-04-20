from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.model_selection import GroupShuffleSplit, train_test_split as sklearn_train_test_split

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))

from experiments.anomaly_detection import AnomalyExperiment


def _cfg():
	return OmegaConf.create(
		{
			"data": {
				"split": {"test_size": 0.2, "random_state": 42, "stratify": True},
				"filter": {"column": None, "value": None},
				"groupby": None,
			},
			"experiment": {
				"name": "anomaly",
				"scaler": "robust",
				"actor_zscore": {"enabled": False},
			},
			"features": {"mode": "explicit", "columns": ["f1", "f2"]},
			"model": {"name": "one_class_svm", "params": {"nu": 0.1, "kernel": "rbf", "gamma": "scale"}},
		}
	)


def _df() -> pd.DataFrame:
	rows = []
	for i in range(20):
		rows.append(
			{
				"duration": i / 10,
				"file": f"f{i}.wav",
				"emotion": 1 if i % 2 == 0 else 2,
				"statement": 1,
				"actor": 1 if i < 10 else 2,
				"label": 0 if i % 2 == 0 else 1,
				"f1": float(i),
				"f2": float(i + 1),
			}
		)
	return pd.DataFrame(rows)


class TestAnomalyPipelineGuards(unittest.TestCase):
	def test_train_test_split_uses_config_values(self) -> None:
		cfg = _cfg()
		df = _df()
		scaler = Mock()
		scaler.fit_transform.side_effect = lambda x, *a, **k: np.zeros((np.asarray(x).shape[0], 2))
		scaler.transform.side_effect = lambda x, *a, **k: np.zeros((np.asarray(x).shape[0], 2))
		model = Mock()
		model.predict.return_value = np.ones(4) * -1
		model.decision_function.return_value = np.zeros(4)

		with (
			patch("experiments.anomaly_detection.load_tabular", return_value=df),
			patch("experiments.anomaly_detection._get_scaler", return_value=scaler),
			patch("experiments.anomaly_detection.create_model", return_value=model),
			patch("experiments.anomaly_detection.train_test_split", wraps=sklearn_train_test_split) as split_mock,
		):
			AnomalyExperiment().run(cfg)

		first_call_kwargs = split_mock.call_args_list[0].kwargs
		self.assertEqual(first_call_kwargs["test_size"], cfg.data.split.test_size)
		self.assertEqual(first_call_kwargs["random_state"], cfg.data.split.random_state)
		model.fit.assert_called_once()
		self.assertEqual(len(model.fit.call_args[0]), 1)

	def test_scaler_transform_is_applied_to_test_data(self) -> None:
		cfg = _cfg()
		df = _df()
		scaler = Mock()
		scaler.fit_transform.return_value = np.zeros((16, 2))
		scaler.transform.return_value = np.zeros((4, 2))
		model = Mock()
		model.predict.return_value = np.ones(4) * -1
		model.decision_function.return_value = np.zeros(4)

		with (
			patch("experiments.anomaly_detection.load_tabular", return_value=df),
			patch("experiments.anomaly_detection._get_scaler", return_value=scaler),
			patch("experiments.anomaly_detection.create_model", return_value=model),
		):
			AnomalyExperiment().run(cfg)

		scaler.transform.assert_called_once()
		model.fit.assert_called_once()
		self.assertEqual(len(model.fit.call_args[0]), 1)

	def test_supervised_model_fit_receives_X_and_y(self) -> None:
		cfg = _cfg()
		cfg.model.name = "logistic_regression"
		cfg.model.params = {"max_iter": 1000, "random_state": 0}
		df = _df()
		scaler = Mock()
		scaler.fit_transform.return_value = np.zeros((16, 2))
		scaler.transform.return_value = np.zeros((4, 2))
		model = Mock()
		model.predict.return_value = np.zeros(4, dtype=int)
		model.predict_proba.return_value = np.column_stack([np.ones(4) * 0.5, np.ones(4) * 0.5])

		with (
			patch("experiments.anomaly_detection.load_tabular", return_value=df),
			patch("experiments.anomaly_detection._get_scaler", return_value=scaler),
			patch("experiments.anomaly_detection.create_model", return_value=model),
		):
			AnomalyExperiment().run(cfg)

		self.assertEqual(len(model.fit.call_args[0]), 2)

	def test_group_shuffle_split_used_when_groupby_set(self) -> None:
		cfg = _cfg()
		cfg.data.groupby = "actor"
		df = _df()
		scaler = Mock()
		scaler.fit_transform.side_effect = lambda x, *args, **kw: np.zeros((np.asarray(x).shape[0], 2))
		scaler.transform.side_effect = lambda x, *args, **kw: np.zeros((np.asarray(x).shape[0], 2))
		model = Mock()
		model.predict.side_effect = lambda x, *args, **kw: np.ones(np.asarray(x).shape[0]) * -1
		model.decision_function.side_effect = lambda x, *args, **kw: np.zeros(np.asarray(x).shape[0])

		with (
			patch("experiments.anomaly_detection.load_tabular", return_value=df),
			patch("experiments.anomaly_detection._get_scaler", return_value=scaler),
			patch("experiments.anomaly_detection.create_model", return_value=model),
			patch("experiments.anomaly_detection.GroupShuffleSplit", wraps=GroupShuffleSplit) as gss_mock,
		):
			AnomalyExperiment().run(cfg)

		gss_mock.assert_called_once()


if __name__ == "__main__":
	unittest.main()

