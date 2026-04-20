from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
	accuracy_score,
	average_precision_score,
	classification_report,
	confusion_matrix,
	f1_score,
	matthews_corrcoef,
	roc_auc_score,
)

# Binary anomaly setup: 0 = normal, 1 = anomaly (aligned with report plots).
_CLASS_LABELS: list[int] = [0, 1]
_TARGET_NAMES: list[str] = ["normal (0)", "anomaly (1)"]
_ROUND_DIGITS = 2


def _round_numbers(obj: Any, ndigits: int = _ROUND_DIGITS) -> Any:
	"""Round all ints/floats in nested dict/list structures to fixed decimals."""
	if obj is None:
		return None
	if isinstance(obj, (float, np.floating)):
		return round(float(obj), ndigits)
	if isinstance(obj, (int, np.integer)):
		return round(float(obj), ndigits)
	if isinstance(obj, dict):
		return {k: _round_numbers(v, ndigits) for k, v in obj.items()}
	if isinstance(obj, (list, tuple)):
		return [_round_numbers(x, ndigits) for x in obj]
	return obj


def build_metrics_dict(
	y_true: np.ndarray,
	y_pred: np.ndarray,
	y_score: np.ndarray | None,
) -> dict[str, Any]:
	yt = np.asarray(y_true)
	yp = np.asarray(y_pred)

	cr = classification_report(
		yt,
		yp,
		labels=_CLASS_LABELS,
		target_names=_TARGET_NAMES,
		output_dict=True,
		zero_division=0,
	)
	cm = confusion_matrix(yt, yp, labels=_CLASS_LABELS)
	tn, fp, fn, tp = (int(x) for x in cm.ravel())

	out: dict[str, Any] = {
		"classification_report": cr,
		"confusion_matrix": {
			"labels": list(_CLASS_LABELS),
			"matrix": cm.tolist(),
		},
		"accuracy": float(accuracy_score(yt, yp)),
		"f1": float(f1_score(yt, yp, pos_label=1, zero_division=0)),
		"mcc": float(matthews_corrcoef(yt, yp)),
		"tn": tn,
		"fp": fp,
		"fn": fn,
		"tp": tp,
		"roc_auc": None,
		"average_precision": None,
	}

	if y_score is not None and len(np.unique(yt)) > 1:
		try:
			out["roc_auc"] = float(roc_auc_score(yt, y_score))
		except ValueError:
			pass
		try:
			out["average_precision"] = float(average_precision_score(yt, y_score))
		except ValueError:
			pass

	return _round_numbers(out)


def write_metrics(path: Path, y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None) -> None:
	m = build_metrics_dict(y_true, y_pred, y_score)
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")
	report_txt = classification_report(
		y_true,
		y_pred,
		labels=_CLASS_LABELS,
		target_names=_TARGET_NAMES,
		zero_division=0,
	)
	path.with_name("classification_report.txt").write_text(report_txt, encoding="utf-8")
