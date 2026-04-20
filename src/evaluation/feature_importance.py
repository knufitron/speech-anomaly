from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def feature_importance_percent(model: Any, n_features: int) -> tuple[np.ndarray, str] | None:
	"""
	Return (percent per feature, method_description) where percents are non-negative and sum to ~100.
	Aligned with feature order at fit time. None if the estimator provides no usable signal.
	"""
	# Tree ensembles / sklearn forest / XGBoost
	imp = getattr(model, "feature_importances_", None)
	if imp is not None:
		arr = np.asarray(imp, dtype=float).ravel()
		if arr.shape[0] != n_features:
			log.warning(
				"feature_importances_ length %s != n_features %s; skipping top-features report",
				arr.shape[0],
				n_features,
			)
			return None
		arr = np.maximum(arr, 0.0)
		s = float(arr.sum())
		if s <= 0:
			return None
		return arr / s * 100.0, "feature_importances_"

	# Linear models: coefficient magnitude
	coef = getattr(model, "coef_", None)
	if coef is not None:
		arr = np.asarray(coef, dtype=float).ravel()
		if arr.size == n_features:
			arr = np.abs(arr)
		elif arr.size % n_features == 0:
			# Multiclass: mean abs coefficient per feature
			arr = np.abs(arr.reshape(-1, n_features)).mean(axis=0)
		else:
			log.warning("coef_ shape incompatible with n_features=%s; skipping top-features report", n_features)
			return None
		s = float(arr.sum())
		if s <= 0:
			return None
		return arr / s * 100.0, "|coef_| (normalized)"

	return None
