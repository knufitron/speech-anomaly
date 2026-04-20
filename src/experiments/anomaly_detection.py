from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

from data.dataset import apply_row_filter
from data.loader import load_tabular
from features.factory import resolve_feature_columns
from models.factory import create_model, is_supervised_model

from .base import Experiment

log = logging.getLogger(__name__)

# Written to reports/data_debug.tsv when cfg.report.data_debug is true.
TEST_DEBUG_META_COLUMNS = ("file", "emotion", "actor", "statement", "duration")


def _build_test_debug_table(
	test_df: pd.DataFrame,
	y_true: np.ndarray,
	y_pred: np.ndarray,
	y_score: np.ndarray | None,
) -> pd.DataFrame:
	"""One row per test sample; aligns with y_true / y_pred row order."""
	cols = [c for c in TEST_DEBUG_META_COLUMNS if c in test_df.columns]
	out = test_df[cols].copy().reset_index(drop=True)
	out["true_label"] = np.asarray(y_true).astype(int)
	out["pred_label"] = np.asarray(y_pred).astype(int)
	if y_score is not None:
		out["score"] = np.asarray(y_score, dtype=float)
	else:
		out["score"] = np.nan
	yt = out["true_label"].to_numpy()
	yp = out["pred_label"].to_numpy()
	out["outcome"] = np.select(
		[(yt == 0) & (yp == 0), (yt == 1) & (yp == 1), (yt == 0) & (yp == 1), (yt == 1) & (yp == 0)],
		["TN", "TP", "FP", "FN"],
		default="?",
	)
	return out


def _anomaly_scores(model, X: np.ndarray) -> np.ndarray:
	if hasattr(model, "decision_function"):
		return -np.ravel(model.decision_function(X))
	if hasattr(model, "score_samples"):
		return -np.ravel(model.score_samples(X))
	raise TypeError(f"No anomaly scoring method found for {type(model).__name__}")


def _supervised_scores(model, X: np.ndarray) -> np.ndarray | None:
	if hasattr(model, "predict_proba"):
		proba = model.predict_proba(X)
		if proba.shape[1] >= 2:
			return np.ravel(proba[:, 1])
	if hasattr(model, "decision_function"):
		return np.ravel(model.decision_function(X))
	return None


def _scores_for_threshold_sweep(y_score: np.ndarray) -> tuple[np.ndarray, str]:
	"""
	Map scores to [0, 1] for threshold comparison. Higher = more likely anomaly (class 1).
	Uses raw values when already in [0, 1]; otherwise min-max on the test split (debug only).
	"""
	s = np.asarray(y_score, dtype=float)
	if s.size == 0:
		return s, "empty scores"
	s = np.nan_to_num(s, nan=np.nanmedian(s), posinf=np.nanmax(s[np.isfinite(s)]), neginf=np.nanmin(s[np.isfinite(s)]))
	smin, smax = float(np.min(s)), float(np.max(s))
	if smax <= smin:
		return np.full_like(s, 0.5), "constant scores; sweep uses 0.5"
	in_01 = smin >= -1e-9 and smax <= 1.0 + 1e-9
	if in_01:
		return s, "scores already in [0, 1] (e.g. predict_proba for class 1)"
	sn = (s - smin) / (smax - smin)
	return sn, "scores min-max normalized to [0, 1] on test (e.g. decision_function or anomaly score)"


def _resolved_probability_threshold(cfg: DictConfig) -> float | None:
	"""`model.probability_threshold` iff set; null/omitted -> use estimator `predict`."""
	raw = cfg.model.get("probability_threshold") if cfg.get("model") else None
	if raw is None:
		return None
	return float(raw)


def _apply_probability_threshold(y_score: np.ndarray, threshold: float) -> np.ndarray:
	"""Hard labels (y_score >= threshold). Caller validates score range and threshold."""
	s = np.asarray(y_score, dtype=float)
	return (s >= float(threshold)).astype(int)


def _debug_print_threshold_sweep(y_true: np.ndarray, y_score: np.ndarray | None) -> None:
	if y_score is None:
		log.warning("debug_threshold_sweep skipped: no y_score")
		return
	s, note = _scores_for_threshold_sweep(y_score)
	log.info("debug_threshold_sweep: %s", note)
	yt = np.asarray(y_true).astype(int)
	for t in np.linspace(0.4, 0.7, 16):
		yp = (s >= t).astype(int)
		cm = confusion_matrix(yt, yp, labels=[0, 1])
		print(f"threshold={t:.2f}")
		print(cm)


class _PassthroughScaler(BaseEstimator, TransformerMixin):
	"""No-op scaling (identity)."""

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		return np.asarray(X, dtype=float)

	def fit_transform(self, X, y=None):
		return self.transform(X)


def _get_scaler(name: str):
	low = str(name).lower().strip()
	if low == "robust":
		return RobustScaler()
	if low in ("standard", "standardscaler"):
		return StandardScaler()
	if low in ("none", "identity", "passthrough"):
		return _PassthroughScaler()
	raise ValueError(f"Unknown scaler '{name}' (use robust, standard, or none)")


def _apply_actor_zscore(
	train_df: pd.DataFrame,
	test_df: pd.DataFrame,
	feature_cols: list[str],
	actor_column: str = "actor",
	label_column: str = "label",
	eps: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	train_out = train_df.copy()
	test_out = test_df.copy()

	if actor_column not in train_out.columns or label_column not in train_out.columns:
		log.warning(
			"Actor z-score skipped: missing '%s' or '%s' columns",
			actor_column,
			label_column,
		)
		return train_out, test_out

	global_normal = train_out[train_out[label_column] == 0]
	if global_normal.empty:
		log.warning("Actor z-score skipped: no train rows with %s==0 (inliers)", label_column)
		return train_out, test_out

	global_mean = global_normal[feature_cols].mean()
	global_std = global_normal[feature_cols].std().fillna(eps).replace(0, eps)

	actor_stats: dict[object, tuple[pd.Series, pd.Series]] = {}
	for actor in train_out[actor_column].unique():
		subset = train_out[(train_out[actor_column] == actor) & (train_out[label_column] == 0)]
		if subset.empty:
			continue
		mean = subset[feature_cols].mean()
		std = subset[feature_cols].std().fillna(eps).replace(0, eps)
		actor_stats[actor] = (mean, std)

	def _transform(frame: pd.DataFrame) -> pd.DataFrame:
		out = frame.copy()
		for actor in out[actor_column].unique():
			mask = out[actor_column] == actor
			mean, std = actor_stats.get(actor, (global_mean, global_std))
			out.loc[mask, feature_cols] = (out.loc[mask, feature_cols] - mean) / std
		return out

	return _transform(train_out), _transform(test_out)


def _train_test_partition(
	df: pd.DataFrame, y: np.ndarray, cfg: DictConfig
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
	group_col = cfg.data.get("groupby")
	if group_col is not None and str(group_col).strip():
		col = str(group_col).strip()
		if col not in df.columns:
			raise ValueError(f"data.groupby column '{col}' not in dataframe (columns: {list(df.columns)})")
		if cfg.data.split.get("stratify", False):
			log.warning("data.split.stratify is ignored when data.groupby is set (using GroupShuffleSplit)")
		groups = df[col].to_numpy()
		gss = GroupShuffleSplit(
			n_splits=1,
			test_size=float(cfg.data.split.test_size),
			random_state=int(cfg.data.split.random_state),
		)
		idx = np.arange(len(df))
		idx_train, idx_test = next(gss.split(idx, y, groups))
		train_df = df.iloc[idx_train].reset_index(drop=True)
		test_df = df.iloc[idx_test].reset_index(drop=True)
		y_train = y[idx_train]
		y_test = y[idx_test]
		log.info("Train/test split: GroupShuffleSplit(groupby=%s)", col)
		return train_df, test_df, y_train, y_test

	log.info("data.groupby is not set: using train_test_split")

	stratify = y if cfg.data.split.get("stratify", True) else None
	train_df, test_df, y_train, y_test = train_test_split(
		df,
		y,
		test_size=cfg.data.split.test_size,
		random_state=cfg.data.split.random_state,
		stratify=stratify,
	)
	return train_df, test_df, y_train, y_test


class AnomalyExperiment(Experiment):
	def run(self, cfg: DictConfig) -> dict:
		df = load_tabular(cfg)
		n_initial = len(df)
		log.info("Initial dataset size %d", n_initial)

		df, did_filter = apply_row_filter(df, cfg)
		if did_filter:
			log.info("Filtered dataset size: %d", len(df))

		feature_cols = resolve_feature_columns(df, cfg)
		y = df["label"].to_numpy()

		train_df, test_df, y_train, y_test = _train_test_partition(df, y, cfg)

		actor_zscore_cfg = cfg.experiment.get("actor_zscore", {})
		if actor_zscore_cfg.get("enabled", False):
			train_df, test_df = _apply_actor_zscore(
				train_df=train_df,
				test_df=test_df,
				feature_cols=feature_cols,
				actor_column=str(actor_zscore_cfg.get("actor_column", "actor")),
				label_column=str(actor_zscore_cfg.get("label_column", "label")),
				eps=float(actor_zscore_cfg.get("eps", 1e-6)),
			)
			log.info("Applied actor z-score normalization (train-derived stats)")

		X_train_df = train_df[feature_cols]
		X_test_df = test_df[feature_cols]

		scaler = _get_scaler(str(cfg.experiment.scaler))
		X_train = scaler.fit_transform(X_train_df)
		X_test = scaler.transform(X_test_df)

		model_name = str(cfg.model.name)
		supervised = is_supervised_model(model_name)
		model = create_model(cfg)

		if supervised:
			model.fit(X_train, y_train)
			y_pred = model.predict(X_test).astype(int)
			y_score = _supervised_scores(model, X_test)
			if y_score is None:
				y_score = y_pred.astype(float)
		else:
			inlier_mask = y_train == 0
			n_inliers = int(inlier_mask.sum())
			if n_inliers == 0:
				raise ValueError(
					"Unsupervised training needs at least one inlier (label 0) row in the training split."
				)
			log.info(
				"Unsupervised fit on %d inlier train rows (label 0) of %d train rows",
				n_inliers,
				len(X_train),
			)
			model.fit(X_train[inlier_mask])
			raw = model.predict(X_test)
			y_pred = (raw == -1).astype(int)
			y_score = _anomaly_scores(model, X_test)
			if int(y_test.sum()) == 0:
				log.warning(
					"Test set has no true anomalies (label 1); one-class evaluation will be one-sided."
				)

		thr = _resolved_probability_threshold(cfg)
		if thr is not None:
			if y_score is None:
				log.warning("model.probability_threshold=%s ignored: no y_score", thr)
			else:
				s = np.asarray(y_score, dtype=float)
				s_min, s_max = float(np.nanmin(s)), float(np.nanmax(s))
				if s_min < -1e-5 or s_max > 1.0 + 1e-5:
					log.warning(
						"model.probability_threshold=%s ignored: y_score not in [0, 1] (min=%.6g max=%.6g); keeping predict() labels",
						thr,
						s_min,
						s_max,
					)
				else:
					log.info("Applying model.probability_threshold=%s on y_score (in [0, 1])", thr)
					y_pred = _apply_probability_threshold(y_score, thr)

		log.info(
			"Train / test sizes: %d / %d | anomalies in test (true): %d",
			len(X_train),
			len(X_test),
			int(y_test.sum()),
		)

		if bool(cfg.experiment.get("debug_threshold_sweep", False)):
			_debug_print_threshold_sweep(y_test, y_score)

		return {
			"y_true": y_test,
			"y_pred": y_pred,
			"y_score": y_score,
			"feature_names": feature_cols,
			"model": model,
			"X_train_scaled": X_train,
			"X_test_scaled": X_test,
			"y_train": y_train,
			"test_debug": _build_test_debug_table(test_df, y_test, y_pred, y_score),
		}
