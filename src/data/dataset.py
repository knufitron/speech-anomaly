from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

META_COLUMNS = ("duration", "file", "emotion", "statement", "actor", "label")

_UNNAMED_INDEX_COL = re.compile(r"^Unnamed:\s*\d+$")


def is_pandas_index_artifact_column(name: str | int) -> bool:
	"""True if `name` is pandas' default when a DataFrame index was written as CSV columns."""
	return bool(_UNNAMED_INDEX_COL.match(str(name).strip()))


def prepare_labels(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
	"""Expect `label` to exist (0=inlier, 1=outlier); define it in preprocessing or upstream tables."""
	_ = cfg  # reserved for future label policy hooks
	out = df.copy()
	if "label" not in out.columns:
		raise ValueError(
			"Dataset must contain a 'label' column (0=inlier/normal, 1=outlier/anomaly). "
			"Compute labels before loading the runner (e.g. in preprocessing)."
		)
	return out


def apply_savee_actor_map(df: pd.DataFrame, mapping: dict[str, Any] | None) -> pd.DataFrame:
	if not mapping:
		return df
	out = df.copy()
	out["actor"] = out["actor"].replace(mapping)
	return out


def _normalize_filter_values(raw: Any) -> list[Any]:
	if raw is None:
		return []
	if OmegaConf.is_list(raw):
		return list(OmegaConf.to_container(raw, resolve=True))
	if isinstance(raw, (list, tuple)):
		return list(raw)
	if isinstance(raw, str):
		s = raw.strip()
		if s.startswith("[") and s.endswith("]"):
			try:
				parsed = json.loads(s.replace("'", '"'))
			except json.JSONDecodeError:
				try:
					parsed = json.loads(s)
				except json.JSONDecodeError as e:
					raise ValueError(f"Cannot parse filter value as list: {raw!r}") from e
			return parsed if isinstance(parsed, list) else [parsed]
	return [raw]


def apply_row_filter(df: pd.DataFrame, cfg: DictConfig) -> tuple[pd.DataFrame, bool]:
	"""
	Filter rows by cfg.data.filter.column and .value.

	value may be a scalar or list (or YAML/JSON list string). If column or value
	is missing / null, returns df unchanged and False.
	"""
	flt = cfg.data.get("filter")
	if flt is None:
		return df, False
	col = flt.get("column")
	val = flt.get("value")
	if col is None or (isinstance(col, str) and not col.strip()):
		return df, False
	if val is None:
		return df, False

	col = str(col).strip()
	if col not in df.columns:
		raise ValueError(f"Filter column '{col}' not in dataframe (have: {list(df.columns)})")

	values = _normalize_filter_values(val)
	if not values:
		return df, False

	out = df[df[col].isin(values)].copy()
	return out, True
