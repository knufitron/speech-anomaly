from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from data.dataset import META_COLUMNS, is_pandas_index_artifact_column

log = logging.getLogger(__name__)


def resolve_feature_columns(df: pd.DataFrame, cfg: DictConfig) -> list[str]:
	mode = cfg.features.get("mode", "all")
	meta = set(META_COLUMNS)

	if mode == "all":
		cols = [
			c
			for c in df.columns
			if c not in meta
			and not is_pandas_index_artifact_column(c)
			and pd.api.types.is_numeric_dtype(df[c])
		]
		log.info("Feature mode=all → %d numeric columns", len(cols))
		return cols

	if mode == "explicit":
		want = list(OmegaConf.to_container(cfg.features.columns, resolve=True))
		missing = [c for c in want if c not in df.columns]
		if missing:
			raise ValueError(f"Missing feature columns: {missing}")
		return want

	if mode == "prefixes":
		prefixes: Iterable[str] = OmegaConf.to_container(cfg.features.prefixes, resolve=True)
		cols = []
		for c in df.columns:
			if c in meta or is_pandas_index_artifact_column(c) or not pd.api.types.is_numeric_dtype(df[c]):
				continue
			if any(c.startswith(p) for p in prefixes):
				cols.append(c)
		log.info("Feature mode=prefixes → %d columns", len(cols))
		return cols

	raise ValueError(f"Unknown features.mode: {mode}")
