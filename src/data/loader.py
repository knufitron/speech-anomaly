from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from .dataset import apply_savee_actor_map, is_pandas_index_artifact_column, prepare_labels

log = logging.getLogger(__name__)


def _read_delimited(path: Path) -> pd.DataFrame:
	if path.suffix.lower() == ".tsv":
		return pd.read_csv(path, sep="\t")
	if path.suffix.lower() == ".csv":
		return pd.read_csv(path)
	return pd.read_csv(path)


def _strip_index_artifact_columns(df: pd.DataFrame) -> pd.DataFrame:
	"""Drop `Unnamed: 0`-style columns from exports that accidentally included the index."""
	to_drop = [c for c in df.columns if is_pandas_index_artifact_column(c)]
	if to_drop:
		log.info("Dropping pandas index artifact columns from tabular data: %s", to_drop)
		return df.drop(columns=to_drop)
	return df


def load_tabular(cfg: DictConfig) -> pd.DataFrame:
	path = Path(to_absolute_path(cfg.data.path))
	parquet_path = path.with_suffix(".parquet")

	if path.suffix.lower() == ".parquet":
		log.info("Loading dataset from %s", path)
		df = pd.read_parquet(path)
	elif parquet_path.exists():
		log.info("Loading dataset from parquet cache %s", parquet_path)
		df = pd.read_parquet(parquet_path)
	else:
		log.info("Loading dataset from %s", path)
		df = _read_delimited(path)
		try:
			df.to_parquet(parquet_path, index=False)
			log.info("Saved parquet cache to %s", parquet_path)
		except Exception as exc:  # best-effort cache creation
			log.warning("Failed to save parquet cache %s (%s)", parquet_path, exc)

	df = _strip_index_artifact_columns(df)

	mapping = cfg.data.get("savee_actor_map")
	if mapping is not None:
		mapping = OmegaConf.to_container(mapping, resolve=True)
	df = apply_savee_actor_map(df, mapping)
	return prepare_labels(df, cfg)
