from __future__ import annotations

import itertools
from typing import Any, Iterator

from omegaconf import DictConfig, OmegaConf


def _as_plain(obj: Any) -> Any:
	if OmegaConf.is_config(obj):
		return OmegaConf.to_container(obj, resolve=True)
	return obj


def cartesian_sweep(sweep: dict[str, Any]) -> Iterator[dict[str, Any]]:
	"""Yield dicts from Cartesian product of per-key option lists."""
	if not sweep:
		yield {}
		return
	keys: list[str] = []
	val_lists: list[list[Any]] = []
	for k, v in sweep.items():
		v = _as_plain(v)
		if not isinstance(v, list):
			v = [v]
		keys.append(k)
		val_lists.append(v)
	for combo in itertools.product(*val_lists):
		yield dict(zip(keys, combo))


def model_param_variants(model_name: str, param_grids: DictConfig | dict) -> list[dict[str, Any]]:
	"""Merge param grid `base` with each combo from `sweep` for one model."""
	pg_all = param_grids
	if OmegaConf.is_config(pg_all):
		if model_name not in pg_all:
			return [{}]
		pg = pg_all[model_name]
	else:
		pg = (pg_all or {}).get(model_name)
		if pg is None:
			return [{}]
	pg = OmegaConf.create(pg) if not OmegaConf.is_config(pg) else pg
	base = _as_plain(pg.get("base"))
	if not base:
		base = {}
	sweep = pg.get("sweep")
	if not sweep:
		return [base]
	sweep_plain = _as_plain(sweep)
	return [{**base, **combo} for combo in cartesian_sweep(sweep_plain)]
