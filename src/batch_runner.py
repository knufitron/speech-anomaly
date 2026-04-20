#!/usr/bin/env python3
"""Run a Cartesian grid of experiments; write per-run dirs and a summary TSV."""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf, open_dict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
	sys.path.insert(0, str(_SRC))

from batch.grid import model_param_variants
from core.cli_overrides import hybrid_argv_to_hydra
from core.runner import run_experiment
from models.registry import MODEL_REGISTRY
from utils.progress_bar import progress_bar

_CONFIG_DIR = str(_PROJECT_ROOT / "configs")


def _installed_models() -> list[str]:
	try:
		from xgboost import XGBClassifier  # noqa: F401
	except ImportError:
		return sorted(k for k in MODEL_REGISTRY if k != "xgb_classifier")
	return sorted(MODEL_REGISTRY)


def _resolve_batch_models(batch: dict[str, Any]) -> list[str]:
	"""
	`batch.models`: explicit allowlist, or null = all installed registry models.
	`batch.models_skip`: names removed after the allowlist (and must still be in MODEL_REGISTRY).
	"""
	raw = batch.get("models")
	if raw:
		candidates = [str(m).strip() for m in raw if str(m).strip()]
	else:
		candidates = list(_installed_models())
	skip_raw = batch.get("models_skip") or []
	skip = {str(s).strip() for s in skip_raw if str(s).strip()}
	out: list[str] = []
	for m in candidates:
		if m in skip:
			continue
		if m not in MODEL_REGISTRY:
			continue
		out.append(m)
	return out


def _exp_subdir_name() -> str:
	now = datetime.now()
	return f"{now:%Y-%m-%d}_{now:%H-%M-%S}.{now.microsecond // 1000:03d}"


def _read_metrics_summary(exp_dir: Path) -> dict[str, Any]:
	p = exp_dir / "metrics.json"
	if not p.exists():
		p = exp_dir / "reports" / "metrics.json"
	if not p.exists():
		return {}
	with p.open(encoding="utf-8") as f:
		return json.load(f)


def _flatten_metrics_for_row(m: dict[str, Any]) -> dict[str, Any]:
	cr = m.get("classification_report") or {}
	macro = cr.get("macro avg") or {}
	out = {
		"accuracy": m.get("accuracy"),
		"f1": m.get("f1"),
		"f1_macro": macro.get("f1-score"),
		"mcc": m.get("mcc"),
		"tp": m.get("tp"),
		"tn": m.get("tn"),
		"fp": m.get("fp"),
		"fn": m.get("fn"),
		"roc_auc": m.get("roc_auc"),
		"average_precision": m.get("average_precision"),
	}
	return {k: v for k, v in out.items()}


def read_top_feature_names(path: Path, n: int) -> list[str]:
	if not path.is_file():
		return []
	df = pd.read_csv(path, sep="\t")
	if "name" not in df.columns:
		return []
	return [str(x) for x in df["name"].head(n).tolist()]


def _save_experiment_record(exp_dir: Path, cfg: DictConfig, notes: str) -> None:
	header = f"# batch_runner experiment record\n# {notes}\n\n"
	text = header + OmegaConf.to_yaml(cfg, resolve=True)
	(exp_dir / "experiment.yaml").write_text(text, encoding="utf-8")


def _copy_metrics_to_root(exp_dir: Path) -> None:
	src = exp_dir / "reports" / "metrics.json"
	dst = exp_dir / "metrics.json"
	if src.exists():
		shutil.copy2(src, dst)


def _merge_model_params(cfg: DictConfig, params: dict[str, Any]) -> None:
	if not params:
		return
	with open_dict(cfg.model):
		plain = OmegaConf.to_container(cfg.model.params, resolve=True) if cfg.model.get("params") else {}
		merged = {**dict(plain), **params}
		cfg.model.params = OmegaConf.create(merged)


def _strip_structured_defaults_for_note(cfg: DictConfig) -> str:
	"""Short human-readable override summary."""
	parts = [
		f"model={cfg.model.name}",
		f"data={cfg.data.name}",
		f"experiment.scaler={cfg.experiment.scaler}",
		f"actor_zscore={cfg.experiment.actor_zscore.enabled}",
		f"features.mode={cfg.features.mode}",
	]
	if cfg.features.get("mode") == "explicit" and cfg.features.get("columns"):
		parts.append(f"features.n_columns={len(cfg.features.columns)}")
	return " ".join(parts)


def build_runner_cfg(
	overrides: list[str],
	model_params: dict[str, Any],
	set_output_dir: str,
	batch_flags: dict[str, Any],
) -> DictConfig:
	GlobalHydra.instance().clear()
	with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
		cfg = compose(config_name="runner", overrides=overrides)
	with open_dict(cfg):
		cfg.run.output_dir = set_output_dir
		if batch_flags.get("metrics_only"):
			# Include `top` so chained top-N runs can read `reports/top_features.tsv`.
			inc = ["metrics", "top"]
			cfg.report.include = inc
		if batch_flags.get("skip_top_png"):
			with open_dict(cfg.report.top):
				cfg.report.top.write_png = False
		cfg.wandb.enabled = False
	_merge_model_params(cfg, model_params)
	return cfg


def _batch_as_dict(batch_node: Any) -> dict[str, Any]:
	return OmegaConf.to_container(batch_node, resolve=True)


def main(cfg: DictConfig) -> None:
	os.environ.setdefault("MPLBACKEND", "Agg")
	b = _batch_as_dict(cfg.batch)
	param_grids = cfg.param_grids
	models = _resolve_batch_models(b)
	if not models:
		print("error: no models to run (check batch.models / batch.models_skip vs MODEL_REGISTRY)", file=sys.stderr)
		return
	datasets = list(b["datasets"])
	scalers = list(b["scalers"])
	zscores = [bool(x) for x in b["actor_zscore"]]
	feature_groups = list(b["feature_groups"])
	max_runs = int(b.get("max_runs") or 0)
	after_top20 = bool(b.get("after_all_top20"))
	top_n = int(b.get("top_n_features") or 20)
	metrics_only = bool(b.get("metrics_only_reports"))
	skip_top_png = bool(b.get("skip_top_png", True))
	flush_every = int(b.get("results_flush_every") or 0)

	out_root_raw = b.get("out_root")
	if out_root_raw and str(out_root_raw).strip():
		batch_root = Path(to_absolute_path(str(out_root_raw)))
	else:
		now = datetime.now()
		batch_root = _PROJECT_ROOT / "output" / "batch_runner" / f"{now:%Y-%m-%d}" / f"{now:%H-%M-%S}"
	batch_root.mkdir(parents=True, exist_ok=True)

	tsv_path = batch_root / "results.tsv"
	rows: list[dict[str, Any]] = []

	def flush_results(*, force: bool = False) -> None:
		if not rows:
			return
		if not force:
			if flush_every <= 0:
				return
			if len(rows) % flush_every != 0:
				return
		pd.DataFrame(rows).to_csv(tsv_path, sep="\t", index=False)
		print(f"[batch] wrote {len(rows)} rows -> {tsv_path}", flush=True)

	def run_one(
		*,
		exp_dir: Path,
		overrides: list[str],
		model_params: dict[str, Any],
		meta: dict[str, Any],
	) -> Path:
		exp_dir.mkdir(parents=True, exist_ok=True)
		rcfg = build_runner_cfg(
			overrides,
			model_params,
			str(exp_dir),
			{"metrics_only": metrics_only, "skip_top_png": skip_top_png},
		)
		note = _strip_structured_defaults_for_note(rcfg) + f" | params={json.dumps(model_params, sort_keys=True, default=str)}"
		_save_experiment_record(exp_dir, rcfg, note)
		if cfg.get("dry_run"):
			print(f"[dry-run] would run -> {exp_dir}")
			return exp_dir
		run_experiment(rcfg)
		_copy_metrics_to_root(exp_dir)
		m = _read_metrics_summary(exp_dir)
		row = {
			"experiment_dir": str(exp_dir.resolve()),
			**meta,
			"model_params_json": json.dumps(model_params, sort_keys=True, default=str),
			**_flatten_metrics_for_row(m),
		}
		rows.append(row)
		flush_results()
		return exp_dir

	primary_jobs: list[tuple[list[str], dict[str, Any], dict[str, Any]]] = []

	for model in models:
		if model not in MODEL_REGISTRY:
			continue
		for params in model_param_variants(model, param_grids):
			for dataset in datasets:
				for scaler in scalers:
					for z in zscores:
						for fg in feature_groups:
							overrides = [
								f"model={model}",
								f"data={dataset}",
								f"experiment.scaler={scaler}",
								f"experiment.actor_zscore.enabled={str(z).lower()}",
								f"features={fg}",
							]
							meta = {
								"model": model,
								"dataset": dataset,
								"scaler": scaler,
								"actor_zscore": z,
								"feature_group": fg,
								"is_top20_chain": False,
								"parent_experiment_dir": "",
							}
							primary_jobs.append((overrides, params, meta))

	limited = primary_jobs
	if max_runs > 0:
		limited = primary_jobs[:max_runs]

	n_primary = len(limited)
	for idx, (overrides, params, meta) in enumerate(limited, start=1):
		exp_dir = batch_root / _exp_subdir_name()
		parent_dir = run_one(exp_dir=exp_dir, overrides=overrides, model_params=params, meta=meta)
		if n_primary > 0:
			progress_bar(idx, n_primary)

		if after_top20 and meta["feature_group"] == "all" and not cfg.get("dry_run"):
			top_path = parent_dir / "reports" / "top_features.tsv"
			cols = read_top_feature_names(top_path, top_n)
			if cols:
				ch_dir = batch_root / _exp_subdir_name()
				ch_dir.mkdir(parents=True, exist_ok=True)
				ch_meta = {**meta, "feature_group": f"all_top{top_n}_chained", "is_top20_chain": True, "parent_experiment_dir": str(parent_dir.resolve())}
				rcfg2 = build_runner_cfg(
					[h for h in overrides if not h.startswith("features=")] + ["features=all"],
					params,
					str(ch_dir),
					{"metrics_only": metrics_only, "skip_top_png": skip_top_png},
				)
				with open_dict(rcfg2.features):
					rcfg2.features.mode = "explicit"
					rcfg2.features.columns = cols
				note2 = _strip_structured_defaults_for_note(rcfg2) + f" | chained_from={parent_dir.name} | params={json.dumps(params, sort_keys=True, default=str)}"
				_save_experiment_record(ch_dir, rcfg2, note2)
				if not cfg.get("dry_run"):
					run_experiment(rcfg2)
					_copy_metrics_to_root(ch_dir)
					m2 = _read_metrics_summary(ch_dir)
					rows.append(
						{
							"experiment_dir": str(ch_dir.resolve()),
							**ch_meta,
							"model_params_json": json.dumps(params, sort_keys=True, default=str),
							**_flatten_metrics_for_row(m2),
						}
					)
					flush_results()
					print(f"[batch] top-{top_n} chained run finished -> {ch_dir.name}", flush=True)

	if rows:
		flush_results(force=True)
	print(f"batch session root: {batch_root}")
	if rows:
		print(f"wrote {tsv_path} ({len(rows)} rows)")
	if cfg.get("dry_run"):
		print(f"[dry-run] planned {len(limited)} primary jobs (+ optional top-{top_n} chains)")


@hydra.main(version_base=None, config_path="../configs", config_name="batch_runner")
def hydra_entry(cfg: DictConfig) -> None:
	main(cfg)


if __name__ == "__main__":
	sys.argv = [sys.argv[0]] + hybrid_argv_to_hydra(sys.argv[1:])
	hydra_entry()
