from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def _wandb_tags_as_list(wandb_cfg: Any, key: str = "tags") -> list[Any]:
	"""Hydra YAML `tags: null` sets the key to None; `.get("tags", [])` still returns None."""
	raw = wandb_cfg.get(key) if wandb_cfg is not None else None
	if raw is None:
		return []
	if OmegaConf.is_config(raw):
		return list(OmegaConf.to_container(raw, resolve=True))
	if isinstance(raw, (list, tuple)):
		return list(raw)
	return [raw]


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
	if not path.exists():
		return None
	try:
		return json.loads(path.read_text(encoding="utf-8"))
	except json.JSONDecodeError:
		return None


def _sanitize_wandb_metric_key(key: str) -> str:
	return key.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")


def _flatten_metrics_for_wandb(obj: Any, prefix: str = "") -> dict[str, float]:
	"""Nested metrics.json -> flat floats for wandb.log (which expects scalars)."""
	out: dict[str, float] = {}
	if obj is None:
		return out
	if isinstance(obj, dict):
		for k, v in obj.items():
			part = _sanitize_wandb_metric_key(str(k))
			p = f"{prefix}/{part}" if prefix else part
			out.update(_flatten_metrics_for_wandb(v, p))
	elif isinstance(obj, list):
		for i, row in enumerate(obj):
			p = f"{prefix}_{i}" if prefix else str(i)
			if isinstance(row, list):
				for j, val in enumerate(row):
					if isinstance(val, (int, float)) and not isinstance(val, bool):
						x = float(val)
						if np.isnan(x):
							continue
						cell = f"{prefix}_cm_{i}_{j}" if prefix else f"cm_{i}_{j}"
						out[_sanitize_wandb_metric_key(cell)] = x
			elif isinstance(row, (int, float)) and not isinstance(row, bool):
				x = float(row)
				if np.isnan(x):
					continue
				out[_sanitize_wandb_metric_key(p)] = x
	elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
		x = float(obj)
		if np.isnan(x):
			return out
		out[_sanitize_wandb_metric_key(prefix)] = x
	return out


def maybe_init_wandb(cfg: DictConfig):
	wandb_cfg = cfg.get("wandb")
	if wandb_cfg is None or not bool(wandb_cfg.get("enabled", False)):
		return None

	import wandb

	mode = str(wandb_cfg.get("mode", "online"))
	project = str(wandb_cfg.get("project", "test-proj"))
	entity = wandb_cfg.get("entity")
	tags = _wandb_tags_as_list(wandb_cfg, "tags")
	tags = [str(t) for t in tags]
	tags.extend(_auto_tags(cfg))
	tags = sorted(set(tags))
	notes = wandb_cfg.get("notes")

	config_payload = OmegaConf.to_container(cfg, resolve=True)
	run = wandb.init(
		project=project,
		entity=entity,
		mode=mode,
		tags=tags,
		notes=notes,
		config=config_payload,
	)
	log.info("W&B initialized: project=%s mode=%s run_id=%s", project, mode, run.id)
	return run


def _auto_tags(cfg: DictConfig) -> list[str]:
	tags: list[str] = []
	dataset_path = str(cfg.data.get("path", "")).lower()
	for marker in ("prosodic", "acoustic", "voice_quality", "all"):
		if marker in dataset_path:
			tags.append(f"features:{marker}")
			break
	if cfg.data.get("name"):
		tags.append(f"dataset:{str(cfg.data.name).lower()}")
	if cfg.model.get("name"):
		tags.append(f"model:{str(cfg.model.name).lower()}")
	return tags


def _as_binary_probs(y_score: Any, y_pred: Any) -> np.ndarray:
	if y_score is not None:
		score = np.asarray(y_score, dtype=float).ravel()
		if score.size > 0:
			s_min = float(np.nanmin(score))
			s_max = float(np.nanmax(score))
			if s_max > s_min:
				p1 = (score - s_min) / (s_max - s_min)
			else:
				p1 = np.zeros_like(score)
			return np.column_stack([1.0 - p1, p1])

	pred = np.asarray(y_pred, dtype=float).ravel()
	pred = np.clip(pred, 0.0, 1.0)
	return np.column_stack([1.0 - pred, pred])


def log_run_artifacts_to_wandb(cfg: DictConfig, result: dict[str, Any], out_dir: Path) -> None:
	import wandb

	metrics_path = out_dir / "reports" / "metrics.json"

	metrics = _read_json_if_exists(metrics_path)
	if metrics is not None:
		flat_metrics = _flatten_metrics_for_wandb(metrics)
		if flat_metrics:
			wandb.log(flat_metrics)

	y_true = result.get("y_true")
	y_pred = result.get("y_pred")
	y_score = result.get("y_score")
	if y_true is not None and y_pred is not None:
		try:
			wandb.log({"num_samples_eval": int(len(y_true)), "num_predicted_anomalies": int(sum(y_pred))})
		except Exception:
			pass
		try:
			wandb.log(
				{
					"conf_mat": wandb.plot.confusion_matrix(
						y_true=np.asarray(y_true),
						preds=np.asarray(y_pred),
						class_names=["normal (0)", "anomaly (1)"],
					)
				}
			)
		except Exception:
			pass
		try:
			probs = _as_binary_probs(y_score, y_pred)
			yt = np.asarray(y_true).astype(int)
			labels = ["normal (0)", "anomaly (1)"]
			try:
				wandb.log({"roc": wandb.plot.roc_curve(yt, probs, labels=labels)})
			except Exception:
				pass
			try:
				wandb.log({"pr": wandb.plot.pr_curve(yt, probs, labels=labels)})
			except Exception:
				pass
		except Exception:
			pass

	_inc = cfg.report.include if cfg.get("report") else None
	if _inc is None:
		report_include_val: Any = "all"
	elif OmegaConf.is_config(_inc):
		report_include_val = OmegaConf.to_container(_inc, resolve=True)
	else:
		report_include_val = _inc

	meta = {
		"dataset_path": str(cfg.data.path),
		"model_name": str(cfg.model.name),
		"model_params": OmegaConf.to_container(cfg.model.params, resolve=True),
		"scaler": str(cfg.experiment.scaler),
		"actor_zscore_enabled": bool(cfg.experiment.actor_zscore.enabled),
		"split_test_size": float(cfg.data.split.test_size),
		"split_random_state": int(cfg.data.split.random_state),
		"split_stratify": bool(cfg.data.split.stratify),
		"report_include": report_include_val,
	}
	wandb.config.update(meta, allow_val_change=True)

	config_path = out_dir / "config_resolved.yaml"
	if config_path.exists():
		artifact = wandb.Artifact("resolved-config", type="config")
		artifact.add_file(str(config_path))
		wandb.log_artifact(artifact)

