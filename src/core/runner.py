from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from evaluation.reports import write_all_reports
from experiments.anomaly_detection import AnomalyExperiment
from utils.logging import setup_run_logging
from utils.wandb_logger import log_run_artifacts_to_wandb, maybe_init_wandb

log = logging.getLogger(__name__)

EXPERIMENTS: dict[str, type] = {
	"anomaly": AnomalyExperiment,
}


def run_experiment(cfg: DictConfig) -> dict[str, Any]:
	setup_run_logging(cfg)
	out = Path(cfg.run.output_dir)
	out.mkdir(parents=True, exist_ok=True)
	(out / "config_resolved.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
	wandb_run = maybe_init_wandb(cfg)

	if cfg.get("dry_run"):
		log.info("Dry run — skipping training and reports.")
		if wandb_run is not None:
			wandb_run.finish()
		return {"dry_run": True}

	experiment_name = str(cfg.experiment.name)
	if experiment_name not in EXPERIMENTS:
		raise ValueError(f"Unknown experiment '{experiment_name}'. Options: {list(EXPERIMENTS)}")

	exp_cls = EXPERIMENTS[experiment_name]
	result = exp_cls().run(cfg)
	write_all_reports(cfg, result)
	if wandb_run is not None:
		log_run_artifacts_to_wandb(cfg, result, out)
		wandb_run.finish()
	log.info("Run complete.")
	return result
