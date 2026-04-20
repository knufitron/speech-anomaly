#!/usr/bin/env python3
"""Run experiment stage from Hydra config, or replay a saved ``config_resolved.yaml`` (``-c`` / ``--config``)."""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
	sys.path.insert(0, str(_root / "src"))

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from core.cli_overrides import (
	extract_external_config_path,
	hybrid_argv_to_hydra,
	run_from_saved_config,
)
from core.runner import run_experiment


@hydra.main(version_base=None, config_path="../configs", config_name="runner")
def main(cfg: DictConfig) -> None:
	with open_dict(cfg):
		cfg.run.output_dir = HydraConfig.get().runtime.output_dir
	run_experiment(cfg)


if __name__ == "__main__":
	_project_root = _root
	_raw_argv = sys.argv[1:]
	_cfg_path, _rest = extract_external_config_path(_raw_argv)
	_hydra_tokens = hybrid_argv_to_hydra(_rest)
	if _cfg_path is None:
		sys.argv = [sys.argv[0]] + _hydra_tokens
		main()
	else:
		run_from_saved_config(_project_root, _cfg_path, _hydra_tokens)

