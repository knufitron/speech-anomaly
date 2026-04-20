from __future__ import annotations

import logging

from omegaconf import DictConfig, OmegaConf

from .registry import MODEL_REGISTRY, SUPERVISED_MODELS

log = logging.getLogger(__name__)


def create_model(cfg: DictConfig):
	name = str(cfg.model.name)
	if name not in MODEL_REGISTRY:
		hint = ""
		if name == "xgb_classifier":
			hint = " (install `xgboost` if you use xgb_classifier)"
		raise ValueError(f"Unknown model '{name}'. Registered: {list(MODEL_REGISTRY)}{hint}")
	params = OmegaConf.to_container(cfg.model.params, resolve=True)
	log.info("Building model %s with params %s", name, params)
	return MODEL_REGISTRY[name](**params)


def is_supervised_model(name: str) -> bool:
	return name in SUPERVISED_MODELS
