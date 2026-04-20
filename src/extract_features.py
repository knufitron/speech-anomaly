#!/usr/bin/env python3
"""Feature extraction stage entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
	sys.path.insert(0, str(_root / "src"))

import hydra
from omegaconf import DictConfig

from core.cli_overrides import hybrid_argv_to_hydra
from features.feature_extractor import extract_features


@hydra.main(version_base=None, config_path="../configs", config_name="feature_extractor")
def main(cfg: DictConfig) -> None:
	extract_features(cfg)


if __name__ == "__main__":
	sys.argv = [sys.argv[0]] + hybrid_argv_to_hydra(sys.argv[1:])
	main()

