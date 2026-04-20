#!/usr/bin/env python3
"""Audio preprocessing stage entrypoint."""

from __future__ import annotations

import sys

import hydra
from omegaconf import DictConfig

from core.cli_overrides import hybrid_argv_to_hydra
from data.preprocessor import preprocess


@hydra.main(version_base=None, config_path="../configs", config_name="preprocessor")
def main(cfg: DictConfig) -> None:
	preprocess(cfg)


if __name__ == "__main__":
	sys.argv = [sys.argv[0]] + hybrid_argv_to_hydra(sys.argv[1:])
	main()
