from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig

_fmt = "%(asctime)s.%(msecs)03d %(levelname)-5s [%(name)s] %(message)s"
_datefmt = "%Y-%m-%d %H:%M:%S"

_LEVELS: dict[str, int] = {
	"CRITICAL": logging.CRITICAL,
	"ERROR": logging.ERROR,
	"WARNING": logging.WARNING,
	"INFO": logging.INFO,
	"DEBUG": logging.DEBUG,
}


def setup_run_logging(cfg: DictConfig) -> Path:
	logging_cfg = cfg.get("logging")
	if logging_cfg is None:
		logging_cfg = {}

	level_name = str(logging_cfg.get("level", "INFO")).upper()
	level = _LEVELS.get(level_name, logging.INFO)

	out_dir = Path(cfg.run.output_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	log_path = out_dir / "run.log"

	root = logging.getLogger()
	root.handlers.clear()
	root.setLevel(level)

	file_handler = logging.FileHandler(log_path, encoding="utf-8")
	file_handler.setLevel(level)
	file_handler.setFormatter(logging.Formatter(_fmt, datefmt=_datefmt))

	console = logging.StreamHandler()
	console.setLevel(level)
	console.setFormatter(logging.Formatter(_fmt, datefmt=_datefmt))

	root.addHandler(file_handler)
	root.addHandler(console)

	mute_level_name = str(logging_cfg.get("mute_level", "WARNING")).upper()
	mute_level = _LEVELS.get(mute_level_name, logging.WARNING)
	for logger_name in logging_cfg.get("mute", []):
		logging.getLogger(str(logger_name)).setLevel(mute_level)

	logging.getLogger(__name__).info("Logging to %s", log_path.resolve())
	return log_path
