from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Iterable

from omegaconf import DictConfig, OmegaConf, open_dict

# Hydra-style overrides that replace a whole config subtree (only valid when composing ``runner`` from
# the config store). They corrupt a saved resolved tree when merged via dotlist.
_GROUP_SWITCH_ROOT_KEYS = frozenset(
	{"model", "data", "features", "experiment", "report", "logging", "run", "wandb"},
)

_REPORT_YAML_PRESETS = frozenset({"basic", "full"})

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIGS_ROOT = _PROJECT_ROOT / "configs"


def _is_report_yaml_preset_override(override: str) -> bool:
	ov = override.strip()
	if "=" not in ov:
		return False
	key, _, val = ov.partition("=")
	return key == "report" and val.strip() in _REPORT_YAML_PRESETS


def _apply_report_yaml_presets(cfg: DictConfig, overrides: list[str]) -> list[str]:
	"""Load ``configs/report/{basic,full}.yaml`` when overrides contain ``report=basic`` / ``report=full``."""
	out: list[str] = []
	for ov in overrides:
		if _is_report_yaml_preset_override(ov):
			name = ov.partition("=")[2].strip()
			path = _CONFIGS_ROOT / "report" / f"{name}.yaml"
			if not path.is_file():
				raise FileNotFoundError(path)
			with open_dict(cfg):
				cfg.report = OmegaConf.load(path)
			continue
		out.append(ov)
	return out


def extract_external_config_path(argv: list[str]) -> tuple[Path | None, list[str]]:
	"""
	Pull ``-c path`` / ``--config path`` / ``--config=path`` from argv; return the path and remaining args.

	Raises:
		ValueError: duplicate or missing path.
	"""
	out: list[str] = []
	i = 0
	config_path: Path | None = None
	while i < len(argv):
		a = argv[i]
		if a in ("-c", "--config"):
			if config_path is not None:
				raise ValueError("only one --config / -c is allowed")
			if i + 1 >= len(argv):
				raise ValueError(f"{a} requires a file path")
			config_path = Path(argv[i + 1]).expanduser()
			i += 2
			continue
		if a.startswith("--config="):
			if config_path is not None:
				raise ValueError("only one --config / -c is allowed")
			config_path = Path(a.split("=", 1)[1].strip()).expanduser()
			i += 1
			continue
		out.append(a)
		i += 1
	return config_path, out


def _is_bad_group_switch_override(override: str) -> bool:
	"""True if ``override`` is a bare Hydra group switch (e.g. ``model=rf``) unsuitable for saved YAML merge."""
	ov = override.strip()
	if not ov or ov.startswith("+"):
		return False
	if "=" not in ov:
		return False
	key, _, _val = ov.partition("=")
	if "." in key:
		return False
	return key in _GROUP_SWITCH_ROOT_KEYS


def _load_saved_runner_yaml(path: Path) -> DictConfig:
	"""Load ``config_resolved.yaml`` / ``experiment.yaml`` written by a previous run."""
	try:
		loaded = OmegaConf.load(path)
	except Exception:
		raw = path.read_text(encoding="utf-8")
		lines: list[str] = []
		skip_header = True
		for line in raw.splitlines():
			if skip_header and line.strip().startswith("#"):
				continue
			skip_header = False
			lines.append(line)
		loaded = OmegaConf.load(io.StringIO("\n".join(lines).lstrip()))
	if not isinstance(loaded, DictConfig):
		loaded = OmegaConf.create(loaded)
	return loaded


def _fresh_runner_output_dir(project_root: Path) -> str:
	now = datetime.now()
	sub = f"{now:%H-%M-%S}.{now.microsecond // 1000:03d}"
	return str((project_root / "output" / "runner" / f"{now:%Y-%m-%d}" / sub).resolve())


def run_from_saved_config(project_root: Path, config_path: Path, hydra_style_overrides: list[str]) -> None:
	"""
	Load a saved resolved config YAML, merge dotted CLI overrides, optionally assign a new ``run.output_dir``,
	then run :func:`core.runner.run_experiment`.

	``hydra_style_overrides`` should be Hydra token lists (e.g. from :func:`hybrid_argv_to_hydra`), not raw argv.
	Raises:
		FileNotFoundError: missing config file.
		ValueError: unsupported whole-group overrides (e.g. ``model=x``); use dotted keys instead.
	"""
	path = config_path.expanduser().resolve()
	if not path.is_file():
		raise FileNotFoundError(path)

	cfg = _load_saved_runner_yaml(path)
	rest_overrides = _apply_report_yaml_presets(cfg, hydra_style_overrides)

	for ov in rest_overrides:
		if _is_bad_group_switch_override(ov):
			raise ValueError(
				"With --config / -c, use dotted keys so values merge into the saved tree "
				"(e.g. `model.name=random_forest`, `data.path_in=...`), not whole-group switches like "
				f"{ov!r} (those only work when composing `runner` from configs/)."
			)

	if rest_overrides:
		cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(rest_overrides))

	explicit_out = any(
		o.lstrip().startswith("run.output_dir=") or o.lstrip().startswith("run.output_dir+=")
		for o in rest_overrides
	)
	with open_dict(cfg):
		with open_dict(cfg.run):
			if not explicit_out:
				cfg.run.output_dir = _fresh_runner_output_dir(project_root)
			else:
				cfg.run.output_dir = str(Path(cfg.run.output_dir).expanduser().resolve())

	from core.runner import run_experiment

	run_experiment(cfg)

_REPORT_CONFIG_GROUPS = frozenset({"basic", "full"})


def _rewrite_report_cli_value(val: str) -> str:
	"""
	Map `--report X` to either a Hydra config group (basic/full) or `report.include=...`.

	Other report options use dotted keys (passed through):
	- `--report.data_debug true` -> report.data_debug=true (TSV dump; not part of include)
	- `--report.umap.n_neighbors 20` -> report.umap.n_neighbors=20
	- `--report.tsne.perplexity 25` -> report.tsne.perplexity=25
	- `--report.top 20` / `--report.top=20` -> report.top.n=20 (top-N feature list)

	Examples:
	- basic, full -> report=basic
	- roc,pr,umap -> report.include=[roc,pr,umap]
	- all -> report.include=all
	"""
	v = val.strip()
	low = v.lower()
	if low in _REPORT_CONFIG_GROUPS:
		return f"report={low}"
	if low == "all":
		return "report.include=all"
	parts = [p.strip() for p in v.split(",") if p.strip()]
	if not parts:
		return f"report={low}"
	if len(parts) == 1:
		return f"report.include=[{parts[0]}]"
	return f"report.include=[{','.join(parts)}]"


def _looks_like_number(token: str) -> bool:
	if token in {"-", "--"}:
		return False
	try:
		float(token)
		return True
	except ValueError:
		return False


def _next_override_value(argv: list[str], i: int) -> tuple[str | None, int]:
	"""Return (value, new_index) if argv[i+1] looks like a value token."""
	if i + 1 >= len(argv):
		return None, i
	nxt = argv[i + 1]
	# If next token doesn't look like another option, treat it as the value.
	# Also allow negative numbers (e.g. -1) as values.
	if not nxt.startswith("-") or _looks_like_number(nxt):
		return nxt, i + 2
	return None, i


def _normalize_key(key: str) -> str:
	# Hydra config uses underscores; allow GNU-style hyphenated CLI.
	if key in {"dry-run", "dryrun"}:
		return "dry_run"
	return key


def _maybe_rewrite_dotted_override(key: str, val: str) -> str | None:
	"""Map CLI shorthands to nested Hydra keys (avoid replacing a whole dict node)."""
	if key == "report.top":
		return f"report.top.n={val.strip()}"
	return None


def hybrid_argv_to_hydra(argv: list[str], *, allowed_top_level_keys: Iterable[str] | None = None) -> list[str]:
	"""
	Convert a mixed CLI into Hydra overrides.

	Examples:
	- `--dry-run` -> `dry_run=true`
	- `--data savee` -> `data=savee`
	- `--model.params.n_estimators 300` -> `model.params.n_estimators=300`
	- `--features.mode explicit` -> `features.mode=explicit`
	- `--model.params.random_state=55` -> `model.params.random_state=55`
	"""
	allowed = set(allowed_top_level_keys) if allowed_top_level_keys is not None else {
		"dry_run",
		"data",
		"experiment",
		"features",
		"logging",
		"model",
		"report",
		"run",
		"wandb",
	}
	passthrough_option_keys = {
		"help",
		"version",
		"config-path",
		"config-name",
		"multirun",
		"job-name",
		"run-dir",
	}

	out: list[str] = []
	i = 0
	while i < len(argv):
		a = argv[i]

		if not a.startswith("--") or len(a) <= 2:
			out.append(a)
			i += 1
			continue

		body = a[2:]
		body_key = body.partition("=")[0]

		# Don't interfere with Hydra's own CLI options.
		if body_key.startswith("hydra-") or body_key in passthrough_option_keys:
			out.append(a)
			i += 1
			continue

		# Normalize known key aliases (e.g. dry-run -> dry_run).
		body_key = _normalize_key(body_key)

		if "=" in body:
			# `--key=value` -> `key=value`
			key, _, val = body.partition("=")
			key = _normalize_key(key)
			rewritten = _maybe_rewrite_dotted_override(key, val)
			if rewritten is not None:
				out.append(rewritten)
			elif key == "report":
				out.append(_rewrite_report_cli_value(val))
			else:
				out.append(f"{key}={val}")
			i += 1
			continue

		# `--key value` or `--key` -> `key=value` (or `key=true`)
		key = body_key
		val, new_i = _next_override_value(argv, i)
		if val is not None:
			rewritten = _maybe_rewrite_dotted_override(key, val)
			if rewritten is not None:
				out.append(rewritten)
			elif key == "report":
				out.append(_rewrite_report_cli_value(val))
			else:
				out.append(f"{key}={val}")
			i = new_i
			continue

		# No value token found; treat as boolean flag.
		if "." in key or key in allowed:
			out.append(f"{key}=true")
		else:
			out.append(a)
		i += 1

	return out

