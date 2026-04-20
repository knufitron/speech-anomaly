from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


# Allow `import core...` when running `python -m unittest`.
_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))

from core.cli_overrides import (
	extract_external_config_path,
	hybrid_argv_to_hydra,
	run_from_saved_config,
)


class TestCliOverrides(unittest.TestCase):
	def test_dry_run_flag(self) -> None:
		self.assertEqual(hybrid_argv_to_hydra(["--dry-run"]), ["dry_run=true"])

	def test_dry_run_value(self) -> None:
		self.assertEqual(hybrid_argv_to_hydra(["--dry-run", "false"]), ["dry_run=false"])

	def test_group_overrides(self) -> None:
		self.assertEqual(hybrid_argv_to_hydra(["--data", "savee"]), ["data=savee"])
		self.assertEqual(hybrid_argv_to_hydra(["--data=savee"]), ["data=savee"])
		self.assertEqual(hybrid_argv_to_hydra(["--experiment", "anomaly"]), ["experiment=anomaly"])
		self.assertEqual(hybrid_argv_to_hydra(["--model", "isolation_forest"]), ["model=isolation_forest"])
		self.assertEqual(hybrid_argv_to_hydra(["--model=isolation_forest"]), ["model=isolation_forest"])
		self.assertEqual(hybrid_argv_to_hydra(["--report", "full"]), ["report=full"])

	def test_report_include_comma_separated(self) -> None:
		self.assertEqual(
			hybrid_argv_to_hydra(["--report", "roc,umap"]),
			["report.include=[roc,umap]"],
		)
		self.assertEqual(
			hybrid_argv_to_hydra(["--report.data_debug", "true"]),
			["report.data_debug=true"],
		)
		self.assertEqual(
			hybrid_argv_to_hydra(["--report.umap.n_neighbors", "20"]),
			["report.umap.n_neighbors=20"],
		)

	def test_report_top_shorthand_sets_nested_n(self) -> None:
		self.assertEqual(hybrid_argv_to_hydra(["--report.top", "15"]), ["report.top.n=15"])
		self.assertEqual(hybrid_argv_to_hydra(["--report.top=7"]), ["report.top.n=7"])

	def test_report_include_all(self) -> None:
		self.assertEqual(hybrid_argv_to_hydra(["--report", "all"]), ["report.include=all"])

	def test_report_single_include(self) -> None:
		self.assertEqual(hybrid_argv_to_hydra(["--report", "metrics"]), ["report.include=[metrics]"])

	def test_nested_overrides_space(self) -> None:
		self.assertEqual(
			hybrid_argv_to_hydra(["--model.params.n_estimators", "300"]),
			["model.params.n_estimators=300"],
		)

	def test_nested_overrides_equals(self) -> None:
		self.assertEqual(
			hybrid_argv_to_hydra(["--model.params.random_state=55"]),
			["model.params.random_state=55"],
		)

	def test_negative_number_value(self) -> None:
		self.assertEqual(
			hybrid_argv_to_hydra(["--model.params.n_jobs", "-1"]),
			["model.params.n_jobs=-1"],
		)

	def test_features_mode_nested(self) -> None:
		self.assertEqual(hybrid_argv_to_hydra(["--features.mode", "explicit"]), ["features.mode=explicit"])

	def test_passthrough_hydra_overrides(self) -> None:
		self.assertEqual(
			hybrid_argv_to_hydra(["dry_run=true", "model=isolation_forest"]),
			["dry_run=true", "model=isolation_forest"],
		)

	def test_passthrough_help(self) -> None:
		self.assertEqual(hybrid_argv_to_hydra(["--help"]), ["--help"])

	def test_extract_external_config_path_short_long_equals(self) -> None:
		p, r = extract_external_config_path(["-c", "/tmp/saved.yaml", "dry_run=true"])
		self.assertEqual(p, Path("/tmp/saved.yaml"))
		self.assertEqual(r, ["dry_run=true"])
		p2, r2 = extract_external_config_path(["--config", "./cfg.yaml"])
		self.assertEqual(p2, Path("./cfg.yaml"))
		self.assertEqual(r2, [])
		p3, r3 = extract_external_config_path(["--config=/abs/experiment.yaml", "a", "b"])
		self.assertEqual(p3, Path("/abs/experiment.yaml"))
		self.assertEqual(r3, ["a", "b"])

	def test_extract_external_config_path_errors(self) -> None:
		with self.assertRaises(ValueError):
			extract_external_config_path(["--config"])
		with self.assertRaises(ValueError):
			extract_external_config_path(["-c", "a.yaml", "--config", "b.yaml"])

	def test_run_from_saved_config_rejects_group_switch(self) -> None:
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			p = root / "c.yaml"
			p.write_text(
				"experiment:\n  name: anomaly\nmodel:\n  name: m\nrun:\n  output_dir: /old\n",
				encoding="utf-8",
			)
			with self.assertRaises(ValueError) as ctx:
				run_from_saved_config(root, p, ["model=isolation_forest"])
			self.assertIn("--config", str(ctx.exception))

	def test_run_from_saved_config_dry_run_writes_new_output(self) -> None:
		yaml = """\
dry_run: true
experiment:
  name: anomaly
  scaler: robust
  actor_zscore:
    enabled: false
model:
  name: logistic_regression
  params: {}
data:
  name: savee
  path_in: dummy.tsv
features:
  mode: all
run:
  output_dir: /zzz/old_run
logging:
  level: INFO
  mute: []
wandb:
  enabled: false
"""
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			p = root / "config_resolved.yaml"
			p.write_text(yaml, encoding="utf-8")
			run_from_saved_config(root, p, [])
			outs = list((root / "output" / "runner").rglob("config_resolved.yaml"))
			self.assertTrue(len(outs) >= 1, "expected new run dir with saved config")
			text = outs[-1].read_text(encoding="utf-8")
			self.assertIn("output/runner", text)
			self.assertNotIn("/zzz/old_run", text)

	def test_run_from_saved_config_report_full_preset(self) -> None:
		yaml = """\
dry_run: true
experiment:
  name: anomaly
  scaler: robust
  actor_zscore:
    enabled: false
model:
  name: logistic_regression
  params: {}
data:
  name: savee
  path_in: dummy.tsv
features:
  mode: all
run:
  output_dir: /zzz/old_run
report:
  include:
    - metrics
logging:
  level: INFO
  mute: []
wandb:
  enabled: false
"""
		with tempfile.TemporaryDirectory() as td:
			root = Path(td)
			p = root / "config_resolved.yaml"
			p.write_text(yaml, encoding="utf-8")
			ov = hybrid_argv_to_hydra(["--report", "full"])
			run_from_saved_config(root, p, ov)
			outs = list((root / "output" / "runner").rglob("config_resolved.yaml"))
			text = outs[-1].read_text(encoding="utf-8")
			self.assertIn("include: all", text)


if __name__ == "__main__":
	unittest.main()

