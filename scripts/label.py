#!/usr/bin/env python3
"""Add a binary `label` column: 0 where a column matches a value, 1 otherwise.

Examples:
  python scripts/label.py -c emotion -v 1 -i data/features/SAVEE_VAD/all.tsv \\
    -o data/features/SAVEE_VAD/all_labeled.tsv

  python scripts/label.py --match-col emotion --match-value 1 --in data/features/SAVEE_VAD/all.tsv
  # writes data/features/SAVEE_VAD/all_labeled.tsv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _parse_match_value(raw: str, series: pd.Series):
	"""Match CLI string to series dtype where possible."""
	if pd.api.types.is_numeric_dtype(series):
		return pd.to_numeric(raw, errors="coerce")
	if pd.api.types.is_bool_dtype(series) and raw.lower() in ("true", "false"):
		return raw.lower() == "true"
	return raw


def _default_out_path(path_in: Path) -> Path:
	return path_in.with_name(f"{path_in.stem}_labeled{path_in.suffix}")


def main(argv: list[str] | None = None) -> int:
	p = argparse.ArgumentParser(description=__doc__)
	p.add_argument("-c", "--match-col", required=True, help="Column name to test (e.g. emotion).")
	p.add_argument(
		"-v",
		"--match-value",
		required=True,
		help="Value that maps to label 0 (inlier); others -> 1.",
	)
	p.add_argument(
		"-i",
		"--in",
		dest="path_in",
		required=True,
		type=Path,
		metavar="PATH",
		help="Input table path (TSV by default).",
	)
	p.add_argument(
		"-o",
		"--out",
		dest="path_out",
		type=Path,
		default=None,
		metavar="PATH",
		help="Output path; default: <stem>_labeled<suffix> next to input.",
	)
	p.add_argument(
		"-s",
		"--sep",
		default="\t",
		help="Field separator (default: tab).",
	)
	args = p.parse_args(argv)

	path_in = args.path_in.expanduser()
	if not path_in.is_file():
		print(f"error: input not found: {path_in}", file=sys.stderr)
		return 1

	path_out = args.path_out.expanduser() if args.path_out else _default_out_path(path_in)
	path_out.parent.mkdir(parents=True, exist_ok=True)

	df = pd.read_csv(path_in, sep=args.sep)
	if args.match_col not in df.columns:
		print(f"error: column {args.match_col!r} not in table (have: {list(df.columns)})", file=sys.stderr)
		return 1

	col = df[args.match_col]
	match_val = _parse_match_value(str(args.match_value).strip(), col)
	if pd.isna(match_val) and pd.api.types.is_numeric_dtype(col):
		print(f"error: could not parse --match-value {args.match_value!r} as numeric", file=sys.stderr)
		return 1

	# 0 = matches condition (e.g. nominal / inlier); 1 = everything else (anomaly)
	df["label"] = (col != match_val).astype(int)
	# Treat NaNs in match column as not matching -> label 1
	mask_nan = col.isna()
	if mask_nan.any():
		df.loc[mask_nan, "label"] = 1

	df.to_csv(path_out, sep=args.sep, index=False)
	print(f"wrote {path_out} ({len(df)} rows, label==0: {int((df['label'] == 0).sum())})")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
