from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from evaluation.feature_importance import feature_importance_percent
from evaluation.metrics import write_metrics

log = logging.getLogger(__name__)

DEFAULT_REPORT_ORDER = ("metrics", "top", "confusion_matrix", "roc", "pr", "umap", "tsne", "pca")


def _fig_path(out: Path, name: str) -> Path:
	out.mkdir(parents=True, exist_ok=True)
	return out / name


def _log_report_debug(report_name: str, **fields: object) -> None:
	"""Log row / array sizes for diagnosing plots (enable logging.level=DEBUG)."""
	parts: list[str] = []
	for key, val in fields.items():
		if val is None:
			parts.append(f"{key}=None")
		else:
			parts.append(f"{key}={val}")
	log.debug("report[%s] %s", report_name, ", ".join(parts))


def resolve_report_include(cfg: DictConfig) -> list[str]:
	"""Expand report.include: default and 'all' -> every registered report in stable order."""
	raw = cfg.report.get("include", "all") if cfg.get("report") else "all"
	if raw is None:
		return list(DEFAULT_REPORT_ORDER)
	if isinstance(raw, str):
		s = raw.strip()
		if not s or s.lower() == "all":
			return list(DEFAULT_REPORT_ORDER)
		return [p.strip() for p in s.split(",") if p.strip()]
	if OmegaConf.is_list(raw):
		return [str(x).strip() for x in OmegaConf.to_container(raw, resolve=True) if str(x).strip()]
	if isinstance(raw, (list, tuple)):
		return [str(x).strip() for x in raw if str(x).strip()]
	return list(DEFAULT_REPORT_ORDER)


def report_metrics(
	cfg: DictConfig,
	y_true: np.ndarray,
	y_pred: np.ndarray,
	y_score: np.ndarray | None,
	out_dir: Path,
) -> None:
	yt = np.asarray(y_true)
	yp = np.asarray(y_pred)
	_log_report_debug(
		"metrics",
		len_y_true=int(yt.shape[0]),
		len_y_pred=int(yp.shape[0]),
		len_y_score=int(np.asarray(y_score).shape[0]) if y_score is not None else None,
	)
	write_metrics(out_dir / "metrics.json", y_true, y_pred, y_score)
	log.info("Wrote metrics.json and classification_report.txt")


def report_confusion_matrix(
	cfg: DictConfig,
	y_true: np.ndarray,
	y_pred: np.ndarray,
	out_dir: Path,
) -> None:
	yt = np.asarray(y_true)
	yp = np.asarray(y_pred)
	_log_report_debug("confusion_matrix", len_y_true=int(yt.shape[0]), len_y_pred=int(yp.shape[0]))
	disp = ConfusionMatrixDisplay.from_predictions(
		y_true,
		y_pred,
		display_labels=["normal (0)", "anomaly (1)"],
	)
	disp.ax_.set_title("Confusion matrix")
	path = _fig_path(out_dir, "confusion_matrix.png")
	disp.figure_.savefig(path, bbox_inches="tight", dpi=150)
	plt.close(disp.figure_)
	log.info("Wrote %s", path)


def report_roc(
	cfg: DictConfig,
	y_true: np.ndarray,
	y_score: np.ndarray,
	out_dir: Path,
) -> None:
	yt = np.asarray(y_true)
	ys = np.asarray(y_score)
	_log_report_debug("roc", len_y_true=int(yt.shape[0]), len_y_score=int(ys.shape[0]))
	if len(np.unique(y_true)) < 2:
		log.warning("ROC skipped: need both classes in y_true")
		return
	path = _fig_path(out_dir, "roc.png")
	display = RocCurveDisplay.from_predictions(y_true, y_score)
	display.ax_.set_title("ROC (1 = anomaly)")
	display.figure_.savefig(path, bbox_inches="tight", dpi=150)
	plt.close(display.figure_)
	log.info("Wrote %s", path)


def report_pr(
	cfg: DictConfig,
	y_true: np.ndarray,
	y_score: np.ndarray,
	out_dir: Path,
) -> None:
	yt = np.asarray(y_true)
	ys = np.asarray(y_score)
	_log_report_debug("pr", len_y_true=int(yt.shape[0]), len_y_score=int(ys.shape[0]))
	if len(np.unique(y_true)) < 2:
		log.warning("PR curve skipped: need both classes in y_true")
		return
	path = _fig_path(out_dir, "pr_curve.png")
	display = PrecisionRecallDisplay.from_predictions(y_true, y_score)
	display.ax_.set_title("Precision–Recall (1 = anomaly)")
	display.figure_.savefig(path, bbox_inches="tight", dpi=150)
	plt.close(display.figure_)
	log.info("Wrote %s", path)


def report_umap(cfg: DictConfig, result: dict[str, Any], out_dir: Path) -> None:
	X_train = result.get("X_train_scaled")
	X_test = result.get("X_test_scaled")
	y_train = result.get("y_train")
	y_true = result.get("y_true")
	if X_train is None or X_test is None or y_train is None or y_true is None:
		log.warning("UMAP skipped: missing train/test embeddings in result")
		return
	X_tr = np.asarray(X_train, dtype=float)
	X_te = np.asarray(X_test, dtype=float)
	y_tr = np.asarray(y_train)
	y_te = np.asarray(y_true)
	_log_report_debug(
		"umap",
		X_train_n=X_tr.shape[0],
		X_train_features=X_tr.shape[1],
		X_test_n=X_te.shape[0],
		len_y_train=int(y_tr.shape[0]),
		len_y_true=int(y_te.shape[0]),
		scatter_total_points=int(X_tr.shape[0] + X_te.shape[0]),
	)
	try:
		import umap  # type: ignore[import-not-found]
	except ImportError:
		log.warning("UMAP skipped: install umap-learn")
		return

	ucfg = cfg.report.get("umap", {})
	n_neighbors = min(int(ucfg.get("n_neighbors", 15)), max(len(X_train) - 1, 2))
	min_dist = float(ucfg.get("min_dist", 0.1))
	random_state = int(ucfg.get("random_state", 42))

	reducer = umap.UMAP(
		n_neighbors=n_neighbors,
		min_dist=min_dist,
		random_state=random_state,
	)
	emb_train = reducer.fit_transform(X_tr)
	emb_test = reducer.transform(X_te)

	fig, ax = plt.subplots(figsize=(8, 6))
	sc1 = ax.scatter(
		emb_train[:, 0],
		emb_train[:, 1],
		c=y_tr.astype(int),
		cmap="coolwarm",
		marker="o",
		alpha=0.55,
		vmin=0,
		vmax=1,
		label="train",
	)
	ax.scatter(
		emb_test[:, 0],
		emb_test[:, 1],
		c=y_te.astype(int),
		cmap="coolwarm",
		marker="x",
		alpha=0.85,
		vmin=0,
		vmax=1,
		label="test",
	)
	fig.colorbar(sc1, ax=ax, label="label (0=normal, 1=anomaly)")
	ax.set_title("UMAP (fit on train, transform train+test)")
	ax.legend()
	ax.set_xlabel("UMAP 1")
	ax.set_ylabel("UMAP 2")
	path = _fig_path(out_dir, "umap.png")
	fig.savefig(path, bbox_inches="tight", dpi=150)
	plt.close(fig)
	log.info("Wrote %s", path)


def report_tsne(cfg: DictConfig, result: dict[str, Any], out_dir: Path) -> None:
	X_train = result.get("X_train_scaled")
	X_test = result.get("X_test_scaled")
	y_train = result.get("y_train")
	y_true = result.get("y_true")
	if X_train is None or X_test is None or y_train is None or y_true is None:
		log.warning("t-SNE skipped: missing train/test features in result")
		return

	X_train = np.asarray(X_train, dtype=float)
	X_test = np.asarray(X_test, dtype=float)
	y_train = np.asarray(y_train).astype(int)
	y_true = np.asarray(y_true).astype(int)

	X_all = np.vstack([X_train, X_test])
	n_train = X_train.shape[0]
	n_total = X_all.shape[0]
	_log_report_debug(
		"tsne",
		X_train_n=X_train.shape[0],
		X_train_features=X_train.shape[1],
		X_test_n=X_test.shape[0],
		len_y_train=int(y_train.shape[0]),
		len_y_true=int(y_true.shape[0]),
		tsne_fit_n=n_total,
		scatter_total_points=n_total,
	)
	if n_total < 2:
		log.warning("t-SNE skipped: need at least 2 samples")
		return

	tcfg = cfg.report.get("tsne", {})
	n_components = int(tcfg.get("n_components", 2))
	random_state_raw = tcfg.get("random_state", 42)
	random_state = None if random_state_raw is None else int(random_state_raw)
	raw_perp = int(tcfg.get("perplexity", 30))
	perplexity = min(raw_perp, max(n_total - 1, 1))

	lr_raw = tcfg.get("learning_rate", "auto")
	if isinstance(lr_raw, str) and lr_raw.lower() == "auto":
		learning_rate: float | str = "auto"
	else:
		learning_rate = float(lr_raw)

	max_iter = int(tcfg.get("max_iter", 1000))

	tsne = TSNE(
		n_components=n_components,
		perplexity=float(perplexity),
		random_state=random_state,
		learning_rate=learning_rate,
		max_iter=max_iter,
		init="pca",
	)
	try:
		X_embedded = tsne.fit_transform(X_all)
	except ValueError as e:
		log.warning("t-SNE skipped: %s", e)
		return

	if X_embedded.shape[1] < 2:
		log.warning("t-SNE skipped: need at least 2 components for 2D plot")
		return
	emb_train = X_embedded[:n_train, :2]
	emb_test = X_embedded[n_train:, :2]

	fig, ax = plt.subplots(figsize=(8, 6))
	sc1 = ax.scatter(
		emb_train[:, 0],
		emb_train[:, 1],
		c=y_train,
		cmap="coolwarm",
		marker="o",
		alpha=0.55,
		vmin=0,
		vmax=1,
		label="train",
	)
	ax.scatter(
		emb_test[:, 0],
		emb_test[:, 1],
		c=y_true,
		cmap="coolwarm",
		marker="x",
		alpha=0.85,
		vmin=0,
		vmax=1,
		label="test",
	)
	fig.colorbar(sc1, ax=ax, label="label (0=normal, 1=anomaly)")
	ax.set_title("t-SNE (train+test joint embedding)")
	ax.legend()
	ax.set_xlabel("t-SNE 1")
	ax.set_ylabel("t-SNE 2")
	path = _fig_path(out_dir, "tsne.png")
	fig.savefig(path, bbox_inches="tight", dpi=150)
	plt.close(fig)
	log.info("Wrote %s", path)


def report_pca(cfg: DictConfig, result: dict[str, Any], out_dir: Path) -> None:
	X_train = result.get("X_train_scaled")
	X_test = result.get("X_test_scaled")
	y_train = result.get("y_train")
	y_true = result.get("y_true")
	if X_train is None or X_test is None or y_train is None or y_true is None:
		log.warning("PCA plot skipped: missing train/test features in result")
		return

	X_train = np.asarray(X_train, dtype=float)
	X_test = np.asarray(X_test, dtype=float)
	y_train = np.asarray(y_train).astype(int)
	y_true = np.asarray(y_true).astype(int)
	_log_report_debug(
		"pca",
		X_train_n=X_train.shape[0],
		X_train_features=X_train.shape[1],
		X_test_n=X_test.shape[0],
		len_y_train=int(y_train.shape[0]),
		len_y_true=int(y_true.shape[0]),
		scatter_total_points=int(X_train.shape[0] + X_test.shape[0]),
		X_train_scaled_max=X_train.max(),
		X_train_scaled_min=X_train.min(),
	)
	if X_train.shape[0] < 2 or X_train.shape[1] < 1:
		log.warning("PCA plot skipped: insufficient train samples or features")
		return

	pcfg = cfg.report.get("pca", {})
	n_comp_req = int(pcfg.get("n_components", 2))
	n_comp = int(min(n_comp_req, X_train.shape[0], X_train.shape[1]))
	if n_comp < 1:
		log.warning("PCA plot skipped: n_components resolved to 0")
		return

	rs_raw = pcfg.get("random_state", 42)
	random_state = None if rs_raw is None else int(rs_raw)

	# Clip to avoid outliers
	X_train = np.clip(X_train, -10, 10)
	X_test = np.clip(X_test, -10, 10)

	pca = PCA(n_components=n_comp, random_state=random_state)
	emb_train = pca.fit_transform(X_train)
	emb_test = pca.transform(X_test)

	if emb_train.shape[1] < 2:
		log.warning("PCA plot skipped: need at least 2 components for 2D plot")
		return

	fig, ax = plt.subplots(figsize=(8, 6))
	# ax.set_xlim(-1e6, 1e6)
	# ax.set_ylim(-1e6, 1e6)
	sc1 = ax.scatter(
		emb_train[:, 0],
		emb_train[:, 1],
		c=y_train,
		cmap="coolwarm",
		marker="o",
		alpha=0.55,
		vmin=0,
		vmax=1,
		label="train",
	)
	ax.scatter(
		emb_test[:, 0],
		emb_test[:, 1],
		c=y_true,
		cmap="coolwarm",
		marker="x",
		alpha=0.85,
		vmin=0,
		vmax=1,
		label="test",
	)
	fig.colorbar(sc1, ax=ax, label="label (0=normal, 1=anomaly)")
	exp_var = pca.explained_variance_ratio_
	ax.set_title(
		"PCA (fit on train, transform test) "
		f"— variance PC1={exp_var[0]:.2%}, PC2={exp_var[1]:.2%}"
		if len(exp_var) > 1
		else f"PCA — variance PC1={exp_var[0]:.2%}"
	)
	ax.legend()
	ax.set_xlabel("PC1")
	ax.set_ylabel("PC2")
	path = _fig_path(out_dir, "pca.png")
	fig.savefig(path, bbox_inches="tight", dpi=150)
	plt.close(fig)
	log.info("Wrote %s", path)


def report_top(cfg: DictConfig, result: dict[str, Any], out_dir: Path) -> None:
	model = result.get("model")
	raw_names = result.get("feature_names")
	if model is None or not raw_names:
		log.warning("top features skipped: missing model or feature_names in result")
		return
	names = list(raw_names)
	top_cfg = cfg.report.get("top", {}) or {}
	n_top = max(1, int(top_cfg.get("n", 20)))

	imp = feature_importance_percent(model, len(names))
	if imp is None:
		log.warning(
			"top features skipped: model %s exposes no feature_importances_ or coef_ compatible with %d features",
			type(model).__name__,
			len(names),
		)
		return
	pct, source = imp

	top_idx = np.argsort(-pct)[:n_top]
	sub_names = [names[i] for i in top_idx]
	sub_pct = pct[top_idx]
	_log_report_debug(
		"top",
		n_features=len(names),
		n_plotted=len(sub_names),
		method=source,
	)

	tsv_lines = ["name\tpercent"] + [f"{nm}\t{float(p):.6f}" for nm, p in zip(sub_names, sub_pct)]
	tsv_path = _fig_path(out_dir, "top_features.tsv")
	tsv_path.write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")
	log.info("Wrote %s", tsv_path)

	write_png = bool(top_cfg.get("write_png", True))
	if not write_png:
		log.info("Skipped top_features.png (report.top.write_png=false); TSV can be plotted offline.")
		return

	# Horizontal bars: most important at the top of the figure
	fig_h = max(4.0, 0.35 * len(sub_names))
	fig, ax = plt.subplots(figsize=(8, fig_h))
	y_pos = np.arange(len(sub_names) - 1, -1, -1)
	ax.barh(y_pos, sub_pct, align="center")
	ax.set_yticks(y_pos)
	ax.set_yticklabels(sub_names)
	ax.set_xlabel("Importance (%)")
	ax.set_title(f"Top {len(sub_names)} features ({source})")
	fig.tight_layout()
	img_path = _fig_path(out_dir, "top_features.png")
	fig.savefig(img_path, bbox_inches="tight", dpi=150)
	plt.close(fig)
	log.info("Wrote %s", img_path)


def report_data_debug(cfg: DictConfig, result: dict[str, Any], out_dir: Path) -> None:
	raw = result.get("test_debug")
	if raw is None:
		log.warning("data_debug skipped: no test_debug table in result")
		return
	if isinstance(raw, pd.DataFrame):
		table = raw
	else:
		table = pd.DataFrame(raw)
	if table.empty:
		log.warning("data_debug skipped: empty test_debug table")
		return
	_log_report_debug("data_debug", test_rows=int(len(table)), n_columns=int(len(table.columns)))
	path = _fig_path(out_dir, "data_debug.tsv")
	table.to_csv(path, sep="\t", index=False, na_rep="")
	log.info("Wrote %s", path)


REPORT_REGISTRY: dict[str, Any] = {
	"metrics": report_metrics,
	"top": report_top,
	"confusion_matrix": report_confusion_matrix,
	"roc": report_roc,
	"pr": report_pr,
	"umap": report_umap,
	"tsne": report_tsne,
	"pca": report_pca,
}


def write_all_reports(cfg: DictConfig, result: dict[str, Any]) -> None:
	reports = Path(cfg.run.output_dir) / "reports"
	reports.mkdir(parents=True, exist_ok=True)

	want = resolve_report_include(cfg)

	y_true = result["y_true"]
	y_pred = result["y_pred"]
	y_score = result.get("y_score")
	yt0 = np.asarray(y_true)
	yp0 = np.asarray(y_pred)
	_log_report_debug(
		"write_all_reports",
		reports_requested=len(want),
		len_y_true=int(yt0.shape[0]),
		len_y_pred=int(yp0.shape[0]),
		len_y_score=int(np.asarray(y_score).shape[0]) if y_score is not None else None,
	)

	for name in want:
		if name not in REPORT_REGISTRY:
			log.warning("Unknown report '%s', skipping", name)
			continue
		fn = REPORT_REGISTRY[name]
		if name in ("umap", "tsne", "pca", "top"):
			fn(cfg, result, reports)
		elif name in ("roc", "pr"):
			if y_score is None:
				log.warning("%s requested but no y_score; skipping", name.upper())
				continue
			fn(cfg, y_true, y_score, reports)
		elif name == "confusion_matrix":
			fn(cfg, y_true, y_pred, reports)
		elif name == "metrics":
			fn(cfg, y_true, y_pred, y_score, reports)

	if cfg.get("report") and bool(cfg.report.get("data_debug", False)):
		report_data_debug(cfg, result, reports)
