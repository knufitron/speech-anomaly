## Setup

Create virtual environment and install dependencies
```bash
./setup.sh
```

## Preprocess stage

```bash
python src/preprocess.py --data.path_in ./data/raw/SAVEE --data.path_out ./data/interim/SAVEE --audio.sr 16000 --audio.denoising false --audio.vad false
```

Configuration lives in `configs/preprocessor.yaml`; override keys from the CLI as in the example above.

| Parameter | Description |
| --- | --- |
| `data.path_in` | Root directory of source audio; files are discovered recursively. |
| `data.path_out` | Output directory for normalized WAV files (same basename, `.wav`). |
| `audio.formats` | Filename extensions to include (e.g. `.wav`, `.mp3`). |
| `audio.sr` | Target sample rate (Hz) passed to ffmpeg on decode/resample. |
| `audio.channels` | Target channel count for ffmpeg output (typically `1` for mono). |
| `audio.denoising` | Reserved; if `true`, a warning is logged and audio is still only resampled. |
| `audio.vad` | If `true`, run a light **VAD** after resampling: load the WAV with librosa, trim leading/trailing low-energy regions with `librosa.effects.trim`, then rewrite 16-bit mono PCM. If trim would remove the entire signal, the file is left unchanged and a warning is logged. |
| `audio.vad_top_db` | dB below the reference frame magnitude treated as silence for `trim` (higher values trim more aggressive silence; typical range about 20–30). Ignored when `audio.vad` is `false`. |
| `audio.vad_pad_ms` | Milliseconds to expand the trimmed interval **on each side** along the original waveform before cutting (reduces clipped speech at boundaries). Set to `0` to use the trim window only. Ignored when `audio.vad` is `false`. |

Example with VAD enabled:

```bash
python src/preprocess.py --audio.vad true --audio.vad_top_db 25 --audio.vad_pad_ms 150
```


## Feature extraction stage

#### Extract all features

```bash
python src/extract_features.py --data.dataset savee --data.features=all --data.path_in ./data/interim/SAVEE --data.path_out ./data/features/SAVEE/all.tsv
python src/extract_features.py --data.dataset ravdess --data.features=all --data.path_in ./data/interim/RAVDESS --data.path_out ./data/features/RAVDESS/all.tsv
```

#### Extract prosodic features only

```bash
python src/extract_features.py --data.dataset savee --data.features=prosodic --data.path_in ./data/interim/SAVEE --data.path_out ./data/features/SAVEE/prosodic.tsv
python src/extract_features.py --data.dataset ravdess --data.features=prosodic --data.path_in ./data/interim/RAVDESS --data.path_out ./data/features/RAVDESS/prosodic.tsv
```

#### Extract acoustic features only

```bash
python src/extract_features.py --data.dataset savee --data.features=acoustic --data.path_in ./data/interim/SAVEE --data.path_out ./data/features/SAVEE/acoustic.tsv
python src/extract_features.py --data.dataset ravdess --data.features=acoustic --data.path_in ./data/interim/RAVDESS --data.path_out ./data/features/RAVDESS/acoustic.tsv
```

#### Extract voice quality features only

```bash
python src/extract_features.py --data.dataset savee --data.features=voice_quality --data.path_in ./data/interim/SAVEE --data.path_out ./data/features/SAVEE/voice_quality.tsv
python src/extract_features.py --data.dataset ravdess --data.features=voice_quality --data.path_in ./data/interim/RAVDESS --data.path_out ./data/features/RAVDESS/voice_quality.tsv
```

#### For multiple custom feature groups extraction put feature group names delimited by comma inside square brackets and wrapped into quotes, otherwise Hydra interpretes comma ambiguously 

```bash
python src/extract_features.py --data.dataset savee --data.features='[prosodic,acoustic]' --data.path_in ./data/interim/SAVEE --data.path_out ./data/features/SAVEE/prosodic+acoustic.tsv
```

## Label column for feature TSVs

`scripts/label.py` adds a **`label`** column: rows where **`match-col` == `match-value`** get **`0`** (inlier); all other rows get **`1`**. Default output is **`<input_stem>_labeled<suffix>`** beside the input file if `--out` / `-o` is omitted.

```bash
python scripts/label.py -c emotion -v 1 -i data/features/SAVEE_VAD/all.tsv \
  -o data/features/SAVEE_VAD/all_labeled.tsv
python scripts/label.py --match-col emotion --match-value 1 --in data/features/SAVEE_VAD/all.tsv
# -> data/features/SAVEE_VAD/all_labeled.tsv
```

| Short | Long | Description |
| --- | --- | --- |
| `-c` | `--match-col` | Column to compare (must exist in the table). |
| `-v` | `--match-value` | Value treated as inlier (`label=0`); parsed as numeric when the column is numeric. |
| `-i` | `--in` | Input table path. |
| `-o` | `--out` | Output path (optional). |
| `-s` | `--sep` | Separator; default tab. |


## Batch hyperparameter search

`src/batch_runner.py` composes a **Cartesian grid** over models, per-model params (see `configs/batch/param_grids/default.yaml`), scalers, actor z-score on/off, datasets, and feature presets (`all`, `prosodic`, `acoustic`, `voice_quality`). It writes:

- **Session root** — `output/batch_runner/<date>/<time>/` (override with `batch.out_root=...`).
- **Per run** — subfolder named like `YYYY-MM-dd_HH-mm-ss.mmm` containing **`config_resolved.yaml`** (full resolved Hydra config), **`experiment.yaml`** (batch note + same config for reproducibility), `reports/` from the normal pipeline, and a copied **`metrics.json`** at the subfolder root.
- **`results.tsv`** in the session root: one row per completed run (`experiment_dir`, common columns, `model_params_json`, **accuracy**, **f1**, **mcc**, **tp/tn/fp/fn**, **roc_auc**, etc.) for sorting in a spreadsheet. It is **rewritten every `batch.results_flush_every` rows** (default 100; set `0` to flush only at the end) so a late crash keeps partial results.
- **`batch.skip_top_png`** (default true): write **`reports/top_features.tsv`** only and skip the matplotlib PNG (you can replot from the TSV offline). Also avoids GUI backends under heavy multiprocessing.

Reports during batch runs default to **`metrics` + `top`** only (so the **top-N chained** step can read `reports/top_features.tsv`). Disable chaining with `batch.after_all_top20=false`.

```bash
# Dry-run two primary jobs (no training)
python src/batch_runner.py dry_run=true batch.max_runs=2

# Limit models and cap grid size
python src/batch_runner.py batch.models='[logistic_regression]' batch.max_runs=10 batch.after_all_top20=false

# Run all models except a few (e.g. skip slow / optional XGB)
python src/batch_runner.py batch.models_skip='[xgb_classifier]' batch.max_runs=5
```

- **`batch.models`**: allowlist (`null` = every model in the registry that is installed, e.g. omits `xgb_classifier` if XGBoost is missing).
- **`batch.models_skip`**: removed after that (CLI or YAML list of names).

Review or edit **`configs/batch/param_grids/default.yaml`** for swept constructor args (e.g. `random_forest`: three `class_weight` presets × `max_depth` × `min_samples_leaf`). Models not listed there run with their YAML defaults only. **`experiment.scaler`** accepts **`none`** (identity scaling) in addition to **`robust`** and **`standard`**.

`metrics.json` (written by all runs) additionally includes **MCC** and scalar **tp**, **tn**, **fp**, **fn**, **accuracy**, and **f1** (anomaly-positive F1).

#### Collect results after the fact

`scripts/collect_batch_results.py` walks one or more batch (or run) trees, prefers `config_resolved.yaml` and falls back to `experiment.yaml`, and merges **config fields + metrics** into a single TSV. Traversal **follows symbolic links** to directories, so a root may list symlinked experiment folders. By default the output is **`./collected_results.tsv`** in the **current working directory**; use **`--out`** to set the path explicitly.

**Output columns (selection):** besides paths and `model`, `dataset`, `scaler`, `features_mode`, `features_json`, `model_params_json`, and the usual metric scalars, **`feature_group`** is a coarse preset label matching batch feature sets (`all`, `prosodic`, `acoustic`, `voice_quality`), or empty if the resolved `features` block does not match those fingerprints. **`is_top`** is **`False`** for normal runs, or the integer **`N`** when the run is a batch **top-N chained** step (detected via `chained_from=` in the `experiment.yaml` header); in that case `feature_group` is **`all`** and **`N`** is the number of explicit columns (the importance subset size).

```bash
python scripts/collect_batch_results.py --root output/batch_runner/<date>/<time-a>
python scripts/collect_batch_results.py --root run-a run-b run-c
python scripts/collect_batch_results.py --root ./trees/foo --out ./summary.tsv
python scripts/collect_batch_results.py --root ./out/batch --require-metrics
# default out: ./collected_results.tsv (CWD); duplicate experiment dirs across roots are merged once
```

| Long | Description |
| --- | --- |
| `--root` | One or more directories to scan (space-separated list). Each is searched recursively for `config_resolved.yaml` / `experiment.yaml`, following symlinked subdirectories. |
| `--out` | Output TSV path. Default: **`collected_results.tsv`** in the current working directory. |
| `--require-metrics` | Skip run folders that have no `metrics.json` and no `reports/metrics.json`. |

#### Monitor batch progress

Run inside `batch_runner/{date}/{time}`:

```bash
watch -n 60 "find . -type f -name config_resolved.yaml -exec grep -A 1 model {} + | grep -v 'model\|--' | cut -f 4 -d ' ' | sort | uniq -c"
```

See something like

```
    432 elliptic_envelope
    432 isolation_forest
    432 local_outlier_factor
   3960 logistic_regression
    576 one_class_svm
     22 random_forest
```


# Run experiments

#### Replay from a saved config (`config_resolved.yaml`)

Each run writes a fully resolved tree to **`config_resolved.yaml`** in its output directory (batch runs also keep **`experiment.yaml`** with a comment header). You can replay or fork that setup without going through Hydra’s `runner` defaults composition:

```bash
python src/run_experiment.py --config path/to/run/config_resolved.yaml
python src/run_experiment.py -c path/to/run/config_resolved.yaml
```

By default a **new** timestamped directory under **`output/runner/<date>/`** is used (the path stored in the YAML is **not** reused, to avoid overwriting artifacts). To choose the output directory explicitly, pass a dotted override:

```bash
python src/run_experiment.py -c path/to/config_resolved.yaml --run.output_dir path/to/new_run_dir
```

Additional arguments are turned into **Hydra-style overrides** and merged into the loaded tree. With **`--config`**, avoid whole-group switches such as **`--model isolation_forest`** or **`model=isolation_forest`**: they replace nested dict nodes with strings and break the saved layout. Use **dotted** keys instead (e.g. **`model.name=isolation_forest`**, **`data.name=savee`**, **`--experiment.scaler robust`**). Exception: **`--report basic`** / **`--report full`** are supported by reloading this repo’s **`configs/report/basic.yaml`** or **`full.yaml`** into **`report`**.

| Option | Description |
| --- | --- |
| `-c` | Short form of **`--config`**. |
| `--config` | Path to **`config_resolved.yaml`** (recommended) or **`experiment.yaml`**. |
| `run.output_dir=...` | Optional; merged literally. If omitted, the script assigns a new **`output/runner/...`** path under the project root (the directory containing **`configs/`**). |
| Other overrides | Prefer dotted keys (**`dry_run=true`**, **`model.params.max_depth=4`**, **`--report roc,umap`**, etc.). |

#### Dry-run

```bash
python src/run_experiment.py --dry-run
```

#### Specify Model constructor parameters

```sh
python src/run_experiment.py --model one_class_svm --model.params.nu 0.1 --model.params.kernel rbf
```

#### CLI override

```bash
python src/run_experiment.py --data savee --model isolation_forest --model.params.n_estimators=300
```

#### Dataset `label` column

Feature tables must include **`label`**: `0` = inlier/normal, `1` = outlier/anomaly. Build this in preprocessing or when exporting TSV; the runner does not derive labels from `emotion` anymore. **Unsupervised / one-class** models are fit only on training rows with **`label == 0`**; **`data.filter`** is only for row-wise dataset restriction (independent of that training subset).

#### Optional row filter on the loaded table

After `load_tabular`, keep only rows where `data.filter.column` is in `data.filter.value` (scalar or list). Examples:

```bash
python src/run_experiment.py --data.filter.column emotion --data.filter.value 1
python src/run_experiment.py --data.filter.column emotion --data.filter.value='[1,2]'
```

#### Scaler override (`robust` / `standard`)

Scaling is configured as `experiment.scaler` (see `configs/experiment/anomaly.yaml`). Override from the CLI:

```bash
python src/run_experiment.py --experiment.scaler robust
python src/run_experiment.py --experiment.scaler standard
python src/run_experiment.py --experiment.scaler=standard
```

The `anomaly` in `experiment/anomaly.yaml` is the Hydra **preset name** (`experiment=anomaly`), not a nested config key — use `experiment.scaler`, not `experiment.anomaly.scaler`.

#### Enable speaker-conditioned Z-score normalization

```bash
python src/run_experiment.py --experiment.actor_zscore.enabled true --experiment.actor_zscore.eps 1e-5
```

#### Reports (default: all)

The `report` config group (`basic` / `full`) defaults to `include: all` (metrics, confusion_matrix, roc, umap).

Pick specific reports (comma-separated; quote in zsh if needed):

```bash
python src/run_experiment.py --report roc,umap
python src/run_experiment.py --report 'roc,umap'
```

Select only the Hydra report preset (no comma):

```bash
python src/run_experiment.py --report full
```

#### Top features

Example (using RAVDESS data, XGBoost, top 20 important features):

```bash
python src/run_experiment.py --data ravdess --model xgb_classifier --report basic --experiment.scaler robust --report.data_debug --features top20
```

- `--features top20` uses the high-impact subset defined in `configs/features/top20.yaml`.
- See `configs/features/` for other feature sets and `configs/model/` for model configs.
- Use CLI overrides or edit config YAMLs for rapid experiments.


#### Train/test split by group (e.g. actor)

Leaks less speaker identity across split: `GroupShuffleSplit` when `data.groupby` is set. `data.split.stratify` is ignored in that mode.

```bash
python src/run_experiment.py --data.groupby actor
```

#### Logging level override

```bash
python src/run_experiment.py --logging.level DEBUG
```

#### Enable Weights & Biases logging

```bash
python src/run_experiment.py --wandb.enabled true --wandb.project test-proj
```

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Project structure

```
project/
│
├── src/
│   ├── core/
│   │   ├── cli_overrides.py
│   │   ├── runner.py
│   ├── run_experiment.py
│   ├── batch_runner.py
│   ├── preprocess.py
│   ├── extract_features.py
│   ├── batch/
│   │   ├── grid.py
│
│   ├── models/
│   │   ├── registry.py
│   │   ├── factory.py
│
│   ├── features/
│   │   ├── base.py
│   │   ├── feature_extractor.py
│   │   ├── factory.py
│
│   ├── data/
│   │   ├── loader.py
│   │   ├── dataset.py
│   │   ├── preprocessor.py
│
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── reports.py
│
│   ├── utils/
│   │   ├── logging.py
│   │   ├── progress_bar.py
│   │   ├── wandb_logger.py
│
├── configs/
│   ├── batch_runner.yaml
│   ├── batch/
│   │   └── param_grids/
│   │       └── default.yaml
│   ├── runner.yaml
│   ├── preprocessor.yaml
│   ├── feature_extractor.yaml
│   ├── logging/
│   │   ├── default.yaml
│
│   ├── model/
│   │   ├── one_class_svm.yaml
│   │   ├── isolation_forest.yaml
│   │   ├── logistic_regression.yaml
│   │   ├── random_forest.yaml
│   │   ├── xgb_classifier.yaml
│   │   ├── local_outlier_factor.yaml
│   │   ├── elliptic_envelope.yaml
│
│   ├── data/
│   │   ├── ravdess.yaml
│   │   ├── savee.yaml
│
│   ├── features/
│   │   ├── acoustic.yaml
│   │   ├── voice_quality.yaml
│
│   ├── experiment/
│   │   ├── anomaly.yaml
│
│   ├── report/
│       ├── basic.yaml
│
├── data/
│   ├── ravdess.tsv
│   ├── savee.tsv
│
├── outputs/
│
├── scripts/
│   ├── distinct_column_values.py
│   ├── generate_synthetic_data.py
│   └── abel.py
│
├── pyproject.toml
└── README.md
```
