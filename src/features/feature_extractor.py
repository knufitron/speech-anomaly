from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from data.filename_parsers import parse_filename
from utils.progress_bar import progress_bar

log = logging.getLogger(__name__)


def parse_feature_groups(raw: object) -> list[str]:
	if isinstance(raw, str):
		if raw.lower() == "all":
			return ["prosodic", "acoustic", "voice_quality"]
		return [s.strip() for s in raw.split(",") if s.strip()]
	if OmegaConf.is_list(raw):
		return [str(x).strip() for x in OmegaConf.to_container(raw, resolve=True) if str(x).strip()]
	if isinstance(raw, (list, tuple)):
		return [str(x).strip() for x in raw if str(x).strip()]
	return ["prosodic", "acoustic", "voice_quality"]


def extract_prosodic_and_acoustic(audio_path: str, sr: int, groups: set[str]) -> dict[str, float]:
	import librosa
	import parselmouth
	from parselmouth.praat import call

	y, sr = librosa.load(audio_path, sr=sr)
	features: dict[str, float] = {}

	# Prosodic only: F0, energy envelope, duration. Not pulled in when only acoustic is requested.
	if "prosodic" in groups:
		snd = parselmouth.Sound(audio_path)
		pitch = call(snd, "To Pitch", 0.0, 75, 600)
		pitch_values = pitch.selected_array["frequency"]
		pitch_values = pitch_values[pitch_values > 0]
		features["F0_mean"] = float(np.mean(pitch_values)) if len(pitch_values) else 0.0
		features["F0_std"] = float(np.std(pitch_values)) if len(pitch_values) else 0.0
		rms = librosa.feature.rms(y=y)[0]
		features["energy_mean"] = float(np.mean(rms))
		features["energy_std"] = float(np.std(rms))
		features["duration"] = float(librosa.get_duration(y=y, sr=sr))

	if "acoustic" in groups:
		mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
		mfcc_delta = librosa.feature.delta(mfcc)
		mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
		for i in range(13):
			features[f"mfcc_{i + 1}_mean"] = float(np.mean(mfcc[i]))
			features[f"mfcc_{i + 1}_std"] = float(np.std(mfcc[i]))
			features[f"mfcc_delta_{i + 1}_mean"] = float(np.mean(mfcc_delta[i]))
			features[f"mfcc_delta_{i + 1}_std"] = float(np.std(mfcc_delta[i]))
			features[f"mfcc_delta2_{i + 1}_mean"] = float(np.mean(mfcc_delta2[i]))
			features[f"mfcc_delta2_{i + 1}_std"] = float(np.std(mfcc_delta2[i]))

		centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
		bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
		rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
		zcr = librosa.feature.zero_crossing_rate(y)[0]
		features["spec_centroid_mean"] = float(np.mean(centroid))
		features["spec_bandwidth_mean"] = float(np.mean(bandwidth))
		features["spec_rolloff_mean"] = float(np.mean(rolloff))
		features["zcr_mean"] = float(np.mean(zcr))

		# Spectral contrast: typically 7 frequency bands
		contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
		contrast_mean = np.mean(contrast, axis=1)
		for i, val in enumerate(contrast_mean):
			features[f"spectral_contrast_{i + 1}_mean"] = float(val)
		# Chroma STFT: 12 bins (C, C#, D, etc.)
		chroma = librosa.feature.chroma_stft(y=y, sr=sr)
		chroma_mean = np.mean(chroma, axis=1)
		for i, val in enumerate(chroma_mean):
			features[f"chroma_{i + 1}_mean"] = float(val)

	return features


def extract_voice_quality(audio_path: str) -> dict[str, float]:
	import parselmouth
	from parselmouth.praat import call

	snd = parselmouth.Sound(audio_path)
	point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)
	jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
	shimmer = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
	harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
	hnr = call(harmonicity, "Get mean", 0, 0)
	return {"jitter": float(jitter), "shimmer": float(shimmer), "hnr": float(hnr)}


def extract_features(cfg: DictConfig) -> None:
	path_in = Path(to_absolute_path(str(cfg.data.path_in)))
	path_out = Path(to_absolute_path(str(cfg.data.path_out)))
	path_out.parent.mkdir(parents=True, exist_ok=True)

	groups = set(parse_feature_groups(cfg.data.get("features", "all")))
	files = sorted([p for p in path_in.rglob("*") if p.is_file() and p.suffix.lower() in {".wav", ".mp3"}])
	log.info("Extracting features (%s) for %d files from %s", ",".join(sorted(groups)), len(files), path_in)

	rows: list[dict[str, object]] = []
	for i, p in enumerate(files, start=1):
		emotion, statement, actor = parse_filename(p.name, str(cfg.data.dataset))
		row: dict[str, object] = {}
		row.update(extract_prosodic_and_acoustic(str(p), int(cfg.audio.sr), groups))
		if "voice_quality" in groups:
			row.update(extract_voice_quality(str(p)))
		row["file"] = str(p)
		row["actor"] = actor
		row["statement"] = statement
		row["emotion"] = emotion
		rows.append(row)
		progress_bar(i, len(files))

	df = pd.DataFrame(rows)
	df.to_csv(path_out, sep="\t", index=False)
	log.info("Wrote features to %s (%d rows, %d columns)", path_out, len(df), len(df.columns))

