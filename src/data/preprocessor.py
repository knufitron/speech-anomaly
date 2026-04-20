from __future__ import annotations

import logging
import wave
from pathlib import Path

import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from utils.progress_bar import progress_bar

log = logging.getLogger(__name__)


def list_audio_files(root: Path, formats: set[str]) -> list[Path]:
	return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in formats]


def resample_to_wav(input_path: Path, output_path: Path, sr: int, channels: int) -> None:
	import ffmpeg

	(
		ffmpeg.input(str(input_path))
		.output(
			str(output_path),
			ar=sr,
			ac=channels,
			acodec="pcm_s16le",
		)
		.overwrite_output()
		.run(capture_stdout=True, capture_stderr=True)
	)


def _write_mono_pcm16_wav(path: Path, y: np.ndarray, sr: int) -> None:
	pcm = np.clip(np.ravel(y), -1.0, 1.0)
	pcm = (pcm * 32767.0).astype(np.int16)
	with wave.open(str(path), "wb") as w:
		w.setnchannels(1)
		w.setsampwidth(2)
		w.setframerate(int(sr))
		w.writeframes(pcm.tobytes())


def _apply_light_vad_trim(wav_path: Path, cfg: DictConfig) -> None:
	"""Trim low-energy silence at start/end (librosa.effects.trim); optional pad expands slice on raw waveform."""
	import librosa

	sr = int(cfg.audio.sr)
	y, _ = librosa.load(str(wav_path), sr=sr, mono=True)
	top_db = float(cfg.audio.get("vad_top_db", 20))
	pad_ms = float(cfg.audio.get("vad_pad_ms", 0))

	y_trim, index = librosa.effects.trim(y, top_db=top_db)
	if y_trim.size == 0:
		log.warning(
			"audio.vad: trim removed all samples for %s (top_db=%s); leaving file unchanged",
			wav_path.name,
			top_db,
		)
		return

	if pad_ms > 0:
		pad = int(sr * pad_ms / 1000.0)
		start = max(0, int(index[0]) - pad)
		end = min(len(y), int(index[1]) + pad)
		y_trim = y[start:end]

	_write_mono_pcm16_wav(wav_path, y_trim, sr)


def process_single_file(path: Path, output_path: Path, cfg: DictConfig) -> None:
	resample_to_wav(path, output_path, sr=int(cfg.audio.sr), channels=int(cfg.audio.channels))

	if cfg.audio.get("denoising", False):
		log.warning("audio.denoising=true is currently a placeholder; output is still resampled WAV.")
	if cfg.audio.get("vad", False):
		_apply_light_vad_trim(output_path, cfg)


def preprocess(cfg: DictConfig) -> None:
	# Hydra job chdir moves cwd; resolve paths relative to launch directory.
	path_in = Path(to_absolute_path(str(cfg.data.path_in)))
	path_out = Path(to_absolute_path(str(cfg.data.path_out)))
	path_out.mkdir(parents=True, exist_ok=True)

	formats = {str(x).lower() for x in cfg.audio.formats}
	files = sorted(list_audio_files(path_in, formats))
	log.info("Preprocessing %d files from %s -> %s", len(files), path_in, path_out)

	for i, src in enumerate(files, start=1):
		dst = path_out / src.with_suffix(".wav").name
		process_single_file(src, dst, cfg)
		progress_bar(i, len(files))

