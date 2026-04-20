from __future__ import annotations

import re
from pathlib import Path

# Example: KL_a1.wav
# KL - actor
# a - emotion
# 1 - statement
SAVEE_RE = re.compile(r"^([A-Z]{2})_([a-z]{1,2})(\d+)")

SAVEE_EMOTION_MAP: dict[str, str] = {
	"a": "5",   # anger
	"c": "2",   # calm (not used in SAVEE, kept for compatibility)
	"d": "7",   # disgust
	"f": "6",   # fear
	"h": "3",   # happiness
	"n": "1",   # neutral
	"sa": "4",  # sadness
	"su": "8",  # surprise
}


def parse_ravdess_filename(path_or_name: str) -> tuple[str, str, str]:
	name = Path(path_or_name).name
	# 03-01-05-02-01-02-09.wav -> emotion, statement, actor
	_, _, emotion, _, statement, _, actor = name.split(".")[0].split("-")
	return str(int(emotion)), str(int(statement)), str(int(actor))


def parse_savee_filename(path_or_name: str) -> tuple[str, str, str]:
	name = Path(path_or_name).name
	match = SAVEE_RE.match(name)
	if not match:
		raise ValueError(f'Cannot parse SAVEE filename: "{name}"')
	actor, emotion_code, sentence = match.groups()
	emotion = SAVEE_EMOTION_MAP[emotion_code]
	return emotion, str(int(sentence)), actor


def parse_filename(path_or_name: str, dataset: str) -> tuple[str, str, str]:
	ds = dataset.lower()
	if ds == "ravdess":
		return parse_ravdess_filename(path_or_name)
	if ds == "savee":
		return parse_savee_filename(path_or_name)
	raise ValueError(f"Unsupported dataset '{dataset}'. Expected one of: ravdess, savee")

