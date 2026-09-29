from __future__ import annotations

import csv
import sys
from itertools import islice
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


# Configuration.  Command-line arguments can override the input and output.
WINDOW_SIZE = 50
STRIDE = 50
MIN_WORDS = 50
MAX_ARTICLES = 10000
INPUT_DIR = Path(__file__).resolve().parent / "processed_data"
OUTPUT_PATH = Path(__file__).resolve().parent / "preprocessed_grpo.pt"
DEVICE = "auto"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from Memory_Ranker import _FullMemoryRanker


def _window_starts(word_count: int) -> list[int]:
	if word_count < max(WINDOW_SIZE, MIN_WORDS):
		return []
	return list(range(0, word_count - WINDOW_SIZE + 1, STRIDE))


def _target_indices(clause_count: int) -> list[int]:
	candidates = list(range(1, clause_count - 1))
	if not candidates:
		return []
	center = (clause_count - 1) / 2
	return sorted(candidates, key=lambda index: (abs(index - center), index))


def _rank_window(ranker: _FullMemoryRanker, raw_text: str) -> list[dict[str, Any]]:
	ranked = ranker.rank(raw_text)
	ordered = sorted(ranked, key=lambda item: int(item["index"]))
	return [
		{"text": str(item["clause"]), "score": float(item["score"])}
		for item in ordered
	]


def _count_articles(path: Path) -> int:
	with path.open("r", newline="", encoding="utf-8") as csv_file:
		reader = csv.DictReader(csv_file)
		if not reader.fieldnames or "text" not in reader.fieldnames:
			raise ValueError("CSV does not contain a 'text' column")
		return min(MAX_ARTICLES, sum(1 for _ in reader))


def _process_csv(path: Path,ranker: _FullMemoryRanker,article_id_start: int,examples: list[dict[str, Any]],stats: dict[str, int],total_articles: int) -> int:
	article_id = article_id_start
	try:
		with path.open("r", newline="", encoding="utf-8") as csv_file:
			reader = csv.DictReader(csv_file)
			if not reader.fieldnames or "text" not in reader.fieldnames:
				raise ValueError("CSV does not contain a 'text' column")

			for row in tqdm(islice(reader, total_articles),desc=f"Processing {path.name}",total=total_articles,unit="article",leave=True):
				current_id = article_id
				article_id += 1
				try:
					text = row.get("text")
					if not isinstance(text, str) or not text.strip():
						stats["articles_skipped"] += 1
						continue

					words = text.split()
					starts = _window_starts(len(words))
					if not starts:
						stats["articles_skipped"] += 1
						continue

					stats["articles_processed"] += 1
					seen_windows: set[tuple[int, int]] = set()
					for start in starts:
						end = start + WINDOW_SIZE
						if (start, end) in seen_windows:
							continue
						seen_windows.add((start, end))
						raw_text = " ".join(words[start:end])
						clauses = _rank_window(ranker, raw_text)
						stats["windows"] += 1
						stats["total_clauses"] += len(clauses)

						if len(clauses) < 3:
							continue
						for target_index in _target_indices(len(clauses)):
							examples.append(
								{
									"article_id": current_id,
									"source_file": path.name,
									"window_start_word": start,
									"window_end_word": end,
									"raw_text": raw_text,
									"clauses": clauses,
									"target_index": target_index,
								}
							)
							stats["training_examples"] += 1
				except Exception as exc:  # One malformed article must not stop a shard.
					stats["articles_skipped"] += 1
					tqdm.write(f"Skipping article {current_id} in {path.name}: {exc}")
	except Exception as exc:
		tqdm.write(f"Skipping file {path}: {exc}")
	return article_id


def main() -> None:
	import argparse

	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
	parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
	args = parser.parse_args()

	csv_files = sorted(args.input_dir.glob("*.csv"))
	if not csv_files:
		raise FileNotFoundError(f"No CSV files found in {args.input_dir}")

	ranker = _FullMemoryRanker(
		memory_model_path=PROJECT_ROOT / "mem_ranker" / "memory_ranker_model_best.pt",
		clause_model_path=PROJECT_ROOT / "Clause_Seg" / "clause_segmentation_model_best.pt",
		device=DEVICE,
	)
	examples: list[dict[str, Any]] = []
	stats = {
		"articles_processed": 0,
		"articles_skipped": 0,
		"windows": 0,
		"training_examples": 0,
		"total_clauses": 0,
	}
	next_article_id = 0
	for csv_path in csv_files:
		next_article_id = _process_csv(
			csv_path,
			ranker,
			next_article_id,
			examples,
			stats,
			_count_articles(csv_path),
		)


	stats["average_clauses_per_window"] = (stats["total_clauses"] / stats["windows"] if stats["windows"] else 0.0)
	stats["average_words_per_window"] = float(WINDOW_SIZE if stats["windows"] else 0)
	output = {
		"schema_version": 1,
		"window_size": WINDOW_SIZE,
		"stride": STRIDE,
		"examples": examples,
		"statistics": stats,
	}
	args.output.parent.mkdir(parents=True, exist_ok=True)
	torch.save(output, args.output)
	print(f"Saved {len(examples)} training examples to {args.output}")
	print(f"Statistics: {stats}")


if __name__ == "__main__":
	main()