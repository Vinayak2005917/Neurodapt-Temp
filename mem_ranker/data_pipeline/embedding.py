from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
PAIRED_DATASET_PATH = HERE / "Breithaupt_Cleaned.csv"
MODEL_DIR = HERE / "models"

MODEL_IDS = {
    "all_mpnet_base_v2": "sentence-transformers/all-mpnet-base-v2",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}
OUTPUT_PATHS = {
    name: HERE / f"embedded_stories_{name}.pt" for name in MODEL_IDS
}


class EmbeddingModels:
    """Load the requested local encoders once, then encode by backend name."""

    def __init__(self, *, device: str | torch.device | None = None) -> None:
        self.device = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.local_models: dict[str, SentenceTransformer] = {}
        self._load_sentence_transformers()

    def _load_sentence_transformers(self) -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        for name, model_id in tqdm(
            MODEL_IDS.items(),
            desc="Loading embedding models",
            unit="model",
        ):
            local_dir = MODEL_DIR / name
            if local_dir.exists() and any(local_dir.iterdir()):
                source = str(local_dir)
            else:
                # from_pretrained downloads the complete model before returning.
                source = model_id
            print(f"Loading {model_id}...")
            model = SentenceTransformer(source, device=self.device)
            if source == model_id:
                try:
                    model.save(str(local_dir))
                except OSError as exc:
                    print(f"Warning: could not cache {model_id} at {local_dir}: {exc}")
            self.local_models[name] = model
            print(f"Ready: {model_id} ({local_dir})")

    def encode(self, model_name: str, texts: list[str]) -> np.ndarray:
        if model_name not in MODEL_IDS:
            raise KeyError(f"Unknown embedding model: {model_name}")
        if not texts:
            dimension = self.dimension(model_name)
            return np.empty((0, dimension), dtype=np.float32)

        values = np.asarray(
            self.local_models[model_name].encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        return values

    def dimension(self, model_name: str) -> int:
        return int(self.local_models[model_name].get_sentence_embedding_dimension())


@lru_cache(maxsize=1)
def _get_models() -> EmbeddingModels:
    return EmbeddingModels()


def embed(text: str, model_name: str = "minilm") -> np.ndarray:
    """Convenience single-text helper used by inference scripts."""
    return _get_models().encode(model_name, [text])[0]


def embed_all_stories() -> dict[str, Path]:
    """Create one embedded-story file per model, preserving the existing schema."""
    if not PAIRED_DATASET_PATH.exists():
        raise FileNotFoundError(f"Source stories not found: {PAIRED_DATASET_PATH}")

    # Preflight all three backends before segmenting/embedding any source story.
    models = EmbeddingModels()

    import csv
    import sys

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from Clause_Segmentation import FullClauseSegmentation

    clause_model_path = PROJECT_ROOT / "Clause_Seg" / "clause_segmentation_model_best.pt"
    if not clause_model_path.exists():
        raise FileNotFoundError(f"Clause segmentation checkpoint not found: {clause_model_path}")
    device = torch.device(models.device)
    segmenter = FullClauseSegmentation(str(clause_model_path), device=device)

    with PAIRED_DATASET_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No source stories found in {PAIRED_DATASET_PATH}")

    embedded_by_model: dict[str, list[dict[str, Any]]] = {
        name: [] for name in MODEL_IDS
    }
    for story_id, row in enumerate(tqdm(rows, desc="Segmenting and embedding stories")):
        original_text = (row.get("Original text") or "").strip()
        retold_texts = [
            (row.get(column) or "").strip()
            for column in ("G1 Story", "G2 Retelling", "G3 Retelling")
        ]
        original_clauses = segmenter.segment(original_text)
        retold_clauses = [segmenter.segment(text) for text in retold_texts]
        all_texts = original_clauses + [
            clause for group in retold_clauses for clause in group
        ]
        encoded = {name: models.encode(name, all_texts) for name in MODEL_IDS}

        for name in MODEL_IDS:
            vectors = encoded[name]
            original_count = len(original_clauses)
            cursor = original_count
            retold_vectors = []
            for clauses in retold_clauses:
                next_cursor = cursor + len(clauses)
                retold_vectors.append(
                    torch.from_numpy(vectors[cursor:next_cursor].copy())
                    if clauses
                    else torch.empty((0, models.dimension(name)), dtype=torch.float32)
                )
                cursor = next_cursor

            embedded_by_model[name].append(
                {
                    "story_id": story_id,
                    "original_clauses": original_clauses,
                    "retold_clauses": retold_clauses,
                    "original_embeddings": torch.from_numpy(
                        vectors[:original_count].copy()
                    ),
                    "retold_embeddings": retold_vectors,
                }
            )

    for name, stories in embedded_by_model.items():
        output_path = OUTPUT_PATHS[name]
        torch.save(stories, output_path)
        print(f"Saved {len(stories)} stories to {output_path}")
    return OUTPUT_PATHS


if __name__ == "__main__":
    embed_all_stories()
