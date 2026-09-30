from pathlib import Path
from functools import lru_cache

import torch

from Clause_Segmentation import FullClauseSegmentation
from mem_ranker.data_pipeline.embedding import EmbeddingModels
from mem_ranker.model import MemoryRanker


ROOT = Path(__file__).resolve().parent
DEFAULT_MEMORY_MODEL = ROOT / "mem_ranker" / "memory_ranker_model_best.pt"
DEFAULT_CLAUSE_MODEL = ROOT / "Clause_Seg" / "clause_segmentation_model_best.pt"
MEMORY_EMBEDDING_MODEL = "all_mpnet_base_v2"
MEMORY_EMBEDDING_DIMENSION = 768


@lru_cache(maxsize=None)
def _load_memory_model(model_path: str, device_name: str) -> MemoryRanker:
    model = MemoryRanker(input_dim=MEMORY_EMBEDDING_DIMENSION).to(device_name)
    state_dict = torch.load(model_path, map_location=device_name, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


class _FullMemoryRanker:
    def __init__(self, memory_model_path: Path, clause_model_path: Path, device: str):
        selected_device = device if device != "auto" else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if selected_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available.")

        self.device = torch.device(selected_device)
        if not memory_model_path.is_file():
            raise FileNotFoundError(f"Memory ranker checkpoint not found: {memory_model_path}")
        if not clause_model_path.is_file():
            raise FileNotFoundError(
                f"Clause segmentation checkpoint not found: {clause_model_path}"
            )

        self.segmenter = FullClauseSegmentation(str(clause_model_path), device=self.device)
        self.embeddings = EmbeddingModels(device=self.device)
        self.embeddings.load_model(MEMORY_EMBEDDING_MODEL)
        self.model = _load_memory_model(str(memory_model_path), str(self.device))

    def rank(self, text: str) -> list[dict[str, object]]:
        clauses = self.segmenter.segment(text)
        if not clauses:
            return []

        scores = self.score_clauses(clauses)
        ranked = [
            {"index": index, "clause": clause, "score": float(scores[index])}
            for index, clause in enumerate(clauses)
        ]
        return sorted(ranked, key=lambda item: float(item["score"]), reverse=True)

    def score_clauses(self, clauses: list[str]) -> list[float]:
        if not clauses:
            return []

        clause_embeddings = self.embeddings.encode(MEMORY_EMBEDDING_MODEL, clauses)
        embeddings = torch.from_numpy(clause_embeddings).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Convert raw logits into calibrated memorability scores.
            scores = torch.sigmoid(self.model(embeddings)).squeeze(0).cpu().tolist()
        return [float(score) for score in scores]

    def score_clause(self, clause: str) -> float:
        scores = self.score_clauses([clause])
        if not scores:
            raise ValueError("Cannot score an empty clause")
        return scores[0]

def main() -> None:
    ranker = _FullMemoryRanker(
        memory_model_path=DEFAULT_MEMORY_MODEL,
        clause_model_path=DEFAULT_CLAUSE_MODEL,
        device="auto",
    )
    PRESET_TEXT = """Once upon a time, a young girl found a small golden key under the old oak tree.
    She carried it home and discovered that it opened a hidden door in the wall of her attic.
    Inside, she found a room filled with dusty books, a tiny lantern, and a letter written by her grandmother.
    The letter explained that the family had always protected a secret garden hidden behind the house.
    When the girl stepped outside, she saw the garden bloom with silver flowers that glowed in the moonlight.
    She realized the key had not only opened a door, but also awakened a promise that had been waiting for her for years"""    
    scores = ranker.rank(PRESET_TEXT)
    print(scores)


if __name__ == "__main__":
    main()
