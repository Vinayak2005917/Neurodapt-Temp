"""Batched pairwise memorability rewards for GRPO training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from mem_ranker.model import MemoryRanker


EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "mem_ranker"
    / "data_pipeline"
    / "models"
    / "all_mpnet_base_v2"
)
EMBEDDING_DIMENSION = 768
# Valid pairwise memorability rewards are bounded to [-1, 1].
LENGTH_REJECTION_REWARD = -1.5
LENGTH_PENALTY_SCALE = 1.0
INVALID_SCORE_WEIGHT = 0.25


class MemoryRewardScorer:
    """Load only the MPNet encoder and ranker needed for reward scoring."""

    def __init__(self, memory_model_path: Path, device: str = "cpu") -> None:
        selected_device = device if device != "auto" else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if selected_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for reward scoring but is unavailable")

        self.device = torch.device(selected_device)
        embedding_source = (
            str(EMBEDDING_MODEL_PATH)
            if EMBEDDING_MODEL_PATH.is_dir()
            else EMBEDDING_MODEL_NAME
        )
        self.embedding_model = SentenceTransformer(
            embedding_source,
            device=str(self.device),
        )
        self.model = MemoryRanker(input_dim=EMBEDDING_DIMENSION)
        state_dict = torch.load(
            memory_model_path,
            map_location="cpu",
            weights_only=True,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.original_embedding_cache: dict[str, np.ndarray] = {}

    def cache_originals(self, originals: list[str]) -> None:
        """Encode each distinct original once and keep its vector in RAM."""
        unique_originals = list(dict.fromkeys(text.strip() for text in originals if text.strip()))
        uncached = [
            text for text in unique_originals if text not in self.original_embedding_cache
        ]
        if not uncached:
            return

        embeddings = self.embedding_model.encode(
            uncached,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        self.original_embedding_cache.update(
            (text, np.asarray(embedding, dtype=np.float32))
            for text, embedding in zip(uncached, embeddings)
        )

    def score_pairs(self, originals: list[str], rewrites: list[str]) -> list[float]:
        if len(originals) != len(rewrites):
            raise ValueError("originals and rewrites must have equal lengths")
        if not originals:
            return []

        self.cache_originals(originals)
        embeddings = self.embedding_model.encode(
            rewrites,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        pair_embeddings = np.stack(
            [
                np.stack((self.original_embedding_cache[original.strip()], rewrite_embedding))
                for original, rewrite_embedding in zip(originals, embeddings)
            ]
        )
        embeddings_tensor = torch.as_tensor(
            np.asarray(pair_embeddings, dtype=np.float32),
            device=self.device,
        )

        with torch.inference_mode():
            scores = torch.sigmoid(self.model(embeddings_tensor))
            differences = scores[:, 1] - scores[:, 0]
        return differences.cpu().tolist()


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion.strip()
    if isinstance(completion, list) and completion:
        last_message = completion[-1]
        if isinstance(last_message, dict):
            return str(last_message.get("content", "")).strip()
    if isinstance(completion, dict):
        return str(completion.get("content", "")).strip()
    return str(completion).strip()


def reward_function_factory(scorer: MemoryRewardScorer):
    """Create a batched reward callback with a strict 150% word-count cap."""

    def reward_function(
        prompts: list[Any],
        completions: list[Any],
        target_clause: list[str],
        **kwargs: Any,
    ) -> list[float]:
        del prompts, kwargs
        rewards = [LENGTH_REJECTION_REWARD] * len(completions)
        score_indices: list[int] = []
        originals: list[str] = []
        rewrites: list[str] = []
        word_counts: list[tuple[int, int]] = []

        for index, (completion, original) in enumerate(zip(completions, target_clause)):
            original_text = str(original).strip()
            rewritten = _completion_text(completion)
            original_word_count = len(original_text.split())
            rewritten_word_count = len(rewritten.split())

            if not original_word_count or not rewritten_word_count:
                rewards[index] = LENGTH_REJECTION_REWARD - 1.0
                continue

            score_indices.append(index)
            originals.append(original_text)
            rewrites.append(rewritten)
            word_counts.append((original_word_count, rewritten_word_count))

        if score_indices:
            scores = scorer.score_pairs(originals, rewrites)
            for index, score, (original_count, rewritten_count) in zip(
                score_indices, scores, word_counts
            ):
                # Valid ranker scores are in [-1, 1]. This penalty keeps every
                # overlength reward below -1 while retaining ranker variation.
                # Integer comparison avoids rounding: reject above 150%.
                if 2 * rewritten_count > 3 * original_count:
                    excess_ratio = rewritten_count / (1.5 * original_count) - 1.0
                    rewards[index] = (
                        LENGTH_REJECTION_REWARD
                        - LENGTH_PENALTY_SCALE * excess_ratio
                        + INVALID_SCORE_WEIGHT * float(score)
                    )
                else:
                    rewards[index] = float(score)

        return rewards

    return reward_function
