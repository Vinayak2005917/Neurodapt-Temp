"""Batched pairwise memorability rewards for GRPO training."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mem_ranker.data_pipeline.embedding import EmbeddingModels
from mem_ranker.model import MemoryRanker


EMBEDDING_DIMENSION = 768
MIN_LENGTH_RATIO = 0.70
MAX_LENGTH_RATIO = 1.50
LENGTH_REJECTION_REWARD = -1.5
LENGTH_PENALTY_SCALE = 1.0
INITIAL_REWARD_WEIGHT = 1.0 / 3.0
MIN_LENGTH_WEIGHT = 0.04
MAX_LENGTH_WEIGHT = INITIAL_REWARD_WEIGHT
DEFAULT_ADAPTATION_WINDOW = 32
DEFAULT_WEIGHT_STEP = 0.05
VIOLATION_RATE_TO_INCREASE = 0.10
BOUNDARY_RATE_TO_INCREASE = 0.50
BOUNDARY_RATE_TO_DECREASE = 0.25


@dataclass
class _AdaptiveRewardWeights:
    memory: float = INITIAL_REWARD_WEIGHT
    length: float = INITIAL_REWARD_WEIGHT
    similarity: float = INITIAL_REWARD_WEIGHT
    samples: int = 0
    violations: int = 0
    boundary_hits: int = 0

    def observe(self, ratio: float, window: int, step: float) -> None:
        self.samples += 1
        if ratio < MIN_LENGTH_RATIO or ratio > MAX_LENGTH_RATIO:
            self.violations += 1
        if ratio <= MIN_LENGTH_RATIO * 1.05 or ratio >= MAX_LENGTH_RATIO * 0.95:
            self.boundary_hits += 1

        if self.samples < window:
            return

        violation_rate = self.violations / self.samples
        boundary_rate = self.boundary_hits / self.samples
        if (
            violation_rate >= VIOLATION_RATE_TO_INCREASE
            or boundary_rate >= BOUNDARY_RATE_TO_INCREASE
        ):
            self._increase_length_weight(step)
        elif violation_rate == 0 and boundary_rate <= BOUNDARY_RATE_TO_DECREASE:
            self._decrease_length_weight(step)

        self.samples = 0
        self.violations = 0
        self.boundary_hits = 0

    def _increase_length_weight(self, step: float) -> None:
        new_length = min(MAX_LENGTH_WEIGHT, self.length + step)
        transferred = new_length - self.length
        if not transferred:
            return
        self.length = new_length
        self.memory -= transferred / 2
        self.similarity -= transferred / 2

    def _decrease_length_weight(self, step: float) -> None:
        new_length = max(MIN_LENGTH_WEIGHT, self.length - step)
        transferred = self.length - new_length
        if not transferred:
            return
        self.length = new_length
        self.memory += transferred / 2
        self.similarity += transferred / 2


@lru_cache(maxsize=None)
def _load_memory_ranker(memory_model_path: str, device_name: str) -> MemoryRanker:
    model = MemoryRanker(input_dim=EMBEDDING_DIMENSION).to(device_name)
    state_dict = torch.load(memory_model_path, map_location=device_name, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


class MemoryRewardScorer:
    """Load only the MPNet encoder and ranker needed for reward scoring."""

    def __init__(self, memory_model_path: Path, device: str = "cpu") -> None:
        selected_device = device if device != "auto" else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if selected_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for reward scoring but is unavailable")

        self.device = torch.device(selected_device)
        self.embedding_model = EmbeddingModels(device=self.device).load_model(
            "all_mpnet_base_v2"
        )
        self.model = _load_memory_ranker(str(memory_model_path), str(self.device))
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

    def score_components(
        self,
        originals: list[str],
        rewrites: list[str],
    ) -> list[tuple[float, float]]:
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
        ranker_scores = ((differences + 1.0) / 2.0).cpu().tolist()
        original_vectors = embeddings_tensor[:, 0]
        rewrite_vectors = embeddings_tensor[:, 1]
        similarities = ((original_vectors * rewrite_vectors).sum(dim=1) + 1.0) / 2.0
        return list(zip(ranker_scores, similarities.cpu().tolist()))

    def score_pairs(self, originals: list[str], rewrites: list[str]) -> list[float]:
        """Return the normalized memory-ranker component for compatibility."""
        return [
            ranker_score
            for ranker_score, _ in self.score_components(originals, rewrites)
        ]


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


def _length_reward(original_count: int, rewritten_count: int) -> tuple[float, bool, float, float]:
    ratio = rewritten_count / original_count
    if MIN_LENGTH_RATIO <= ratio <= MAX_LENGTH_RATIO:
        return 1.0, True, ratio, 0.0

    if ratio < MIN_LENGTH_RATIO:
        excess_ratio = MIN_LENGTH_RATIO / max(ratio, 1e-9) - 1.0
    else:
        excess_ratio = ratio / MAX_LENGTH_RATIO - 1.0
    rejection_penalty = LENGTH_REJECTION_REWARD - LENGTH_PENALTY_SCALE * excess_ratio
    return 0.0, False, ratio, rejection_penalty


def reward_function_factory(
    scorer: MemoryRewardScorer,
    adaptation_window: int = DEFAULT_ADAPTATION_WINDOW,
    weight_step: float = DEFAULT_WEIGHT_STEP,
):
    """Create a batched reward callback with adaptive component weights."""
    if adaptation_window <= 0:
        raise ValueError("adaptation_window must be greater than zero")
    if weight_step <= 0:
        raise ValueError("weight_step must be greater than zero")

    weights = _AdaptiveRewardWeights()

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

            if not original_word_count:
                rewards[index] = LENGTH_REJECTION_REWARD - 1.0
                continue
            if not rewritten_word_count:
                rewards[index] = LENGTH_REJECTION_REWARD - 1.0
                weights.observe(0.0, adaptation_window, weight_step)
                continue

            score_indices.append(index)
            originals.append(original_text)
            rewrites.append(rewritten)
            word_counts.append((original_word_count, rewritten_word_count))

        if score_indices:
            scores = scorer.score_components(originals, rewrites)
            for index, (memory_score, similarity_score), (original_count, rewritten_count) in zip(
                score_indices, scores, word_counts
            ):
                length_score, length_valid, ratio, rejection_penalty = _length_reward(
                    original_count,
                    rewritten_count,
                )
                weights.observe(ratio, adaptation_window, weight_step)
                weighted_reward = (
                    weights.memory * float(memory_score)
                    + weights.length * length_score
                    + weights.similarity * float(similarity_score)
                )
                rewards[index] = (
                    weighted_reward
                    if length_valid
                    else min(weighted_reward, rejection_penalty)
                )

        return rewards

    reward_function.reward_weights = weights
    return reward_function