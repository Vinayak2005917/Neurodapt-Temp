from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
BASE_MODEL_PATH = HERE / "models" / "Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = HERE / "outputs" / "Qwen2.5-0.5B-Instruct-GRPO"
MEMORY_RANKER_PATH = PROJECT_ROOT / "mem_ranker" / "memory_ranker_model_best.pt"
EMBEDDING_MODEL_PATH = (
    PROJECT_ROOT / "mem_ranker" / "data_pipeline" / "models" / "all_mpnet_base_v2"
)
EMBEDDING_DIMENSION = 768

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mem_ranker.model import MemoryRanker

DEFAULT_PROMPTS = [
    """Rewrite ONLY the target clause.

Keep its meaning unchanged.

Do not add information.

Do not use information from other clauses.

Output exactly one sentence.

Keep the rewrite at or below 12 words (150% of the original's 8 words).

Context:

The company had struggled for several years.

Its revenue finally began to recover.

[TARGET] The new product became popular with customers.

Sales increased rapidly during the following months.

Output:"""
    ,
    """Rewrite ONLY the target clause.

Keep its meaning unchanged.

Do not add information.

Do not use information from other clauses.

Output exactly one sentence.

Keep the rewrite at or below 12 words (150% of the original's 8 words).

Context:

The hikers had been walking since dawn.

[TARGET] The hikers found shelter before the storm arrived.

Rain began to fall across the mountain trail.

Output:""",
    """Rewrite ONLY the target clause.

Keep its meaning unchanged.

Do not add information.

Do not use information from other clauses.

Output exactly one sentence.

Keep the rewrite at or below 12 words (150% of the original's 8 words).

Context:

The museum had been closed for renovations.

[TARGET] The museum opened a new wing last spring.

Visitors can now see the ancient coins on display.

Output:""",
    """Rewrite ONLY the target clause.

Keep its meaning unchanged.

Do not add information.

Do not use information from other clauses.

Output exactly one sentence.

Keep the rewrite at or below 13 words (150% of the original's 9 words, rounded down).

Context:

Her grandmother wrote to her every winter.

[TARGET] She kept the old letter in a wooden box.

Years later, she found it while cleaning the attic.

Output:""",
]


def load_base_model():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    return AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL_PATH),
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=None,
            top_p=None,
            top_k=None,
        )

    generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def release_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_context(prompt: str) -> tuple[list[str], int] | None:
    """Read the ordered context clauses and the single [TARGET] clause."""
    lines = prompt.splitlines()
    context_start = next(
        (index for index, line in enumerate(lines) if line.strip().lower() == "context:"),
        None,
    )
    if context_start is None:
        return None

    context_lines = []
    for line in lines[context_start + 1 :]:
        if line.strip().lower() == "output:":
            break
        if line.strip():
            context_lines.append(line.strip())
    else:
        return None

    target_indices = [
        index
        for index, line in enumerate(context_lines)
        if line.startswith("[TARGET]")
    ]
    if len(target_indices) != 1:
        return None

    target_index = target_indices[0]
    context_lines[target_index] = context_lines[target_index][len("[TARGET]") :].strip()
    if not all(context_lines):
        return None
    return context_lines, target_index


class ContextualMemoryRanker:
    """Score candidate rewrites in the surrounding clause sequence."""

    def __init__(self) -> None:
        if not MEMORY_RANKER_PATH.is_file():
            raise FileNotFoundError(f"Memory ranker checkpoint not found: {MEMORY_RANKER_PATH}")
        if not EMBEDDING_MODEL_PATH.is_dir():
            raise FileNotFoundError(f"MPNet embedding model not found: {EMBEDDING_MODEL_PATH}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model = SentenceTransformer(
            str(EMBEDDING_MODEL_PATH), device=str(self.device)
        )
        self.model = MemoryRanker(input_dim=EMBEDDING_DIMENSION)
        state_dict = torch.load(
            MEMORY_RANKER_PATH, map_location="cpu", weights_only=True
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def score_candidates(
        self, context_clauses: list[str], target_index: int, candidates: list[str]
    ) -> list[float]:
        stories = [
            context_clauses[:target_index] + [candidate] + context_clauses[target_index + 1 :]
            for candidate in candidates
        ]
        flat_clauses = [clause for story in stories for clause in story]
        embeddings = self.embedding_model.encode(
            flat_clauses,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        embeddings_tensor = torch.as_tensor(
            embeddings.reshape(len(stories), len(context_clauses), EMBEDDING_DIMENSION),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.inference_mode():
            scores = torch.sigmoid(self.model(embeddings_tensor))[:, target_index]
        return scores.cpu().tolist()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the original Qwen model and its trained LoRA adapter."
    )
    parser.add_argument(
        "--prompt",
        action="append",
        help="User prompt text; repeat this option to compare several prompts.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=24,
        help="Maximum generated tokens for each model (default: 24).",
    )
    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if not BASE_MODEL_PATH.is_dir():
        raise FileNotFoundError(f"Base model directory not found: {BASE_MODEL_PATH}")
    if not (ADAPTER_PATH / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"LoRA adapter not found at {ADAPTER_PATH}; train the model first."
        )

    prompts = args.prompt if args.prompt else DEFAULT_PROMPTS
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL_PATH))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading the original base model in 4-bit...")
    base_model = load_base_model()
    base_model.eval()
    base_outputs = [
        generate(base_model, tokenizer, prompt, args.max_new_tokens)
        for prompt in prompts
    ]
    release_model(base_model)

    print("Loading the same base model with the trained LoRA adapter...")
    adapted_base = load_base_model()
    tuned_model = PeftModel.from_pretrained(adapted_base, str(ADAPTER_PATH))
    tuned_model.eval()
    tuned_outputs = [
        generate(tuned_model, tokenizer, prompt, args.max_new_tokens)
        for prompt in prompts
    ]
    release_model(tuned_model)

    for index, (prompt, base_output, tuned_output) in enumerate(
        zip(prompts, base_outputs, tuned_outputs), start=1
    ):
        print(f"\n{'=' * 80}\nPROMPT {index}\n{'=' * 80}\n{prompt}")
        print(f"\n--- ORIGINAL MODEL ---\n{base_output or '[empty output]'}")
        print(f"\n--- FINE-TUNED MODEL ---\n{tuned_output or '[empty output]'}")

    parsed_contexts = [extract_context(prompt) for prompt in prompts]
    scorable_indices = [
        index
        for index, context in enumerate(parsed_contexts)
        if context is not None and base_outputs[index] and tuned_outputs[index]
    ]
    if scorable_indices:
        print("\nLoading the memory ranker and MPNet encoder...")
        ranker = ContextualMemoryRanker()
        for index in scorable_indices:
            context_clauses, target_index = parsed_contexts[index]
            scores = ranker.score_candidates(
                context_clauses,
                target_index,
                [base_outputs[index], tuned_outputs[index]],
            )
            winner = "ORIGINAL MODEL" if scores[0] >= scores[1] else "FINE-TUNED MODEL"
            print(f"\n--- CONTEXTUAL MEMORY RANKING (PROMPT {index + 1}) ---")
            print(f"Original model score:   {scores[0]:.4f}")
            print(f"Fine-tuned model score: {scores[1]:.4f}")
            print(f"Higher-ranked rewrite:  {winner}")

    unscorable_indices = [
        index
        for index, context in enumerate(parsed_contexts)
        if context is None and base_outputs[index] and tuned_outputs[index]
    ]
    for index in unscorable_indices:
        print(
            f"\nSkipping memory ranking for prompt {index + 1}: add a Context: section "
            "with one [TARGET] line and an Output: delimiter."
        )


if __name__ == "__main__":
    main()
