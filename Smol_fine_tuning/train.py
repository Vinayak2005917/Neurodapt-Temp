# train_grpo.py

import os
import random
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from trl import GRPOConfig, GRPOTrainer
from reward_scorer import MemoryRewardScorer, reward_function_factory


MODEL_PATH = "./models/Qwen2.5-0.5B-Instruct"
DATASET_PATH = "./data_pipeline/preprocessed_grpo.pt"
OUTPUT_DIR = "./outputs/Qwen2.5-0.5B-Instruct-GRPO"
MAX_TRAIN_EXAMPLES = int(os.getenv("GRPO_MAX_TRAIN_EXAMPLES", "20000"))
TRAIN_SAMPLE_SEED = int(os.getenv("GRPO_TRAIN_SAMPLE_SEED", "42"))

if MAX_TRAIN_EXAMPLES <= 0:
    raise ValueError("GRPO_MAX_TRAIN_EXAMPLES must be greater than zero")

# Load tokenizer before constructing chat-template prompts.
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# Load preprocessed dataset

raw_data = torch.load(DATASET_PATH, weights_only=False)

valid_indices = [
    i
    for i, item in enumerate(raw_data["examples"])
    if 0 <= item["target_index"] < len(item["clauses"])
]
if len(valid_indices) > MAX_TRAIN_EXAMPLES:
    valid_indices = random.Random(TRAIN_SAMPLE_SEED).sample(
        valid_indices, MAX_TRAIN_EXAMPLES
    )

examples = []

for example_index in valid_indices:
    item = raw_data["examples"][example_index]
    clauses = item["clauses"]
    target_index = item["target_index"]

    target_clause = clauses[target_index]["text"]
    target_word_count = len(target_clause.split())
    max_rewrite_words = (3 * target_word_count) // 2

    context_lines = []

    for i, clause in enumerate(clauses):
        if i == target_index:
            context_lines.append(f"[TARGET] {clause['text']}")
        else:
            context_lines.append(clause["text"])

    prompt = f"""Rewrite ONLY the target clause.

Keep its meaning unchanged.

Do not add information.

Do not use information from other clauses.

Output exactly one sentence.

Keep the rewrite at or below {max_rewrite_words} words (150% of the original's {target_word_count} words).

Context:

{chr(10).join(context_lines)}

Output:"""

    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

    examples.append(
        {
            "prompt": chat_prompt,
            "target_clause": target_clause,
            "original_score": float(clauses[target_index]["score"]),
        }
    )


dataset = Dataset.from_list(examples)
print(
    f"Training examples: {len(dataset)} selected from "
    f"{len(raw_data['examples'])} validatable dataset rows "
    f"(seed={TRAIN_SAMPLE_SEED})"
)


# 4-bit QLoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

model.config.use_cache = False
model.enable_input_require_grads()


# LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
)


# Reward

memory_ranker = MemoryRewardScorer(
    memory_model_path=PROJECT_ROOT / "mem_ranker" / "memory_ranker_model_best.pt",
    device=os.getenv("MEMORY_RANKER_DEVICE", "auto"),
)
reward_function = reward_function_factory(memory_ranker)


# GRPO configuration

training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,

    num_generations=4,

    learning_rate=5e-6,

    num_train_epochs=1,

    logging_steps=10,
    save_steps=500,
    save_total_limit=2,

    bf16=True,
    fp16=False,
    gradient_checkpointing=True,

    max_completion_length=48,

    gradient_checkpointing_kwargs={"use_reentrant": False},

    use_vllm=False,

    report_to="none",

    remove_unused_columns=False,
)


# Trainer

trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_function,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
)


print("Starting GRPO training...")

trainer.train()

print("Saving model...")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Done.")