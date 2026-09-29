import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./models/Qwen2.5-0.5B-Instruct"

messages = [
    {
        "role": "user",
        "content": """Rewrite ONLY the target clause.

Keep its meaning unchanged.

Do not add information.

Do not use information from other clauses.

Output exactly one sentence.

Context:

The company had struggled for several years.

Its revenue finally began to recover.

[TARGET] The new product became popular with customers.

Sales increased rapidly during the following months.

Output:"""
    }
]

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
)

model.eval()

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=24,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        temperature=None,
        top_p=None,
        top_k=None,
    )

generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

response = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True,
).strip()

print("\n--- MODEL OUTPUT ---")
print(response)