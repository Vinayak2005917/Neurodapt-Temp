from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "./models/Qwen2.5-0.5B-Instruct"

print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Downloading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.save_pretrained(OUTPUT_DIR)

print(f"Model saved to: {OUTPUT_DIR}")