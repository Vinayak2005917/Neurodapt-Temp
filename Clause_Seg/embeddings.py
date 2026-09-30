from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


MODEL_ID = "google-bert/bert-base-uncased"
LOCAL_BERT_MODEL = Path(__file__).resolve().parent.parent / "models" / "bert-base-uncased"
HF_CACHE = LOCAL_BERT_MODEL.parent / "huggingface_cache"

# BERT-base has 512 positional embeddings. Keep the token sequence and the
# returned token list truncated identically so clause segmentation never gets
# a token/embedding length mismatch on long stories.
MAX_TOKEN_LENGTH = 512


def _local_model_is_ready() -> bool:
    return (
        (LOCAL_BERT_MODEL / "config.json").is_file()
        and any(
            (LOCAL_BERT_MODEL / filename).is_file()
            for filename in ("model.safetensors", "pytorch_model.bin")
        )
        and any(
            (LOCAL_BERT_MODEL / filename).is_file()
            for filename in ("tokenizer.json", "vocab.txt")
        )
    )


@lru_cache(maxsize=None)
def _load_bert(device_name: str):
    """Download BERT once, save a local copy, and retain it per device."""
    LOCAL_BERT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    HF_CACHE.mkdir(parents=True, exist_ok=True)

    if _local_model_is_ready():
        source = str(LOCAL_BERT_MODEL)
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
        model = AutoModel.from_pretrained(source, local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=str(HF_CACHE))
        model = AutoModel.from_pretrained(MODEL_ID, cache_dir=str(HF_CACHE))
        LOCAL_BERT_MODEL.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(LOCAL_BERT_MODEL)
        model.save_pretrained(LOCAL_BERT_MODEL, safe_serialization=True)

    selected_device = torch.device(device_name)
    print(f"Using BERT model from {LOCAL_BERT_MODEL} on {selected_device}")
    model = model.to(selected_device)
    model.eval()
    return tokenizer, model, selected_device


#tokenize the original text
#embedd the original text
#cat the embeddings of each token with the embeddings of the original text
#return [n,2*embedding_dim] where n is the number of tokens in the original text
def text_to_embeddings(text, device=None):
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer, model, selected_device = _load_bert(str(selected_device))
    encoding = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_TOKEN_LENGTH,
    )

    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    encoding.pop("offset_mapping", None)

    # Transformer embedding lookups require integer index tensors.  Some
    # tokenizer/model combinations return token_type_ids (and occasionally
    # attention_mask) with a floating dtype, which causes CUDA failures in
    # BERT's embedding layer even though input_ids was already corrected.
    for key in ("input_ids", "token_type_ids", "attention_mask"):
        if key in encoding:
            encoding[key] = encoding[key].long()

    encoding = {
        key: value.to(selected_device)
        for key, value in encoding.items()
    }

    with torch.no_grad():
        outputs = model(**encoding)

    text_embedding = outputs.last_hidden_state.mean(dim=1)
    token_embeddings = outputs.last_hidden_state

    concatenated_embeddings = torch.cat(
        [
            token_embeddings,
            text_embedding.unsqueeze(1).expand(
                -1,
                token_embeddings.size(1),
                -1,
            ),
        ],
        dim=2,
    )

    return concatenated_embeddings

def text_to_tokens(text, device=None):
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer, _, _ = _load_bert(str(selected_device))
    encoding = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_TOKEN_LENGTH,
    )
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    return tokens
