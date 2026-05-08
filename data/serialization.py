import json
from pathlib import Path


TOKENIZER_FORMAT_VERSION = "1.0"


def save_tokenizer(tokenizer, path: str) -> None:

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    special_tokens = getattr(tokenizer, "special_tokens", {})

    vocab_serializable = {
        str(token_id): list(token_bytes)
        for token_id, token_bytes in tokenizer.vocab.items()
    }

    data = {
        "version": TOKENIZER_FORMAT_VERSION,
        "vocab_size": tokenizer.vocab_size,
        "special_tokens": special_tokens,
        "vocab": vocab_serializable,
        "merge_order": [
            {
                "pair": [pair[0], pair[1]],
                "token_id": token_id,
            }
            for pair, token_id in tokenizer.merge_order
        ],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"Saved tokenizer to {path}")
    print(f"  Vocab size   : {len(tokenizer.vocab)}")
    print(f"  Merges saved : {len(tokenizer.merge_order)}")
    print(f"  Special tokens: {list(special_tokens.keys())}")


def load_tokenizer(tokenizer_cls, path: str):

    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    version = data.get("version", "unknown")
    if version != TOKENIZER_FORMAT_VERSION:
        print(
            f"Warning: tokenizer version mismatch "
            f"(file={version}, expected={TOKENIZER_FORMAT_VERSION})"
        )

    tokenizer = tokenizer_cls(vocab_size=data["vocab_size"])

    if "vocab" in data:
        tokenizer.vocab = {
            int(token_id): bytes(token_bytes)
            for token_id, token_bytes in data["vocab"].items()
        }

    if "special_tokens" in data:
        tokenizer.special_tokens = data["special_tokens"]
        tokenizer.unk_id = tokenizer.special_tokens.get("[UNK]")
        tokenizer.pad_id = tokenizer.special_tokens.get("[PAD]")
        tokenizer.bos_id = tokenizer.special_tokens.get("[BOS]")
        tokenizer.eos_id = tokenizer.special_tokens.get("[EOS]")

    for item in data["merge_order"]:
        pair = tuple(item["pair"])
        token_id = item["token_id"]
        tokenizer.merges[pair] = token_id
        tokenizer.merge_order.append((pair, token_id))

    print(f"Loaded tokenizer from {path}")
    print(f"  Vocab size    : {len(tokenizer.vocab)}")
    print(f"  Merges loaded : {len(tokenizer.merge_order)}")
    print(f"  Special tokens: {list(tokenizer.special_tokens.keys())}")

    return tokenizer