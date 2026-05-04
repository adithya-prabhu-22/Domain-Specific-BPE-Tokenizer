import json
from pathlib import Path


def save_tokenizer(
    tokenizer,
    path: str,
) -> None:

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = {
        "vocab_size": tokenizer.vocab_size,
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


def load_tokenizer(
    tokenizer_cls,
    path: str,
):

    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tokenizer = tokenizer_cls(vocab_size=data["vocab_size"])

    for item in data["merge_order"]:

        pair = tuple(item["pair"])
        token_id = item["token_id"]

        tokenizer.merges[pair] = token_id
        tokenizer.merge_order.append((pair, token_id))

        tokenizer.vocab[token_id] = (
            tokenizer.vocab[pair[0]]
            + tokenizer.vocab[pair[1]]
        )

    return tokenizer