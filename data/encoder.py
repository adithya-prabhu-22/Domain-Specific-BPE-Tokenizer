from data.vocab import text_to_bytes


def merge_pair(
    ids: list[int],
    pair: tuple[int, int],
    new_token_id: int,
) -> list[int]:

    merged_ids = []

    i = 0

    while i < len(ids):

        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            merged_ids.append(new_token_id)
            i += 2

        else:
            merged_ids.append(ids[i])
            i += 1

    return merged_ids


def encode_text(
    tokenizer,
    text: str,
) -> list[int]:

    ids = text_to_bytes(text)

    for pair, token_id in tokenizer.merge_order:

        ids = merge_pair(
            ids=ids,
            pair=pair,
            new_token_id=token_id,
        )

    return ids