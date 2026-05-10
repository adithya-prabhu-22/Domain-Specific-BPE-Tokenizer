from data.vocab import text_to_bytes


def merge_pair(
    ids: list[int],
    pair: tuple[int, int],
    new_token_id: int,
) -> list[int]:
    """
    Merge all non-overlapping occurrences of a given pair.

    Kept for trainer.py compatibility.
    """
    merged_ids = []
    i = 0

    while i < len(ids):
        if (
            i < len(ids) - 1
            and ids[i] == pair[0]
            and ids[i + 1] == pair[1]
        ):
            merged_ids.append(new_token_id)
            i += 2
        else:
            merged_ids.append(ids[i])
            i += 1

    return merged_ids


def build_merge_ranks(tokenizer):
    """
    Build lookup table:
    pair -> (rank, token_id)

    Lower rank = higher priority.
    """
    return {
        pair: (rank, token_id)
        for rank, (pair, token_id) in enumerate(tokenizer.merge_order)
    }


def find_best_pair(
    ids: list[int],
    merge_ranks: dict,
):
    """
    Find the highest-priority mergeable adjacent pair
    currently present in ids.
    """
    best_pair = None
    best_rank = float("inf")
    best_token_id = None

    for i in range(len(ids) - 1):
        pair = (ids[i], ids[i + 1])

        if pair in merge_ranks:
            rank, token_id = merge_ranks[pair]

            if rank < best_rank:
                best_rank = rank
                best_pair = pair
                best_token_id = token_id

    return best_pair, best_token_id


def encode_text(
    tokenizer,
    text: str,
) -> list[int]:
    text = text.lower()

    ids = text_to_bytes(text)

    merge_ranks = build_merge_ranks(tokenizer)

    while len(ids) >= 2:
        best_pair, new_token_id = find_best_pair(
            ids=ids,
            merge_ranks=merge_ranks,
        )

        if best_pair is None:
            break

        ids = merge_pair(
            ids=ids,
            pair=best_pair,
            new_token_id=new_token_id,
        )

    unk = getattr(tokenizer, "unk_id", 256)
    ids = [i if i in tokenizer.vocab else unk for i in ids]

    return ids