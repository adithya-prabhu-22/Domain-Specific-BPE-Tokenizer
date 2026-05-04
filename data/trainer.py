from collections import Counter

from data.encoder import merge_pair
from data.vocab import text_to_bytes


def get_adjacent_pairs(
    ids: list[int],
) -> list[tuple[int, int]]:

    return [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]


def count_pair_frequencies(
    ids: list[int],
) -> Counter:

    return Counter(get_adjacent_pairs(ids))


def get_most_frequent_pair(
    pair_frequencies: Counter,
):

    if not pair_frequencies:
        return None

    return pair_frequencies.most_common(1)[0][0]


def train_step(
    ids: list[int],
    new_token_id: int,
):

    pair_frequencies = count_pair_frequencies(ids)

    pair = get_most_frequent_pair(pair_frequencies)

    if pair is None:
        return ids, None

    merged_ids = merge_pair(
        ids=ids,
        pair=pair,
        new_token_id=new_token_id,
    )

    return merged_ids, pair


def train_bpe(
    tokenizer,
    text: str,
) -> list[int]:

    ids = text_to_bytes(text)

    next_token_id = 256

    while next_token_id < tokenizer.vocab_size:

        ids, pair = train_step(
            ids=ids,
            new_token_id=next_token_id,
        )

        if pair is None:
            break

        tokenizer.merges[pair] = next_token_id
        tokenizer.merge_order.append((pair, next_token_id))

        tokenizer.vocab[next_token_id] = (
            tokenizer.vocab[pair[0]]
            + tokenizer.vocab[pair[1]]
        )

        next_token_id += 1

    return ids