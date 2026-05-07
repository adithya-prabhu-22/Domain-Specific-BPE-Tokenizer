from collections import Counter
from pathlib import Path
import re

from data.encoder import merge_pair
from data.vocab import text_to_bytes


def build_word_frequencies(corpus_dir: str) -> Counter[str]:

    word_freqs = Counter()
    corpus_path = Path(corpus_dir)

    for file_path in sorted(corpus_path.glob("*.txt")):

        print(f"Reading: {file_path}")

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:

            for line in f:
                words = re.findall(r"\S+", line)
                word_freqs.update(words)

    return word_freqs


def get_adjacent_pairs(
    ids: list[int],
) -> list[tuple[int, int]]:

    return [
        (ids[i], ids[i + 1])
        for i in range(len(ids) - 1)
    ]


def count_pair_frequencies(
    ids: list[int],
) -> Counter[tuple[int, int]]:

    return Counter(get_adjacent_pairs(ids))


def count_weighted_pair_frequencies(
    word_token_ids: dict[str, list[int]],
    word_freqs: Counter[str],
) -> Counter[tuple[int, int]]:

    pair_frequencies = Counter()

    for word, ids in word_token_ids.items():

        frequency = word_freqs[word]

        for pair in get_adjacent_pairs(ids):
            pair_frequencies[pair] += frequency

    return pair_frequencies


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


def train_bpe_from_word_frequencies(
    tokenizer,
    word_freqs: Counter[str],
    checkpoint_every: int | None = None,
    checkpoint_dir: str | None = None,
) -> dict[str, list[int]]:

    word_token_ids = {
        word: text_to_bytes(word)
        for word in word_freqs
    }

    next_token_id = 256

    while next_token_id < tokenizer.vocab_size:

        pair_frequencies = count_weighted_pair_frequencies(
            word_token_ids=word_token_ids,
            word_freqs=word_freqs,
        )

        pair = get_most_frequent_pair(pair_frequencies)

        if pair is None:
            break

        new_word_token_ids = {}

        for word, ids in word_token_ids.items():

            new_word_token_ids[word] = merge_pair(
                ids=ids,
                pair=pair,
                new_token_id=next_token_id,
            )

        word_token_ids = new_word_token_ids

        tokenizer.merges[pair] = next_token_id
        tokenizer.merge_order.append((pair, next_token_id))

        tokenizer.vocab[next_token_id] = (
            tokenizer.vocab[pair[0]]
            + tokenizer.vocab[pair[1]]
        )

        if next_token_id % 100 == 0:
            print(
                f"Learned token {next_token_id} | "
                f"Pair: {pair} | "
                f"Frequency: {pair_frequencies[pair]}"
            )

        if (
            checkpoint_every is not None
            and checkpoint_dir is not None
            and next_token_id % checkpoint_every == 0
        ):
            checkpoint_path = (
                f"{checkpoint_dir}/"
                f"bpe_checkpoint_token_{next_token_id}.json"
            )

            tokenizer.save(checkpoint_path)

            print(f"Checkpoint saved: {checkpoint_path}")

        next_token_id += 1

    return word_token_ids