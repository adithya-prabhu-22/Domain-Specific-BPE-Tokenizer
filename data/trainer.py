from collections import Counter, defaultdict
from pathlib import Path
import re

from data.encoder import merge_pair
from data.vocab import text_to_bytes


def build_word_frequencies(corpus_dir: str) -> Counter:

    word_freqs = Counter()
    corpus_path = Path(corpus_dir)

    for file_path in sorted(corpus_path.glob("*.txt")):

        print(f"Reading: {file_path}")

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:

            for line in f:
                words = re.findall(r"[a-z0-9]+|[^\w\s]", line.lower())
                word_freqs.update(words)

    return word_freqs


def count_weighted_pair_frequencies(
    word_token_ids: dict,
    word_freqs: Counter,
) -> Counter:

    pair_frequencies = Counter()

    for word, ids in word_token_ids.items():

        freq = word_freqs[word]

        for i in range(len(ids) - 1):
            pair_frequencies[(ids[i], ids[i + 1])] += freq

    return pair_frequencies


def get_most_frequent_pair(
    pair_frequencies: Counter,
):

    if not pair_frequencies:
        return None

    return pair_frequencies.most_common(1)[0][0]


def train_bpe(
    tokenizer,
    text: str,
):

    word_freqs = Counter(
        re.findall(
            r"[a-z0-9]+|[^\w\s]",
            text.lower(),
        )
    )

    return train_bpe_from_word_frequencies(
        tokenizer=tokenizer,
        word_freqs=word_freqs,
    )


def train_bpe_from_word_frequencies(
    tokenizer,
    word_freqs: Counter,
    checkpoint_every: int | None = None,
    checkpoint_dir: str | None = None,
):

    word_freqs = Counter(
        {
            word.lower(): freq
            for word, freq in word_freqs.items()
        }
    )

    word_token_ids = {
        word: list(text_to_bytes(word))
        for word in word_freqs
    }

    pair_to_words = defaultdict(set)

    for word, ids in word_token_ids.items():

        for i in range(len(ids) - 1):
            pair_to_words[(ids[i], ids[i + 1])].add(word)

    pair_frequencies = count_weighted_pair_frequencies(
        word_token_ids=word_token_ids,
        word_freqs=word_freqs,
    )

    next_token_id = max(tokenizer.vocab.keys()) + 1

    while next_token_id < tokenizer.vocab_size:

        pair = get_most_frequent_pair(pair_frequencies)

        if pair is None:
            break

        best_freq = pair_frequencies.get(pair, 0)

        if best_freq < tokenizer.min_frequency:
            print(
                f"Frequency {best_freq} below threshold "
                f"{tokenizer.min_frequency}. "
                f"Stopping at vocab size {next_token_id}."
            )
            break

        affected_words = list(
            pair_to_words.get(pair, set())
        )

        for word in affected_words:

            old_ids = word_token_ids[word]

            new_ids = merge_pair(
                ids=old_ids,
                pair=pair,
                new_token_id=next_token_id,
            )

            for i in range(len(old_ids) - 1):

                p = (old_ids[i], old_ids[i + 1])

                pair_frequencies[p] -= word_freqs[word]

                if pair_frequencies[p] <= 0:
                    del pair_frequencies[p]

                pair_to_words[p].discard(word)

            for i in range(len(new_ids) - 1):

                p = (new_ids[i], new_ids[i + 1])

                pair_frequencies[p] += word_freqs[word]
                pair_to_words[p].add(word)

            word_token_ids[word] = new_ids

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
                f"Frequency: {best_freq}"
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