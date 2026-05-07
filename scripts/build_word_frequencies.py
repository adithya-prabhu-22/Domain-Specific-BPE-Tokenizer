import pickle
import re
import time

from collections import Counter
from pathlib import Path


CORPUS_DIRS = [
    "/content/drive/MyDrive/final_corpus/general",
    "/content/drive/MyDrive/final_corpus/pubmed",
    "/content/drive/MyDrive/final_corpus/pmc_open",
]

OUTPUT_PATH = "resources/word_freqs.pkl"

CHECKPOINT_EVERY = 10


def update_word_frequencies(
    file_path: Path,
    word_freqs: Counter,
) -> None:

    print(f"Reading: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:

        for line in f:

            words = re.findall(r"\S+", line)

            word_freqs.update(words)


def save_checkpoint(
    word_freqs: Counter,
    output_path: str,
) -> None:

    with open(output_path, "wb") as f:
        pickle.dump(word_freqs, f)

    print(f"Checkpoint saved: {output_path}")


def main() -> None:

    start = time.time()

    word_freqs = Counter()

    total_files = 0

    for corpus_dir in CORPUS_DIRS:

        corpus_path = Path(corpus_dir)

        print(f"\nProcessing corpus: {corpus_dir}")

        for file_path in sorted(corpus_path.glob("*.txt")):

            update_word_frequencies(
                file_path=file_path,
                word_freqs=word_freqs,
            )

            total_files += 1

            if total_files % CHECKPOINT_EVERY == 0:

                save_checkpoint(
                    word_freqs=word_freqs,
                    output_path=OUTPUT_PATH,
                )

                print(
                    f"Files processed: {total_files}"
                )

                print(
                    f"Unique words: {len(word_freqs):,}"
                )

    save_checkpoint(
        word_freqs=word_freqs,
        output_path=OUTPUT_PATH,
    )

    end = time.time()

    print("\nDone.")

    print(f"Time taken: {end - start:.2f} seconds")

    print(f"Unique words: {len(word_freqs):,}")

    print(
        f"Total words: "
        f"{sum(word_freqs.values()):,}"
    )


if __name__ == "__main__":
    main()