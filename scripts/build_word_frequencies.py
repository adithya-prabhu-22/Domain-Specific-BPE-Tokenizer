import pickle
import re
import time

from collections import Counter
from pathlib import Path


CORPUS_LIMITS = {
    "/content/drive/MyDrive/final_corpus/general": 30,
    "/content/drive/MyDrive/final_corpus/pubmed": 10,
    "/content/drive/MyDrive/final_corpus/pmc_open": 80,
}
OUTPUT_PATH = "resources/word_freqs.pkl"

CHECKPOINT_EVERY = 10


def update_word_frequencies(
    file_path: Path,
    word_freqs: Counter,
) -> None:

    print(f"Reading: {file_path}")

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:

        for line in f:

            words = re.findall(r"\S+", line)

            word_freqs.update(words)


def save_checkpoint(
    word_freqs: Counter,
    output_path: str,
) -> None:

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "wb") as f:
        pickle.dump(word_freqs, f)

    print(f"Checkpoint saved: {output_path}")


def main() -> None:

    start = time.time()

    word_freqs = Counter()

    total_files = 0

    for corpus_dir, max_files in CORPUS_LIMITS.items():

        corpus_path = Path(corpus_dir)

        print(f"\nProcessing corpus: {corpus_dir}")
        print(f"Max files: {max_files}")

        chunk_files = sorted(corpus_path.glob("*.txt"))[:max_files]

        if not chunk_files:
            raise ValueError(
                f"No .txt chunk files found in: {corpus_dir}"
            )

        for file_path in chunk_files:

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

                print(f"Files processed: {total_files}")
                print(f"Unique words: {len(word_freqs):,}")
                print(f"Total words: {sum(word_freqs.values()):,}")

    save_checkpoint(
        word_freqs=word_freqs,
        output_path=OUTPUT_PATH,
    )

    end = time.time()

    print("\nDone.")
    print(f"Time taken: {end - start:.2f} seconds")
    print(f"Files processed: {total_files}")
    print(f"Unique words: {len(word_freqs):,}")
    print(f"Total words: {sum(word_freqs.values()):,}")


if __name__ == "__main__":
    main()