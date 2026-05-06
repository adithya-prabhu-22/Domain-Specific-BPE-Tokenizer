from pathlib import Path
import re

from datasets import load_dataset
from tqdm import tqdm


OUTPUT_DIR = Path("/content/drive/MyDrive/final_corpus/general")

TARGET_WORDS = 150_000_000
CHUNK_WORDS = 1_000_000


def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_text(example: dict) -> str:
    if "text" in example and example["text"]:
        return example["text"]

    return ""


def save_chunk(words: list[str], chunk_id: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    path = OUTPUT_DIR / f"chunk_{chunk_id:06d}.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write(" ".join(words))

    print(f"Saved {path} with {len(words):,} words")


def main() -> None:
    dataset = load_dataset(
        "openwebtext",
        split="train",
        streaming=True,
    )

    total_words = 0
    chunk_words = []
    chunk_id = 1

    for example in tqdm(dataset):
        text = clean_text(get_text(example))

        if not text:
            continue

        words = text.split()

        for word in words:
            chunk_words.append(word)
            total_words += 1

            if len(chunk_words) >= CHUNK_WORDS:
                save_chunk(chunk_words, chunk_id)

                chunk_words = []
                chunk_id += 1

                print(f"Total collected: {total_words:,}")

            if total_words >= TARGET_WORDS:
                break

        if total_words >= TARGET_WORDS:
            break

    if chunk_words:
        save_chunk(chunk_words, chunk_id)

    print(f"\nDone. Total general words collected: {total_words:,}")


if __name__ == "__main__":
    main()