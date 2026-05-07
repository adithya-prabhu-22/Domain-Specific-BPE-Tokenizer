import re
import time
from pathlib import Path


RAW_CORPUS_DIRS = {
    "general": "/content/drive/MyDrive/final_corpus/general",
    "pubmed": "/content/drive/MyDrive/final_corpus/pubmed",
    "pmc_open": "/content/drive/MyDrive/final_corpus/pmc_open",
}

CLEANED_BASE_DIR = Path("/content/drive/MyDrive/cleaned_corpus")


def clean_text(text: str) -> str:
    text = text.replace("\n", " ")

    # Remove HTML/XML tags if present
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove citation-like bracket noise: [1], [12,13], [Fig. 1]
    text = re.sub(r"\[[^\]]*\]", " ", text)

    # Remove repeated separators
    text = re.sub(r"[-_=]{3,}", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_file(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()

    cleaned_text = clean_text(raw_text)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)


def main() -> None:
    start = time.time()

    total_files = 0

    for corpus_name, raw_dir in RAW_CORPUS_DIRS.items():
        raw_path = Path(raw_dir)
        cleaned_dir = CLEANED_BASE_DIR / corpus_name

        print(f"\nCleaning corpus: {corpus_name}")
        print(f"Input : {raw_path}")
        print(f"Output: {cleaned_dir}")

        for file_path in sorted(raw_path.glob("*.txt")):
            output_path = cleaned_dir / file_path.name

            clean_file(
                input_path=file_path,
                output_path=output_path,
            )

            total_files += 1

            print(f"Cleaned: {file_path.name}")

    end = time.time()

    print("\nCleaning complete.")
    print(f"Files cleaned: {total_files}")
    print(f"Time taken: {end - start:.2f} seconds")


if __name__ == "__main__":
    main()