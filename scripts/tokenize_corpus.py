import json
import time
from pathlib import Path

from domain_specific_bpe_tokenizer.bpe_tokenizer import BPETokenizer


TOKENIZER_PATH = "resources/bpe_medical_final.json"

CLEANED_CORPUS_DIRS = {
    "general": "/content/drive/MyDrive/cleaned_corpus/general",
    "pubmed": "/content/drive/MyDrive/cleaned_corpus/pubmed",
    "pmc_open": "/content/drive/MyDrive/cleaned_corpus/pmc_open",
}

TOKENIZED_BASE_DIR = Path("/content/drive/MyDrive/tokenized_corpus")


def save_token_ids(
    token_ids: list[int],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(token_ids, f)


def tokenize_file(
    tokenizer: BPETokenizer,
    input_path: Path,
    output_path: Path,
) -> int:
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    token_ids = tokenizer.encode(text)

    save_token_ids(
        token_ids=token_ids,
        output_path=output_path,
    )

    return len(token_ids)


def main() -> None:
    start = time.time()

    tokenizer = BPETokenizer.load(TOKENIZER_PATH)

    total_files = 0
    total_tokens = 0

    for corpus_name, corpus_dir in CLEANED_CORPUS_DIRS.items():
        corpus_path = Path(corpus_dir)
        output_dir = TOKENIZED_BASE_DIR / corpus_name

        print(f"\nTokenizing corpus: {corpus_name}")
        print(f"Input : {corpus_path}")
        print(f"Output: {output_dir}")

        for file_path in sorted(corpus_path.glob("*.txt")):
            output_path = output_dir / file_path.with_suffix(".json").name

            token_count = tokenize_file(
                tokenizer=tokenizer,
                input_path=file_path,
                output_path=output_path,
            )

            total_files += 1
            total_tokens += token_count

            print(
                f"Tokenized: {file_path.name} | "
                f"Tokens: {token_count:,} | "
                f"Total tokens: {total_tokens:,}"
            )

    end = time.time()

    print("\nTokenization complete.")
    print(f"Files tokenized: {total_files}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Time taken: {end - start:.2f} seconds")


if __name__ == "__main__":
    main()