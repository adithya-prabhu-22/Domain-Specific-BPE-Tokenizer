from pathlib import Path

from domain_specific_bpe_tokenizer import BPETokenizer


MEDICAL_TERMS = [
    "myocardial infarction",
    "electrocardiogram",
    "hepatocellular carcinoma",
    "pneumothorax",
    "neurodegenerative disorder",
]


def load_evaluation_text(
    file_path: str,
) -> str:

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:

        return file.read()


def evaluate_text(
    tokenizer: BPETokenizer,
    text: str,
) -> None:

    tokens = tokenizer.encode(text)

    original_bytes = len(
        text.encode("utf-8")
    )

    token_count = len(tokens)

    compression_ratio = (
        original_bytes / token_count
        if token_count > 0
        else 0
    )

    average_chars_per_token = (
        len(text) / token_count
        if token_count > 0
        else 0
    )

    print("\n" + "=" * 60)
    print("TOKENIZER EVALUATION")
    print("=" * 60)

    print(f"Original characters      : {len(text)}")
    print(f"Original bytes           : {original_bytes}")
    print(f"Number of tokens         : {token_count}")

    print(
        f"Compression ratio        : "
        f"{compression_ratio:.4f}"
    )

    print(
        f"Average chars/token      : "
        f"{average_chars_per_token:.4f}"
    )


def evaluate_medical_terms(
    tokenizer: BPETokenizer,
) -> None:

    print("\n" + "=" * 60)
    print("MEDICAL TERM FRAGMENTATION")
    print("=" * 60)

    for term in MEDICAL_TERMS:

        encoded = tokenizer.encode(term)

        print(f"\nTerm: {term}")

        print(f"Token count: {len(encoded)}")

        print(f"Encoded tokens: {encoded}")


def evaluate_unknown_tokens(
    tokenizer: BPETokenizer,
    text: str,
) -> None:

    tokens = tokenizer.encode(text)

    unk_count = tokens.count(
        tokenizer.unk_id
    )

    total_tokens = len(tokens)

    oov_rate = (
        unk_count / total_tokens
        if total_tokens > 0
        else 0
    )

    print("\n" + "=" * 60)
    print("UNKNOWN TOKEN ANALYSIS")
    print("=" * 60)

    print(f"Unknown token count      : {unk_count}")

    print(f"Total token count        : {total_tokens}")

    print(f"OOV rate                 : {oov_rate:.6f}")


def main():

    tokenizer_path = (
        "resources/trained_tokenizer/"
        "bpe_medical_52k.json"
    )

    evaluation_path = (
        "resources/evaluation/"
        "medical_eval.txt"
    )

    tokenizer = BPETokenizer.load(
        tokenizer_path,
    )

    text = load_evaluation_text(
        evaluation_path,
    )

    evaluate_text(
        tokenizer=tokenizer,
        text=text,
    )

    evaluate_medical_terms(
        tokenizer=tokenizer,
    )

    evaluate_unknown_tokens(
        tokenizer=tokenizer,
        text=text,
    )


if __name__ == "__main__":

    main()
