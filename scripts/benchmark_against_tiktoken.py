import tiktoken

from domain_specific_bpe_tokenizer import BPETokenizer


MEDICAL_TERMS = [
    "myocardial infarction",
    "electrocardiogram",
    "hepatocellular carcinoma",
    "pneumothorax",
    "neurodegenerative disorder",
]


def compare_single_term(
    custom_tokenizer: BPETokenizer,
    gpt2_tokenizer,
    term: str,
) -> None:

    custom_tokens = custom_tokenizer.encode(
        term,
    )

    gpt2_tokens = gpt2_tokenizer.encode(
        term,
    )

    print("=" * 80)

    print(f"TERM: {term}")

    print("-" * 80)

    print(
        f"Custom BPE Token Count : "
        f"{len(custom_tokens)}"
    )

    print(
        f"GPT-2 Token Count      : "
        f"{len(gpt2_tokens)}"
    )

    print("-" * 80)

    print(
        f"Custom BPE Tokens      : "
        f"{custom_tokens}"
    )

    print(
        f"GPT-2 Tokens           : "
        f"{gpt2_tokens}"
    )

    print("=" * 80)
    print()


def compare_full_text(
    custom_tokenizer: BPETokenizer,
    gpt2_tokenizer,
    text: str,
) -> None:

    custom_tokens = custom_tokenizer.encode(
        text,
    )

    gpt2_tokens = gpt2_tokenizer.encode(
        text,
    )

    custom_token_count = len(custom_tokens)

    gpt2_token_count = len(gpt2_tokens)

    original_bytes = len(
        text.encode("utf-8")
    )

    custom_compression_ratio = (
        original_bytes / custom_token_count
        if custom_token_count > 0
        else 0
    )

    gpt2_compression_ratio = (
        original_bytes / gpt2_token_count
        if gpt2_token_count > 0
        else 0
    )

    print("\n" + "=" * 80)
    print("FULL TEXT BENCHMARK")
    print("=" * 80)

    print(f"Original Bytes              : {original_bytes}")

    print()

    print(
        f"Custom BPE Token Count      : "
        f"{custom_token_count}"
    )

    print(
        f"GPT-2 Token Count           : "
        f"{gpt2_token_count}"
    )

    print()

    print(
        f"Custom Compression Ratio    : "
        f"{custom_compression_ratio:.4f}"
    )

    print(
        f"GPT-2 Compression Ratio     : "
        f"{gpt2_compression_ratio:.4f}"
    )

    print("=" * 80)


def load_evaluation_text(
    file_path: str,
) -> str:

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:

        return file.read()


def main():

    tokenizer_path = (
        "resources/trained_tokenizer/"
        "bpe_medical_52k.json"
    )

    evaluation_path = (
        "resources/evaluation/"
        "medical_eval.txt"
    )

    custom_tokenizer = BPETokenizer.load(
        tokenizer_path,
    )

    gpt2_tokenizer = tiktoken.get_encoding(
        "gpt2",
    )

    print("\n")
    print("=" * 80)
    print("MEDICAL TERM FRAGMENTATION BENCHMARK")
    print("=" * 80)

    for term in MEDICAL_TERMS:

        compare_single_term(
            custom_tokenizer=custom_tokenizer,
            gpt2_tokenizer=gpt2_tokenizer,
            term=term,
        )

    evaluation_text = load_evaluation_text(
        evaluation_path,
    )

    compare_full_text(
        custom_tokenizer=custom_tokenizer,
        gpt2_tokenizer=gpt2_tokenizer,
        text=evaluation_text,
    )


if __name__ == "__main__":

    main()