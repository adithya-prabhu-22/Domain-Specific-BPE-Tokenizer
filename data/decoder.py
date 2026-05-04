def decode_tokens(
    tokenizer,
    token_ids: list[int],
) -> str:

    byte_sequence = b""

    for token_id in token_ids:

        if token_id not in tokenizer.vocab:
            raise ValueError(f"Unknown token id: {token_id}")

        byte_sequence += tokenizer.vocab[token_id]

    return byte_sequence.decode("utf-8")