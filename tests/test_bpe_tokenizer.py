from domain_specific_bpe_tokenizer import BPETokenizer


def test_tokenizer_initialization():

    tokenizer = BPETokenizer(
        vocab_size=500,
    )

    assert tokenizer.vocab_size == 500

    assert "[UNK]" in tokenizer.special_tokens
    assert "[PAD]" in tokenizer.special_tokens
    assert "[BOS]" in tokenizer.special_tokens
    assert "[EOS]" in tokenizer.special_tokens


def test_tokenizer_train():

    tokenizer = BPETokenizer(
        vocab_size=300,
    )

    tokenizer.train(
        "hello hello hello world",
    )

    assert len(tokenizer.vocab) > 260