from domain_specific_bpe_tokenizer import BPETokenizer


def test_save_creates_file():

    tokenizer = BPETokenizer(
        vocab_size=500,
    )

    tokenizer.train(
        "heart disease prediction",
    )

    tokenizer.save(
        "resources/test_tokenizer.json",
    )


def test_load_returns_tokenizer():

    tokenizer = BPETokenizer(
        vocab_size=500,
    )

    tokenizer.train(
        "heart disease prediction",
    )

    path = "resources/test_tokenizer.json"

    tokenizer.save(path)

    loaded = BPETokenizer.load(path)

    assert isinstance(
        loaded,
        BPETokenizer,
    )