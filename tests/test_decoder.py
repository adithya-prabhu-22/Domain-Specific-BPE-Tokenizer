
from domain_specific_bpe_tokenizer import BPETokenizer


def test_decode_returns_string():

    tokenizer = BPETokenizer(
        vocab_size=500,
    )

    tokenizer.train(
        "hello world",
    )

    encoded = tokenizer.encode(
        "hello",
    )

    decoded = tokenizer.decode(
        encoded,
    )

    assert isinstance(decoded, str)


def test_decode_matches_original_text():

    tokenizer = BPETokenizer(
        vocab_size=500,
    )

    text = "medical diagnosis"

    tokenizer.train(text)

    encoded = tokenizer.encode(text)

    decoded = tokenizer.decode(encoded)

    assert decoded == text