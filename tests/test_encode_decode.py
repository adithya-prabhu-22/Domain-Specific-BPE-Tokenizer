
from domain_specific_bpe_tokenizer import BPETokenizer


def test_encode_decode_roundtrip():

    tokenizer = BPETokenizer(
        vocab_size=500,
    )

    text = "patient diagnosed with pneumonia"

    tokenizer.train(text)

    encoded = tokenizer.encode(text)

    decoded = tokenizer.decode(encoded)

    assert decoded == text


def test_encode_output_not_empty():

    tokenizer = BPETokenizer(
        vocab_size=500,
    )

    tokenizer.train(
        "hello world",
    )

    encoded = tokenizer.encode(
        "hello",
    )

    assert len(encoded) > 0