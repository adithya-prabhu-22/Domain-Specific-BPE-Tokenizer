import pytest

from domain_specific_bpe_tokenizer import BPETokenizer


def test_invalid_vocab_size_type():

    with pytest.raises(TypeError):

        BPETokenizer(
            vocab_size="500",
        )


def test_invalid_vocab_size_value():

    with pytest.raises(ValueError):

        BPETokenizer(
            vocab_size=100,
        )


def test_invalid_decode_input():

    tokenizer = BPETokenizer()

    with pytest.raises(Exception):

        tokenizer.decode(
            ["invalid", "tokens"],
        )


def test_invalid_load_path():

    with pytest.raises(Exception):

        BPETokenizer.load(
            "invalid_path.json",
        )


def test_empty_string_encoding():

    tokenizer = BPETokenizer()

    encoded = tokenizer.encode("")

    assert isinstance(encoded, list)