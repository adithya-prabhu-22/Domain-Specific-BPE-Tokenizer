
from domain_specific_bpe_tokenizer.vocab import build_base_vocab


def test_build_base_vocab():

    vocab = build_base_vocab()

    assert isinstance(vocab, dict)

    assert len(vocab) == 256


def test_vocab_keys_are_integers():

    vocab = build_base_vocab()

    for key in vocab.keys():

        assert isinstance(key, int)


def test_vocab_values_are_bytes():

    vocab = build_base_vocab()

    for value in vocab.values():

        assert isinstance(value, bytes)