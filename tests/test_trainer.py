

from domain_specific_bpe_tokenizer import BPETokenizer


def test_training_adds_merges():

    tokenizer = BPETokenizer(
        vocab_size=300,
    )

    tokenizer.train(
        "aa aa aa aa bb bb",
    )

    assert len(tokenizer.merges) > 0


def test_training_increases_vocab():

    tokenizer = BPETokenizer(
        vocab_size=300,
    )

    initial_vocab_size = len(tokenizer.vocab)

    tokenizer.train(
        "machine learning machine learning",
    )

    assert len(tokenizer.vocab) > initial_vocab_size