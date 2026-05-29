
from domain_specific_bpe_tokenizer import BPETokenizer


def test_special_tokens_exist():

    tokenizer = BPETokenizer()

    assert "[UNK]" in tokenizer.special_tokens
    assert "[PAD]" in tokenizer.special_tokens
    assert "[BOS]" in tokenizer.special_tokens
    assert "[EOS]" in tokenizer.special_tokens


def test_special_token_ids_unique():

    tokenizer = BPETokenizer()

    ids = list(
        tokenizer.special_tokens.values(),
    )

    assert len(ids) == len(set(ids))


def test_special_token_ids_are_integers():

    tokenizer = BPETokenizer()

    for token_id in tokenizer.special_tokens.values():

        assert isinstance(token_id, int)