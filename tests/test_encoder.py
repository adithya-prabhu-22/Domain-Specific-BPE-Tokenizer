from domain_specific_bpe_tokenizer.bpe_tokenizer import BPETokenizer


def test_encode_consistency():
    tokenizer = BPETokenizer(vocab_size=261)
    text = "hello world"

    tokenizer.train(text)

    assert tokenizer.encode(text) == tokenizer.encode(text)