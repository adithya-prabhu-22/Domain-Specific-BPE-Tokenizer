from data.bpe_tokenizer import BPETokenizer


def test_encode_consistency():
    tokenizer = BPETokenizer(vocab_size=260)
    text = "hello world"

    tokenizer.train(text)

    assert tokenizer.encode(text) == tokenizer.encode(text)