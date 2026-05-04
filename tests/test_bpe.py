from data.bpe_tokenizer import BPETokenizer


def test_end_to_end():
    tokenizer = BPETokenizer(vocab_size=260)
    text = "cf cf cf cf"

    tokenizer.train(text)
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert decoded == text