from domain_specific_bpe_tokenizer.bpe_tokenizer import BPETokenizer


def test_save_load(tmp_path):
    tokenizer = BPETokenizer(vocab_size=261)
    text = "cf cf cf cf"

    tokenizer.train(text)
    tokenizer.save(tmp_path / "tok.json")

    loaded = BPETokenizer.load(tmp_path / "tok.json")

    assert loaded.decode(loaded.encode(text)) == text