# tests/test_save_load.py

from domain_specific_bpe_tokenizer import BPETokenizer


def test_save_and_load_consistency():

    tokenizer = BPETokenizer(
        vocab_size=500,
    )

    text = "myocardial infarction"

    tokenizer.train(text)

    path = "resources/test_tokenizer.json"

    tokenizer.save(path)

    loaded = BPETokenizer.load(path)

    original_encoded = tokenizer.encode(text)

    loaded_encoded = loaded.encode(text)

    assert original_encoded == loaded_encoded


def test_loaded_tokenizer_decodes_correctly():

    tokenizer = BPETokenizer(
        vocab_size=500,
    )

    text = "pneumonia diagnosis"

    tokenizer.train(text)

    path = "resources/test_tokenizer.json"

    tokenizer.save(path)

    loaded = BPETokenizer.load(path)

    encoded = loaded.encode(text)

    decoded = loaded.decode(encoded)

    assert decoded == text