from domain_specific_bpe_tokenizer import BPETokenizer


tokenizer = BPETokenizer(vocab_size=500)

text = "cf cf cf cf"

tokenizer.train(text)

encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)

tokenizer.save("resources/bpe_tokenizer.json")

loaded_tokenizer = BPETokenizer.load(
    "resources/bpe_tokenizer.json"
)

loaded_encoded = loaded_tokenizer.encode(text)
loaded_decoded = loaded_tokenizer.decode(loaded_encoded)

print("Encoded:", encoded)
print("Decoded:", decoded)
print("Match:", text == decoded)

print("Loaded encoded:", loaded_encoded)
print("Loaded decoded:", loaded_decoded)
print("Loaded match:", text == loaded_decoded)
