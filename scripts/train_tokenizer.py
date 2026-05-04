from data.bpe_tokenizer import BPETokenizer
import time

print("Loading data...")

with open("resources/data.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("Data loaded. Length:", len(text))

tokenizer = BPETokenizer(vocab_size=5000)

print("Training tokenizer...")
start = time.time()

tokenizer.train(text)

end = time.time()
print(f"Training complete. Time taken: {end - start:.2f} seconds")

print("Testing tokenizer...")

sample = "patient treatment"
encoded = tokenizer.encode(sample)
decoded = tokenizer.decode(encoded)

print("Sample:", sample)
print("Encoded:", encoded[:20])
print("Decoded:", decoded)

assert decoded == sample

print("Tokenizer test passed.")

print("Saving tokenizer...")
tokenizer.save("resources/bpe_medical.json")
print("Saved.")