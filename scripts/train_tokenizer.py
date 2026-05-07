import pickle
import time

from data.bpe_tokenizer import BPETokenizer


WORD_FREQS_PATH = "resources/word_freqs.pkl"
OUTPUT_PATH = "resources/bpe_medical_final.json"
VOCAB_SIZE = 8000


print("Loading word frequency table...")

start = time.time()

with open(WORD_FREQS_PATH, "rb") as f:
    word_freqs = pickle.load(f)

if not word_freqs:
    raise ValueError(
        f"No word frequencies found in: {WORD_FREQS_PATH}"
    )

end = time.time()

print(f"Word frequency table loaded in {end - start:.2f} seconds")
print("Unique words:", len(word_freqs))
print("Total words:", sum(word_freqs.values()))

tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE)

print("Training tokenizer with weighted word frequencies...")

start = time.time()

tokenizer.train_from_word_frequencies(word_freqs)

end = time.time()

print(f"Training complete. Time taken: {end - start:.2f} seconds")

print("Testing tokenizer...")

samples = [
    "patient treatment",
    "hypertension and diabetes",
    "cardiovascular disease",
    "immunotherapy improved survival outcomes",
]

for sample in samples:
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)

    print("Sample:", sample)
    print("Encoded:", encoded[:20])
    print("Decoded:", decoded)

    assert decoded == sample

print("Tokenizer test passed.")

print("Saving tokenizer...")

tokenizer.save(OUTPUT_PATH)

print("Saved:", OUTPUT_PATH)