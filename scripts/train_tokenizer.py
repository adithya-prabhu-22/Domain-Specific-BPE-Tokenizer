import time

from data.bpe_tokenizer import BPETokenizer
from data.trainer import build_word_frequencies


CORPUS_DIR = "resources/medical_corpus"
OUTPUT_PATH = "resources/bpe_medical_1m.json"
VOCAB_SIZE = 5000


print("Building word frequency table...")

start = time.time()

word_freqs = build_word_frequencies(CORPUS_DIR)

if not word_freqs:
    raise ValueError(
        f"No words found in corpus directory: {CORPUS_DIR}. "
        "Make sure it contains .txt chunk files."
    )

end = time.time()

print(f"Word frequency table built in {end - start:.2f} seconds")
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