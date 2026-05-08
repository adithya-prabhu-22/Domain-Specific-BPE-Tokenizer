import pickle
import time
from data.bpe_tokenizer import BPETokenizer

WORD_FREQS_PATH = "resources/word_freqs.pkl"
OUTPUT_PATH     = "resources/bpe_medical_24k.json"
CHECKPOINT_DIR  = "/content/drive/MyDrive/tokenizer_checkpoints_24k/"
VOCAB_SIZE       = 24000
CHECKPOINT_EVERY = 1000
MIN_FREQUENCY    = 3

print("Loading word frequency table...")
start = time.time()

with open(WORD_FREQS_PATH, "rb") as f:
    word_freqs = pickle.load(f)

if not word_freqs:
    raise ValueError(f"No word frequencies found in: {WORD_FREQS_PATH}")

end = time.time()
print(f"Word frequency table loaded in {end - start:.2f} seconds")
print("Unique words before filter:", len(word_freqs))
print("Total words before filter:", sum(word_freqs.values()))

print(f"Filtering words with frequency < {MIN_FREQUENCY}...")
word_freqs = {
    word: frequency
    for word, frequency in word_freqs.items()
    if frequency >= MIN_FREQUENCY
}

if not word_freqs:
    raise ValueError(
        "Word frequency table became empty after filtering. "
        "Try lowering MIN_FREQUENCY."
    )

print("Unique words after filter:", len(word_freqs))
print("Total words after filter:", sum(word_freqs.values()))

tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE)

print("Training tokenizer with weighted word frequencies...")
start = time.time()

tokenizer.train_from_word_frequencies(
    word_freqs,
    checkpoint_every=CHECKPOINT_EVERY,
    checkpoint_dir=CHECKPOINT_DIR,
)

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
    print("Sample  :", sample)
    print("Encoded :", encoded[:20])
    print("Decoded :", decoded)
    assert decoded == sample

print("Tokenizer test passed.")

print("Saving tokenizer...")
tokenizer.save(OUTPUT_PATH)
print("Saved:", OUTPUT_PATH)