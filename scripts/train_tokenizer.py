import time

from data.bpe_tokenizer import BPETokenizer
from data.trainer import build_word_frequencies


CORPUS_DIR = "resources/medical_corpus"
OUTPUT_PATH = "resources/bpe_medical.json"
VOCAB_SIZE = 5000


print("Building word frequency table...")

start = time.time()

word_freqs = build_word_frequencies(CORPUS_DIR)

end = time.time()

print(f"Word frequency table built in {end - start:.2f} seconds")
print("Unique words:", len(word_freqs))
print("Total words:", sum(word_freqs.values()))

print("Creating training text from word frequency table...")

training_text = " ".join(
    word
    for word, _ in word_freqs.most_common()
)

print("Training text length:", len(training_text))

tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE)

print("Training tokenizer...")
start = time.time()

tokenizer.train(training_text)

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
tokenizer.save(OUTPUT_PATH)
print("Saved:", OUTPUT_PATH)