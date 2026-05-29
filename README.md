# Domain-Specific-BPE-Tokenizer

A from-scratch implementation of a scalable domain-specific Byte Pair Encoding (BPE) tokenizer for medical and general NLP corpora.

This project implements a complete byte-level BPE tokenizer pipeline without relying on external tokenizer frameworks. It includes scalable corpus collection, corpus cleaning, weighted BPE training, merge-rule optimization, serialization, evaluation tooling, benchmarking against GPT-2/tiktoken, automated testing, and PyPI-ready packaging.

The tokenizer was designed for large-scale medical-domain language modeling experiments and was later integrated into a custom GPT-style language model training pipeline.

---

## Project Overview

This repository focuses on building and evaluating a domain-specific BPE tokenizer from scratch.

Core goals:

* Implement byte-level BPE training from scratch
* Support scalable weighted word-frequency training
* Train tokenizers on medical and general corpora
* Optimize merge-rule lookup and encoding speed
* Provide serialization and checkpoint support
* Benchmark against GPT-2/tiktoken
* Evaluate fragmentation quality for biomedical terms
* Create a reusable PyPI-ready tokenizer package

---

## Key Features

* Byte-level BPE tokenizer
* Weighted word-frequency BPE training
* Domain-specific tokenizer training
* Medical corpus preprocessing pipeline
* Incremental merge-rule learning
* Optimized ranked merge lookup
* Save/load tokenizer support
* Checkpointed tokenizer training
* Scalable corpus chunk processing
* Special token support
* Serialization utilities
* PyPI-ready package structure
* Automated unit tests
* Tokenizer evaluation pipeline
* GPT-2/tiktoken benchmarking

---

## Repository Structure

```text
Domain-Specific-BPE-Tokenizer/
├── domain_specific_bpe_tokenizer/
│   ├── __init__.py
│   ├── bpe_tokenizer.py
│   ├── trainer.py
│   ├── encoder.py
│   ├── decoder.py
│   ├── serialization.py
│   └── vocab.py
│
├── tests/
│   ├── test_bpe_tokenizer.py
│   ├── test_vocab.py
│   ├── test_trainer.py
│   ├── test_encoder.py
│   ├── test_decoder.py
│   ├── test_serialization.py
│   ├── test_save_load.py
│   ├── test_encode_decode.py
│   ├── test_special_tokens.py
│   └── test_invalid_inputs.py
│
├── examples/
│   ├── basic_usage.py
│   ├── load_pretrained_tokenizer.py
│   └── medical_tokenization_demo.py
│
├── scripts/
│   ├── prepare_medical_corpus.py
│   ├── collect_general_corpus.py
│   ├── collect_pubmed_corpus.py
│   ├── collect_pmc_open_corpus.py
│   ├── clean_corpus.py
│   ├── build_word_frequencies.py
│   ├── train_tokenizer.py
│   ├── tokenize_corpus.py
│   ├── evaluate_tokenizer.py
│   └── benchmark_against_tiktoken.py
│
├── resources/
│   ├── raw_corpus/
│   ├── cleaned_corpus/
│   ├── tokenized_corpus/
│   ├── trained_tokenizer/
│   └── evaluation/
│
├── docs/
│   ├── architecture.md
│   ├── evaluation.md
│   └── benchmarks.md
│
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── pyproject.toml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/adithya-prabhu-22/Domain-Specific-BPE-Tokenizer.git

cd Domain-Specific-BPE-Tokenizer
```

---

### 2. Install the Package

```bash
pip install -e .
```

---

### 3. Install Optional Benchmark Dependency

```bash
pip install tiktoken
```

---

## Package Usage

### Import the Tokenizer

```python
from domain_specific_bpe_tokenizer import BPETokenizer
```

---

### Train a Tokenizer

```python
from domain_specific_bpe_tokenizer import BPETokenizer

tokenizer = BPETokenizer(
    vocab_size=5000,
    min_frequency=2,
)

text = "medical corpus text goes here"

tokenizer.train(text)
```

---

### Encode Text

```python
encoded = tokenizer.encode(
    "myocardial infarction"
)

print(encoded)
```

---

### Decode Tokens

```python
decoded = tokenizer.decode(encoded)

print(decoded)
```

---

### Save Tokenizer

```python
tokenizer.save(
    "resources/trained_tokenizer/bpe_tokenizer.json"
)
```

---

### Load Tokenizer

```python
loaded_tokenizer = BPETokenizer.load(
    "resources/trained_tokenizer/bpe_tokenizer.json"
)
```

---

## Special Tokens

The tokenizer supports the following built-in special tokens:

| Token   | Purpose               |
| ------- | --------------------- |
| `[UNK]` | Unknown token         |
| `[PAD]` | Padding token         |
| `[BOS]` | Beginning of sequence |
| `[EOS]` | End of sequence       |

---

## Training Pipeline

The tokenizer training pipeline supports scalable corpus preprocessing.

Pipeline stages:

1. Corpus collection
2. Corpus cleaning
3. Word-frequency building
4. Weighted BPE training
5. Merge-rule optimization
6. Tokenization
7. Evaluation
8. Benchmarking

---

## Corpus Sources

The project includes scripts for collecting:

* General English corpora
* PubMed abstracts
* PMC Open Access biomedical papers

Example:

```bash
python -m scripts.collect_pubmed_corpus
```

---

## Weighted BPE Training

The tokenizer supports weighted word-frequency BPE training for scalable preprocessing.

Features:

* Frequency-based merge learning
* Rare-word filtering
* Checkpoint saving
* Large-corpus subset selection
* Optimized pair-frequency updates

Example:

```bash
python -m scripts.train_tokenizer
```

---

## Tokenizer Evaluation

The repository includes tokenizer evaluation tooling.

Evaluation metrics include:

* Compression ratio
* Average characters per token
* Unknown token rate
* Medical term fragmentation
* Token count statistics

Run evaluation:

```bash
python -m scripts.evaluate_tokenizer
```

---

## Benchmark Against GPT-2/tiktoken

The repository includes benchmarking against GPT-2/tiktoken.

Comparison metrics:

* Token count
* Compression ratio
* Medical-term fragmentation
* Domain-token efficiency

Run benchmark:

```bash
python -m scripts.benchmark_against_tiktoken
```

---

## Example Benchmark Result

### Medical Term Fragmentation

| Medical Term             | Custom BPE Tokens | GPT-2 Tokens |
| ------------------------ | ----------------: | -----------: |
| electrocardiogram        |                 1 |            5 |
| pneumothorax             |                 1 |            5 |
| myocardial infarction    |                 3 |            6 |
| hepatocellular carcinoma |                 3 |            7 |

The custom medical-domain tokenizer significantly reduces fragmentation for biomedical terminology compared with GPT-2/tiktoken.

---

## Tokenizer Training Scale

The tokenizer training pipeline evolved across multiple scalability stages:

| Stage                      | Vocabulary Size |               Corpus Scale |
| -------------------------- | --------------: | -------------------------: |
| Initial Prototype          |             500 |        Small sample corpus |
| Intermediate Training      |             24K | Multi-million token corpus |
| Final Biomedical Tokenizer |             52K | ~240M-token medical corpus |

---

## Optimization Features

The tokenizer includes several performance optimizations:

* Ranked merge lookup
* Pair-frequency indexing
* Safe affected-word updates
* Chunk-based corpus processing
* Cached frequency-table loading
* Incremental merge updates

These optimizations significantly improved scalability during large-corpus tokenizer training.

---

## Testing

The repository includes a comprehensive automated test suite.

Current coverage includes:

* Tokenizer initialization
* Encoding
* Decoding
* Save/load serialization
* Special tokens
* Invalid input handling
* End-to-end tokenizer consistency

Run tests:

```bash
pytest
```

### Current Status

```text
26 tests passed
```

---

## PyPI Packaging

The repository is structured as a reusable Python package using `pyproject.toml`.

Editable installation:

```bash
pip install -e .
```

The public API is exposed through:

```python
from domain_specific_bpe_tokenizer import BPETokenizer
```

---

## Documentation

Additional documentation is available in:

```text
docs/
```

Including:

* tokenizer architecture
* evaluation methodology
* benchmark reports

---

## Integration with LLM Training

This tokenizer was later integrated into a custom GPT-style medical language model training pipeline.

The tokenizer was used for:

* domain-specific tokenization
* autoregressive GPT training
* tokenizer-quality experiments
* KV-cache benchmarking studies

---

## Future Work

Future improvements include:

* Rust implementation for faster preprocessing
* Parallel BPE training
* Memory-efficient billion-token preprocessing
* Streaming tokenizer training
* Unicode normalization improvements
* Faster serialization formats
* Hugging Face tokenizer interoperability
* Tokenizer visualization tools
* Vocabulary pruning experiments

---

## Conclusion

This project demonstrates a complete from-scratch implementation of a scalable domain-specific BPE tokenizer.

The repository evolved from a simple educational tokenizer into a scalable biomedical NLP preprocessing pipeline featuring weighted BPE training, optimized merge learning, tokenizer evaluation, benchmarking against GPT-2/tiktoken, automated testing, and PyPI-ready packaging.

The strongest results appeared in medical-domain tokenization, where the tokenizer significantly reduced fragmentation for biomedical terminology compared with GPT-2/tiktoken.

Overall, the project demonstrates the importance of tokenizer design, merge-rule optimization, scalable preprocessing, and domain-specific vocabulary construction for modern NLP and LLM systems.
