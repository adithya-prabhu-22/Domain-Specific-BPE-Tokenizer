# Domain-Specific BPE Tokenizer

A from-scratch implementation of a scalable Byte Pair Encoding (BPE) tokenizer designed for domain-specific NLP applications, particularly medical and scientific text.

This project implements the complete tokenizer development pipeline, including:

- Corpus collection and preprocessing
- Weighted BPE training
- Merge-rule learning and optimization
- Tokenizer serialization
- Evaluation tooling
- Benchmarking against OpenAI's tiktoken
- Automated testing
- PyPI package distribution

The tokenizer was developed as part of a larger custom GPT-style language modeling pipeline and was trained on both medical and general-domain corpora.

---

## Features

- Byte-level BPE tokenizer implementation from scratch
- Weighted word-frequency BPE training
- Domain-specific vocabulary learning
- Configurable vocabulary size
- Configurable minimum token frequency
- Encoding and decoding support
- Save and load trained tokenizers
- Serialization utilities
- Automated unit testing
- Evaluation tooling
- Benchmarking against OpenAI tiktoken
- PyPI-ready package distribution

---

## Installation

Install directly from PyPI:

```bash
pip install adithya-domain-specific-bpe-tokenizer
```

Or install from source:

```bash
git clone https://github.com/adithya-prabhu-22/Domain-Specific-BPE-Tokenizer.git

cd Domain-Specific-BPE-Tokenizer

pip install .
```

---

## Quick Start

### Train a Tokenizer

```python
from domain_specific_bpe_tokenizer import BPETokenizer

corpus = """
Cardiovascular disease requires cardiovascular monitoring.
Cancer immunotherapy improves treatment outcomes.
"""

tokenizer = BPETokenizer(
    vocab_size=1000,
    min_frequency=1
)

tokenizer.train(corpus)
```

---

### Encode Text

```python
tokens = tokenizer.encode(
    "Cardiovascular disease treatment"
)

print(tokens)
```

Example Output:

```python
[284, 287, 312]
```

---

### Decode Text

```python
decoded = tokenizer.decode(tokens)

print(decoded)
```

Output:

```text
cardiovascular disease treatment
```

---

### Save a Tokenizer

```python
tokenizer.save(
    "medical_tokenizer.json"
)
```

---

### Load a Tokenizer

```python
from domain_specific_bpe_tokenizer import BPETokenizer

tokenizer = BPETokenizer.load(
    "medical_tokenizer.json"
)
```

---

## Example Learned Compression

After training on a medical corpus, the tokenizer learns domain-specific merge rules that significantly reduce token fragmentation.

Example results:

```text
Word: electrocardiography
Characters: 19
BPE Tokens: 1
```

```text
Word: gastroenterology
Characters: 16
BPE Tokens: 1
```

```text
Word: cardiovascularimmunotherapy
Characters: 27
BPE Tokens: 2
```

These results demonstrate how domain-specific merge learning can efficiently represent biomedical terminology compared with character-level tokenization.

---

## Project Structure

```text
Domain-Specific-BPE-Tokenizer/
│
├── domain_specific_bpe_tokenizer/
│   ├── __init__.py
│   ├── bpe_tokenizer.py
│   ├── trainer.py
│   ├── encoder.py
│   ├── decoder.py
│   ├── serialization.py
│   └── vocab.py
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
├── tests/
│
├── resources/
│
├── LICENSE
├── README.md
└── pyproject.toml
```

---

## Training Pipeline

The tokenizer training workflow consists of:

1. Corpus collection
2. Corpus cleaning
3. Word-frequency generation
4. Weighted BPE training
5. Merge-rule learning
6. Tokenization
7. Evaluation
8. Benchmarking

---

## Evaluation

The repository includes tokenizer evaluation tooling.

Evaluation metrics include:

- Compression ratio
- Average characters per token
- Vocabulary utilization
- Medical term fragmentation
- Token count statistics

Run evaluation:

```bash
python scripts/evaluate_tokenizer.py
```

---

## Benchmarking Against tiktoken

The repository includes benchmarking against OpenAI's tiktoken tokenizer.

Comparison metrics include:

- Token count
- Compression ratio
- Medical-term fragmentation
- Domain-token efficiency

Run benchmarking:

```bash
python scripts/benchmark_against_tiktoken.py
```

---

## Tokenizer Training Scale

The tokenizer evolved across multiple training stages:

| Stage | Vocabulary Size |
|---------|---------:|
| Initial Prototype | 500 |
| Intermediate Training | 24K |
| Final Medical Tokenizer | 52K |

The final tokenizer was trained using large-scale medical and general-domain corpora to improve compression efficiency for biomedical terminology.

---

## Testing

The repository includes automated unit tests covering:

- Tokenizer initialization
- BPE training
- Encoding
- Decoding
- Serialization
- Save and load functionality
- Special tokens
- Invalid input handling

Run tests:

```bash
pytest
```

---

## PyPI Distribution

The tokenizer is available as a reusable Python package:

```bash
pip install adithya-domain-specific-bpe-tokenizer
```

Public API:

```python
from domain_specific_bpe_tokenizer import BPETokenizer
```

PyPI Package:

https://pypi.org/project/adithya-domain-specific-bpe-tokenizer/

---

## Current Release

### Version 0.1.0

Included features:

- BPE tokenizer training
- Encoding and decoding
- Save and load functionality
- Evaluation tools
- Benchmarking tools
- Automated testing
- PyPI distribution

---

## Roadmap

### Version 0.2.0

Planned improvements:

- Pretrained medical tokenizer distribution
- One-line pretrained tokenizer loading API
- GitHub Actions CI/CD
- Extended benchmarking suite
- Additional domain-specific corpora
- Improved tokenizer visualization

---

## License

This project is licensed under the MIT License.

---

## Author

**Adithya Prabhu**

GitHub:
https://github.com/adithya-prabhu-22

PyPI:
https://pypi.org/project/adithya-domain-specific-bpe-tokenizer/