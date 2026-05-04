from collections import Counter
import json
from pathlib import Path


class BPETokenizer:

    def __init__(self, vocab_size: int = 500):

        if not isinstance(vocab_size, int):
            raise TypeError("vocab_size must be an integer.")

        if vocab_size <= 256:
            raise ValueError("vocab_size must be greater than 256.")

        self.vocab_size = vocab_size
        self.merges = {}
        self.merge_order = []
        self.vocab = self.build_base_vocab()

    def build_base_vocab(self) -> dict[int, bytes]:

        return {i: bytes([i]) for i in range(256)}

    def text_to_bytes(self, text: str) -> list[int]:

        if not isinstance(text, str):
            raise TypeError("Input text must be a string.")

        return list(text.encode("utf-8"))

    def get_adjacent_pairs(
        self,
        ids: list[int],
    ) -> list[tuple[int, int]]:

        return [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]

    def count_pair_frequencies(
        self,
        ids: list[int],
    ) -> Counter:

        return Counter(self.get_adjacent_pairs(ids))

    def get_most_frequent_pair(
        self,
        pair_frequencies: Counter,
    ):

        if not pair_frequencies:
            return None

        return pair_frequencies.most_common(1)[0][0]

    def merge_pair(
        self,
        ids: list[int],
        pair: tuple[int, int],
        new_token_id: int,
    ) -> list[int]:

        merged_ids = []

        i = 0

        while i < len(ids):

            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                merged_ids.append(new_token_id)
                i += 2

            else:
                merged_ids.append(ids[i])
                i += 1

        return merged_ids

    def train_step(
        self,
        ids: list[int],
        new_token_id: int,
    ):

        pair_frequencies = self.count_pair_frequencies(ids)

        pair = self.get_most_frequent_pair(pair_frequencies)

        if pair is None:
            return ids, None

        merged_ids = self.merge_pair(
            ids=ids,
            pair=pair,
            new_token_id=new_token_id,
        )

        return merged_ids, pair
from data.decoder import decode_tokens
from data.encoder import encode_text
from data.serialization import load_tokenizer, save_tokenizer
from data.trainer import train_bpe
from data.vocab import build_base_vocab


class BPETokenizer:

    def __init__(self, vocab_size: int = 500):

        if not isinstance(vocab_size, int):
            raise TypeError("vocab_size must be an integer.")

        if vocab_size <= 256:
            raise ValueError("vocab_size must be greater than 256.")

        self.vocab_size = vocab_size
        self.merges = {}
        self.merge_order = []
        self.vocab = build_base_vocab()

    def train(
        self,
        text: str,
    ) -> list[int]:

        return train_bpe(
            tokenizer=self,
            text=text,
        )

    def encode(
        self,
        text: str,
    ) -> list[int]:

        return encode_text(
            tokenizer=self,
            text=text,
        )

    def decode(
        self,
        token_ids: list[int],
    ) -> str:

        return decode_tokens(
            tokenizer=self,
            token_ids=token_ids,
        )

    def save(
        self,
        path: str,
    ) -> None:

        save_tokenizer(
            tokenizer=self,
            path=path,
        )

    @classmethod
    def load(
        cls,
        path: str,
    ) -> "BPETokenizer":

        return load_tokenizer(
            tokenizer_cls=cls,
            path=path,
        )


if __name__ == "__main__":

    tokenizer = BPETokenizer(vocab_size=260)

    text = "cf cf cf cf"

    tokenizer.train(text)

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    tokenizer.save("resources/bpe_tokenizer.json")

    loaded_tokenizer = BPETokenizer.load("resources/bpe_tokenizer.json")

    loaded_encoded = loaded_tokenizer.encode(text)
    loaded_decoded = loaded_tokenizer.decode(loaded_encoded)

    print("Encoded:", encoded)
    print("Decoded:", decoded)
    print("Match:", text == decoded)

    print("Loaded encoded:", loaded_encoded)
    print("Loaded decoded:", loaded_decoded)
    print("Loaded match:", text == loaded_decoded)
    def train(
        self,
        text: str,
    ) -> list[int]:

        ids = self.text_to_bytes(text)

        next_token_id = 256

        while next_token_id < self.vocab_size:

            ids, pair = self.train_step(
                ids=ids,
                new_token_id=next_token_id,
            )

            if pair is None:
                break

            self.merges[pair] = next_token_id
            self.merge_order.append((pair, next_token_id))

            self.vocab[next_token_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

            next_token_id += 1

        return ids

    def encode(
        self,
        text: str,
    ) -> list[int]:

        ids = self.text_to_bytes(text)

        for pair, token_id in self.merge_order:

            ids = self.merge_pair(
                ids=ids,
                pair=pair,
                new_token_id=token_id,
            )

        return ids

    def decode(
        self,
        token_ids: list[int],
    ) -> str:

        byte_sequence = b""

        for token_id in token_ids:

            if token_id not in self.vocab:
                raise ValueError(f"Unknown token id: {token_id}")

            byte_sequence += self.vocab[token_id]

        return byte_sequence.decode("utf-8")

    def save(
        self,
        path: str,
    ) -> None:

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = {
            "vocab_size": self.vocab_size,
            "merge_order": [
                {
                    "pair": [pair[0], pair[1]],
                    "token_id": token_id,
                }
                for pair, token_id in self.merge_order
            ],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load(
        cls,
        path: str,
    ) -> "BPETokenizer":

        path = Path(path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls(vocab_size=data["vocab_size"])

        for item in data["merge_order"]:

            pair = tuple(item["pair"])
            token_id = item["token_id"]

            tokenizer.merges[pair] = token_id
            tokenizer.merge_order.append((pair, token_id))

            tokenizer.vocab[token_id] = (
                tokenizer.vocab[pair[0]] + tokenizer.vocab[pair[1]]
            )

        return tokenizer


if __name__ == "__main__":

    tokenizer = BPETokenizer(vocab_size=260)

    text = "cf cf cf cf"

    tokenizer.train(text)

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    tokenizer.save("resources/bpe_tokenizer.json")

    loaded_tokenizer = BPETokenizer.load("resources/bpe_tokenizer.json")

    loaded_encoded = loaded_tokenizer.encode(text)
    loaded_decoded = loaded_tokenizer.decode(loaded_encoded)

    print("Encoded:", encoded)
    print("Decoded:", decoded)
    print("Match:", text == decoded)

    print("Loaded encoded:", loaded_encoded)
    print("Loaded decoded:", loaded_decoded)
    print("Loaded match:", text == loaded_decoded)
