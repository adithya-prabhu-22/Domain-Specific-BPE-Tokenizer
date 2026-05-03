from collections import Counter


class BPETokenizer:

    def __init__(self, vocab_size: int = 500):

        if not isinstance(vocab_size, int):
            raise TypeError("vocab_size must be an integer.")

        if vocab_size <= 256:
            raise ValueError("vocab_size must be greater than 256.")

        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = self.build_base_vocab()

    def build_base_vocab(self) -> dict[int, bytes]:

        return {i: bytes([i]) for i in range(256)}

    def text_to_bytes(self, text: str) -> list[int]:

        if not isinstance(text, str):
            raise TypeError("Input text must be a string.")

        return list(text.encode("utf-8"))

    def bytes_to_text(self, byte_ids: list[int]) -> str:

        if not isinstance(byte_ids, list):
            raise TypeError("byte_ids must be a list of integers.")

        for byte_id in byte_ids:

            if not isinstance(byte_id, int):
                raise TypeError("All byte ids must be integers.")

            if byte_id < 0 or byte_id > 255:
                raise ValueError("Byte ids must be in the range 0 to 255.")

        return bytes(byte_ids).decode("utf-8")

    def get_adjacent_pairs(
        self,
        ids: list[int],
    ) -> list[tuple[int, int]]:

        if not isinstance(ids, list):
            raise TypeError("ids must be a list of integers.")

        for token_id in ids:

            if not isinstance(token_id, int):
                raise TypeError("All token ids must be integers.")

        pairs = []

        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            pairs.append(pair)

        return pairs

    def count_pair_frequencies(
        self,
        ids: list[int],
    ) -> Counter[tuple[int, int]]:

        pairs = self.get_adjacent_pairs(ids)

        return Counter(pairs)

    def get_most_frequent_pair(
        self,
        pair_frequencies: Counter[tuple[int, int]],
    ) -> tuple[int, int] | None:

        if not pair_frequencies:
            return None

        return pair_frequencies.most_common(1)[0][0]

    def merge_pair(
        self,
        ids: list[int],
        pair: tuple[int, int],
        new_token_id: int,
    ) -> list[int]:

        if not isinstance(ids, list):
            raise TypeError("ids must be a list of integers.")

        if not isinstance(pair, tuple):
            raise TypeError("pair must be a tuple.")

        if len(pair) != 2:
            raise ValueError("pair must contain exactly two token ids.")

        if not isinstance(new_token_id, int):
            raise TypeError("new_token_id must be an integer.")

        merged_ids = []

        i = 0

        while i < len(ids):

            if (
                i < len(ids) - 1
                and ids[i] == pair[0]
                and ids[i + 1] == pair[1]
            ):
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
    ) -> tuple[list[int], tuple[int, int] | None]:

        pair_frequencies = self.count_pair_frequencies(ids)

        most_frequent_pair = self.get_most_frequent_pair(
            pair_frequencies
        )

        if most_frequent_pair is None:
            return ids, None

        merged_ids = self.merge_pair(
            ids=ids,
            pair=most_frequent_pair,
            new_token_id=new_token_id,
        )

        return merged_ids, most_frequent_pair

    def train(
        self,
        text: str,
    ) -> list[int]:

        ids = self.text_to_bytes(text)

        next_token_id = 256

        while next_token_id < self.vocab_size:

            ids, learned_pair = self.train_step(
                ids=ids,
                new_token_id=next_token_id,
            )

            if learned_pair is None:
                break

            self.merges[learned_pair] = next_token_id

            self.vocab[next_token_id] = (
                self.vocab[learned_pair[0]]
                + self.vocab[learned_pair[1]]
            )

            next_token_id += 1

        return ids

    def decode_token_ids(
        self,
        token_ids: list[int],
    ) -> str:

        if not isinstance(token_ids, list):
            raise TypeError("token_ids must be a list of integers.")

        byte_sequence = b""

        for token_id in token_ids:

            if not isinstance(token_id, int):
                raise TypeError("All token ids must be integers.")

            if token_id not in self.vocab:
                raise ValueError(f"Unknown token id: {token_id}")

            byte_sequence += self.vocab[token_id]

        return byte_sequence.decode("utf-8")


if __name__ == "__main__":

    tokenizer = BPETokenizer(vocab_size=260)

    text = "cf cf cf cf"

    trained_ids = tokenizer.train(text)

    decoded_text = tokenizer.decode_token_ids(trained_ids)

    print("Original text:", text)
    print("Trained ids:", trained_ids)
    print("Merge rules:", tokenizer.merges)
    print("Vocab[256]:", tokenizer.vocab[256])
    print("Vocab[256] decoded:", tokenizer.vocab[256].decode("utf-8"))
    print("Vocab[257]:", tokenizer.vocab[257])
    print("Vocab[257] decoded:", tokenizer.vocab[257].decode("utf-8"))
    print("Decoded from token ids:", decoded_text)
    print("Match:", text == decoded_text)