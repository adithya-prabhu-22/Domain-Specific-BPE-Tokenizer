from data.decoder import decode_tokens
from data.encoder import encode_text
from data.serialization import load_tokenizer, save_tokenizer
from data.trainer import train_bpe, train_bpe_from_word_frequencies
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

    def train_from_word_frequencies(
        self,
        word_freqs,
    ):

        return train_bpe_from_word_frequencies(
            tokenizer=self,
            word_freqs=word_freqs,
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