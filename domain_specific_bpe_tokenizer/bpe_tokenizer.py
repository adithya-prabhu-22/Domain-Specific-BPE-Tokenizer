from domain_specific_bpe_tokenizer.decoder import decode_tokens
from domain_specific_bpe_tokenizer.encoder import encode_text
from domain_specific_bpe_tokenizer.serialization import load_tokenizer, save_tokenizer
from domain_specific_bpe_tokenizer.trainer import train_bpe, train_bpe_from_word_frequencies
from domain_specific_bpe_tokenizer.vocab import build_base_vocab


SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]


class BPETokenizer:

    def __init__(
        self,
        vocab_size: int = 500,
        min_frequency: int = 2,
    ):

        if not isinstance(vocab_size, int):
            raise TypeError("vocab_size must be an integer.")

        if vocab_size <= 260:
            raise ValueError("vocab_size must be greater than 260.")

        self.vocab_size = vocab_size
        self.min_frequency = min_frequency

        self.merges = {}
        self.merge_order = []
        self.vocab = build_base_vocab()

        self.special_tokens = {}

        for token in SPECIAL_TOKENS:
            token_id = len(self.vocab)
            self.vocab[token_id] = token.encode("utf-8")
            self.special_tokens[token] = token_id

        self.unk_id = self.special_tokens["[UNK]"]
        self.pad_id = self.special_tokens["[PAD]"]
        self.bos_id = self.special_tokens["[BOS]"]
        self.eos_id = self.special_tokens["[EOS]"]

    def train(
        self,
        text: str,
    ):

        return train_bpe(
            tokenizer=self,
            text=text,
        )

    def train_from_word_frequencies(
        self,
        word_freqs,
        checkpoint_every=None,
        checkpoint_dir=None,
    ):

        return train_bpe_from_word_frequencies(
            tokenizer=self,
            word_freqs=word_freqs,
            checkpoint_every=checkpoint_every,
            checkpoint_dir=checkpoint_dir,
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

    tokenizer = BPETokenizer(vocab_size=500)

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