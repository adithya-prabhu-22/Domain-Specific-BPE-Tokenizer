def build_base_vocab() -> dict[int, bytes]:

    return {i: bytes([i]) for i in range(256)}


def text_to_bytes(text: str) -> list[int]:

    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    return list(text.encode("utf-8"))