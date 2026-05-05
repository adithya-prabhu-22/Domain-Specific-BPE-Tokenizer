from pathlib import Path


PROJECT_STRUCTURE = {
    "data": [
        "__init__.py",
        "bpe_tokenizer.py",
        "vocab.py",
        "trainer.py",
        "encoder.py",
        "decoder.py",
        "serialization.py",
    ],

    "tests": [
        "__init__.py",
        "test_bpe.py",
        "test_vocab.py",
        "test_trainer.py",
        "test_encoder.py",
        "test_decoder.py",
        "test_serialization.py",
    ],

    "resources": [
        ".gitkeep",
    ],

    "resources/medical_corpus": [
        ".gitkeep",
    ],

    "scripts": [
        "prepare_medical_corpus.py",   # 🔥 NEW FILE
        "train_tokenizer.py",
    ],

    "root_files": [
        "README.md",
        "requirements.txt",
        ".gitignore",
    ],
}


def create_project_structure(base_path: Path, structure: dict) -> None:

    for folder_name, files in structure.items():

        if folder_name == "root_files":

            for file_name in files:

                file_path = base_path / file_name
                file_path.touch(exist_ok=True)

                print(f"[FILE] {file_path}")

            continue

        folder_path = base_path / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)

        print(f"[DIR ] {folder_path}")

        for file_name in files:

            file_path = folder_path / file_name
            file_path.touch(exist_ok=True)

            print(f"[FILE] {file_path}")


if __name__ == "__main__":

    project_root = Path.cwd()

    create_project_structure(
        base_path=project_root,
        structure=PROJECT_STRUCTURE,
    )

    print("\nProject structure created successfully.")