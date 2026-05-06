from pathlib import Path
import re
import tarfile
import shutil
import requests
import xml.etree.ElementTree as ET

from tqdm import tqdm


OUTPUT_DIR = Path("/content/drive/MyDrive/final_corpus/pmc_open")
WORK_DIR = Path("/content/pmc_work")

TARGET_WORDS = 426_000_000
CHUNK_WORDS = 1_000_000

BIOC_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pub/wilbur/BioC-PMC"


def clean_text(text: str) -> str:
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\([^\)]*?\d{4}[^\)]*?\)", " ", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def save_chunk(words: list[str], chunk_id: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    path = OUTPUT_DIR / f"chunk_{chunk_id:06d}.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write(" ".join(words))

    print(f"Saved {path} with {len(words):,} words")


def list_bioc_archives() -> list[str]:
    response = requests.get(BIOC_BASE_URL, timeout=60)
    response.raise_for_status()

    archive_names = re.findall(
        r'href="([^"]+\.tar\.gz)"',
        response.text,
    )

    archive_names = sorted(set(archive_names))

    if not archive_names:
        raise RuntimeError(
            "No .tar.gz BioC archives found. "
            "Check BIOC_BASE_URL or FTP listing format."
        )

    print(f"Found {len(archive_names)} BioC archives")

    return archive_names


def download_file(url: str, output_path: Path) -> bool:
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            if r.status_code != 200:
                print(f"Skip {url} | status={r.status_code}")
                return False

            total = int(r.headers.get("content-length", 0))

            with open(output_path, "wb") as f:
                with tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=output_path.name,
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

        return True

    except Exception as e:
        print(f"Download failed: {url} | {e}")
        return False


def extract_text_from_bioc_xml(xml_path: Path) -> list[str]:
    texts = []

    try:
        context = ET.iterparse(xml_path, events=("end",))

        for _, elem in context:
            if elem.tag == "passage":
                passage_texts = []

                for child in elem:
                    if child.tag == "text" and child.text:
                        passage_texts.append(child.text)

                if passage_texts:
                    text = clean_text(" ".join(passage_texts))

                    if text:
                        texts.append(text)

                elem.clear()

    except Exception as e:
        print(f"XML parse failed: {xml_path} | {e}")

    return texts


def process_archive(
    archive_path: Path,
    chunk_words: list[str],
    chunk_id: int,
    total_words: int,
) -> tuple[list[str], int, int]:

    extract_dir = WORK_DIR / archive_path.name.replace(".tar.gz", "")
    extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(extract_dir)

    except Exception as e:
        print(f"Extraction failed: {archive_path} | {e}")
        return chunk_words, chunk_id, total_words

    xml_files = list(extract_dir.rglob("*.xml"))

    for xml_file in xml_files:
        texts = extract_text_from_bioc_xml(xml_file)

        for text in texts:
            words = text.split()

            for word in words:
                chunk_words.append(word)
                total_words += 1

                if len(chunk_words) >= CHUNK_WORDS:
                    save_chunk(chunk_words, chunk_id)

                    chunk_words = []
                    chunk_id += 1

                    print(f"Total collected: {total_words:,}")

                if total_words >= TARGET_WORDS:
                    return chunk_words, chunk_id, total_words

    shutil.rmtree(extract_dir, ignore_errors=True)

    return chunk_words, chunk_id, total_words


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    archive_names = list_bioc_archives()

    chunk_words = []
    chunk_id = 1
    total_words = 0

    for archive_name in archive_names:
        if total_words >= TARGET_WORDS:
            break

        url = f"{BIOC_BASE_URL}/{archive_name}"
        archive_path = WORK_DIR / archive_name

        ok = download_file(url, archive_path)

        if not ok:
            continue

        chunk_words, chunk_id, total_words = process_archive(
            archive_path=archive_path,
            chunk_words=chunk_words,
            chunk_id=chunk_id,
            total_words=total_words,
        )

        archive_path.unlink(missing_ok=True)

        print(f"Finished archive: {archive_name}")
        print(f"Total PMC Open words collected: {total_words:,}")

        if total_words >= TARGET_WORDS:
            break

    if chunk_words:
        save_chunk(chunk_words, chunk_id)

    print(f"\nDone. Total PMC Open words collected: {total_words:,}")


if __name__ == "__main__":
    main()