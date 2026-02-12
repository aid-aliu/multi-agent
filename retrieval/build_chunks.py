import json
from pathlib import Path

from pypdf import PdfReader

from chunking import load_tokenizer, dynamic_chunk_page
from settings import (
    DATA_RAW_DIR,
    DATA_OUT_DIR,
    CHUNKS_JSONL,
    CHUNK_MAX_TOKENS,
    CHUNK_TOKEN_OVERLAP,
)

def extract_pages(pdf_path: Path):
    reader = PdfReader(str(pdf_path))
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text:
            yield i, text


def main():
    DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer()
    all_chunks = []

    for pdf_path in DATA_RAW_DIR.glob("*.pdf"):
        doc_name = pdf_path.stem

        for page_num, page_text in extract_pages(pdf_path):
            chunks = dynamic_chunk_page(
                tokenizer=tokenizer,
                doc_name=doc_name,
                page_num=page_num,
                page_text=page_text,
                max_tokens=CHUNK_MAX_TOKENS,
                overlap=CHUNK_TOKEN_OVERLAP,
            )

            for c in chunks:
                all_chunks.append({
                    "text": c.text,
                    "metadata": c.metadata,
                })

    with open(CHUNKS_JSONL, "w", encoding="utf-8") as f:
        for row in all_chunks:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"✔ Saved {len(all_chunks)} chunks to {CHUNKS_JSONL}")


if __name__ == "__main__":
    main()
