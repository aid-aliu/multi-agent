import json
import time
from pathlib import Path
from typing import List

import faiss
import numpy as np
import requests

from settings import (
    CHUNKS_JSONL,
    FAISS_DIR,
    OLLAMA_EMBED_MODEL,
)

# -----------------------------
# Ollama embedding helper
# -----------------------------

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Calls Ollama to embed a list of texts.
    Shows progress and timing.
    """
    embeddings = []
    total = len(texts)
    start_time = time.time()

    for i, text in enumerate(texts, 1):
        t0 = time.time()

        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={
                "model": OLLAMA_EMBED_MODEL,
                "prompt": text,
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        embeddings.append(data["embedding"])

        dt = time.time() - t0

        if i % 10 == 0 or i == 1:
            elapsed = time.time() - start_time
            avg = elapsed / i
            remaining = avg * (total - i)

            print(
                f"[{i}/{total}] "
                f"{dt:.2f}s | avg {avg:.2f}s | ETA {remaining/60:.1f} min",
                flush=True
            )

    return np.array(embeddings, dtype="float32")


# -----------------------------
# Main indexing pipeline
# -----------------------------

def main():
    FAISS_DIR.mkdir(parents=True, exist_ok=True)

    texts = []
    metadatas = []

    # 1. Load chunks
    with open(CHUNKS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            texts.append(row["text"])
            metadatas.append(row["metadata"])

    print(f"Loaded {len(texts)} chunks")

    # 2. Embed (this is the slow part)
    vectors = embed_texts(texts)
    dim = vectors.shape[1]

    print(f"Embedding dimension: {dim}")

    # 3. Build FAISS index
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    # 4. Persist index and metadata
    faiss.write_index(index, str(FAISS_DIR / "index.faiss"))

    with open(FAISS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadatas, f, ensure_ascii=False, indent=2)

    print(f"✔ FAISS index saved to {FAISS_DIR}")


if __name__ == "__main__":
    main()
