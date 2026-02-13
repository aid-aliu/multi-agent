import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from retrieval.settings import CHUNKS_JSONL, TOP_K
from retrieval.query_index import query_index


@dataclass(frozen=True)
class Evidence:
    idx: int
    rank: int
    distance: float
    doc_name: str
    page: int
    section: str
    text: str

    def citation(self) -> str:
        sec = self.section if self.section and self.section != "NO_SECTION" else "NO_SECTION"
        return f"{self.doc_name} | page {self.page} | section {sec} | chunk {self.idx}"


class EvidenceStore:
    def __init__(self, chunks_jsonl_path):
        self._path = chunks_jsonl_path
        self._chunks: Optional[List[Dict[str, Any]]] = None

    def load(self) -> None:
        if self._chunks is not None:
            return

        chunks = []
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                chunks.append(row)

        self._chunks = chunks

    def get_text_and_meta(self, idx: int) -> Dict[str, Any]:
        self.load()
        assert self._chunks is not None

        if idx < 0 or idx >= len(self._chunks):
            raise IndexError(f"Chunk idx out of range: {idx} (0..{len(self._chunks)-1})")

        return self._chunks[idx]


class ResearchAgent:
    def __init__(
        self,
        store: EvidenceStore,
        top_k: int = TOP_K,
        distance_threshold: float = 0.60,
        max_characters_per_chunk: int = 1400,
    ):
        self.store = store
        self.top_k = top_k
        self.distance_threshold = distance_threshold
        self.max_chars = max_characters_per_chunk

    def _gate_not_found(self, hits: List[Dict[str, Any]]) -> Optional[str]:
        if not hits:
            return "Not found in sources."

        best = hits[0]["distance"]
        if best > self.distance_threshold:
            return "Not found in sources."

        return None

    def search(self, question: str) -> Dict[str, Any]:
        hits = query_index(question, top_k=self.top_k)

        not_found_msg = self._gate_not_found(hits)
        if not_found_msg:
            return {
                "status": "not_found",
                "message": not_found_msg,
                "question": question,
            }

        evidence_blocks: List[Evidence] = []

        for h in hits:
            idx = h["idx"]
            row = self.store.get_text_and_meta(idx)

            text = (row.get("text") or "").strip()
            meta = row.get("metadata") or {}

            doc_name = meta.get("doc_name") or h.get("doc_name") or "UNKNOWN_DOC"
            page = meta.get("page") or h.get("page") or -1
            section = meta.get("section") or h.get("section") or "NO_SECTION"

            if len(text) > self.max_chars:
                text = text[: self.max_chars].rstrip() + "…"

            evidence_blocks.append(
                Evidence(
                    idx=idx,
                    rank=h["rank"],
                    distance=h["distance"],
                    doc_name=str(doc_name),
                    page=int(page),
                    section=str(section),
                    text=text,
                )
            )

        return {
            "status": "found",
            "question": question,
            "evidence": [
                {
                    "rank": e.rank,
                    "distance": e.distance,
                    "citation": e.citation(),
                    "doc_name": e.doc_name,
                    "page": e.page,
                    "section": e.section,
                    "text": e.text,
                }
                for e in evidence_blocks
            ],
        }


if __name__ == "__main__":
    store = EvidenceStore(CHUNKS_JSONL)
    agent = ResearchAgent(store=store)

    q = "management of agitation in dementia patients"
    out = agent.search(q)

    if out["status"] == "not_found":
        print(out["message"])
    else:
        for ev in out["evidence"]:
            print("\n---")
            print(ev["citation"])
            print(f"distance={ev['distance']:.4f}")
            print(ev["text"])
