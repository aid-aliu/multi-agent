import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import logging

from retrieval.settings import CHUNKS_JSONL, TOP_K
from retrieval.query_index import query_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Evidence:
    """Immutable evidence object with citation generation."""
    idx: int
    rank: int
    distance: float
    doc_name: str
    page: int
    section: str
    text: str

    def citation(self) -> str:
        """Generate a formatted citation string."""
        # Clean up section display
        sec = self.section if self.section and self.section != "NO_SECTION" else "NO_SECTION"
        return f"{self.doc_name} | page {self.page} | section {sec} | chunk {self.idx}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with citation included."""
        return {
            "idx": self.idx,
            "rank": self.rank,
            "distance": self.distance,
            "citation": self.citation(),
            "doc_name": self.doc_name,
            "page": self.page,
            "section": self.section,
            "text": self.text,
        }


class EvidenceStore:
    """Loads and provides access to chunked document data."""

    def __init__(self, chunks_jsonl_path: str):
        self._path = chunks_jsonl_path
        self._chunks: Optional[List[Dict[str, Any]]] = None

    def load(self) -> None:
        """Lazy load chunks from JSONL file."""
        if self._chunks is not None:
            return

        chunks = []
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        chunks.append(row)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping malformed JSON on line {line_num}: {e}")
                        continue
        except FileNotFoundError:
            logger.error(f"Chunks file not found: {self._path}")
            raise
        except Exception as e:
            logger.error(f"Error loading chunks: {e}")
            raise

        if not chunks:
            logger.warning(f"No chunks loaded from {self._path}")

        self._chunks = chunks
        logger.info(f"Loaded {len(chunks)} chunks from {self._path}")

    def get_text_and_meta(self, idx: int) -> Dict[str, Any]:
        """
        Retrieve chunk by index.

        Args:
            idx: Zero-based chunk index

        Returns:
            Dict with 'text' and 'metadata' keys

        Raises:
            IndexError: If idx is out of range
        """
        self.load()
        assert self._chunks is not None

        if idx < 0 or idx >= len(self._chunks):
            raise IndexError(
                f"Chunk idx out of range: {idx} (valid range: 0..{len(self._chunks) - 1})"
            )

        return self._chunks[idx]

    def get_chunk_count(self) -> int:
        """Return total number of chunks available."""
        self.load()
        return len(self._chunks) if self._chunks else 0


class ResearchAgent:
    """
    Retrieves and ranks evidence from document store based on semantic similarity.

    This agent is responsible for:
    - Querying the vector index
    - Filtering low-quality results
    - Enriching hits with metadata
    - Returning structured evidence with citations
    """

    def __init__(
            self,
            store: EvidenceStore,
            top_k: int = TOP_K,
            distance_threshold: float = 0.60,
            max_characters_per_chunk: int = 1400,
    ):
        """
        Args:
            store: EvidenceStore instance for loading chunk text/metadata
            top_k: Maximum number of results to retrieve
            distance_threshold: Max cosine distance (0-1); higher = stricter
            max_characters_per_chunk: Truncate text beyond this length
        """
        self.store = store
        self.top_k = top_k
        self.distance_threshold = distance_threshold
        self.max_chars = max_characters_per_chunk

    def _gate_not_found(self, hits: List[Dict[str, Any]]) -> Optional[str]:
        """
        Determine if results are too poor to use.

        Returns:
            Error message if gated, None if results are acceptable
        """
        if not hits:
            return "Not found in sources."

        best_distance = hits[0]["distance"]
        if best_distance > self.distance_threshold:
            logger.info(
                f"Best result distance {best_distance:.4f} exceeds threshold {self.distance_threshold}"
            )
            return "Not found in sources."

        return None

    def search(self, question: str) -> Dict[str, Any]:
        """
        Execute research query and return structured evidence.

        Args:
            question: Natural language query

        Returns:
            Dict with:
            - status: "found" or "not_found"
            - evidence: List of evidence dicts (if found)
            - message: Error message (if not found)
            - question: Original query
        """
        if not question or not question.strip():
            logger.warning("Empty question provided to search")
            return {
                "status": "not_found",
                "message": "Empty query provided.",
                "question": question,
                "evidence": [],
            }

        logger.info(f"Research query: {question}")

        try:
            hits = query_index(question, top_k=self.top_k)
        except Exception as e:
            logger.error(f"Query index failed: {e}")
            return {
                "status": "error",
                "message": f"Vector search failed: {str(e)}",
                "question": question,
                "evidence": [],
            }

        # Check quality gate
        not_found_msg = self._gate_not_found(hits)
        if not_found_msg:
            logger.info(f"Quality gate triggered: {not_found_msg}")
            return {
                "status": "not_found",
                "message": not_found_msg,
                "question": question,
                "evidence": [],
            }

        # Enrich hits with full text and metadata
        evidence_blocks: List[Evidence] = []

        for h in hits:
            idx = h.get("idx")
            if idx is None:
                logger.warning(f"Hit missing 'idx': {h}")
                continue

            try:
                row = self.store.get_text_and_meta(idx)
            except IndexError as e:
                logger.error(f"Failed to fetch chunk {idx}: {e}")
                continue

            # Extract text
            text = (row.get("text") or "").strip()
            if not text:
                logger.warning(f"Chunk {idx} has no text, skipping")
                continue

            # Extract metadata (may be nested or flat)
            meta = row.get("metadata") or {}

            # Prioritize metadata fields, fall back to hit-level fields
            doc_name = (
                    meta.get("doc_name")
                    or h.get("doc_name")
                    or "UNKNOWN_DOC"
            )
            page = meta.get("page") or h.get("page")
            if page is None:
                page = -1  # Indicate missing page

            section = (
                    meta.get("section")
                    or h.get("section")
                    or "NO_SECTION"
            )

            # Truncate long text
            if len(text) > self.max_chars:
                text = text[: self.max_chars].rstrip() + "…"

            evidence_blocks.append(
                Evidence(
                    idx=idx,
                    rank=h.get("rank", len(evidence_blocks) + 1),
                    distance=h.get("distance", 1.0),
                    doc_name=str(doc_name),
                    page=int(page),
                    section=str(section),
                    text=text,
                )
            )

        if not evidence_blocks:
            logger.warning("All hits failed to produce valid evidence")
            return {
                "status": "not_found",
                "message": "No valid evidence could be extracted from search results.",
                "question": question,
                "evidence": [],
            }

        logger.info(f"Found {len(evidence_blocks)} evidence blocks")

        return {
            "status": "found",
            "question": question,
            "evidence": [e.to_dict() for e in evidence_blocks],
        }


if __name__ == "__main__":
    # Example usage
    try:
        store = EvidenceStore(CHUNKS_JSONL)
        agent = ResearchAgent(store=store)

        q = "management of agitation in dementia patients"
        out = agent.search(q)

        if out["status"] == "not_found":
            print(f"❌ {out['message']}")
        elif out["status"] == "error":
            print(f"⚠️  Error: {out['message']}")
        else:
            print(f"✅ Found {len(out['evidence'])} evidence blocks\n")
            for ev in out["evidence"]:
                print("---")
                print(f"[E{ev['rank']}] {ev['citation']}")
                print(f"Distance: {ev['distance']:.4f}")
                print(f"{ev['text'][:200]}...")
                print()
    except Exception as e:
        logger.error(f"Example failed: {e}")
        import traceback

        traceback.print_exc()