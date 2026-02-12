import re
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from transformers import AutoTokenizer


@dataclass
class Chunk:
    """
    Represents one atomic, indexable unit of meaning.
    """
    text: str
    metadata: Dict[str, Any]


# --- Regex patterns for dynamic structure detection ---

# NICE-style recommendation numbers: 1.2.31, 2.1.4, etc.
SECTION_ID_RE = re.compile(r"(?m)^(?P<section>\d+(?:\.\d+)+)\s+")

# Bullet points: -, •, *, 1), 1.
BULLET_RE = re.compile(r"(?m)^\s*(?:[-•*]|\d+\)|\d+\.)\s+")

# Table-like lines: multiple spaces or pipes
TABLE_RE = re.compile(r"(?:\s{2,}|\|)")


def load_tokenizer():
    """
    Loads a tokenizer ONLY for counting tokens.
    This does NOT perform embeddings.
    """
    return AutoTokenizer.from_pretrained("bert-base-uncased")


# ---------- Helper functions ----------

def _token_windows(tokenizer, text: str, max_tokens: int, overlap: int) -> List[str]:
    """
    Splits text into overlapping token windows.
    Used only when a block is too large.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not tokens:
        return []

    windows = []
    step = max_tokens - overlap
    i = 0

    while i < len(tokens):
        window_tokens = tokens[i:i + max_tokens]
        windows.append(tokenizer.decode(window_tokens).strip())
        if i + max_tokens >= len(tokens):
            break
        i += step

    return windows


def _split_by_sections(text: str) -> List[Tuple[str, str]]:
    """
    Splits text by NICE-style section IDs.
    If none exist, returns a single section.
    """
    matches = list(SECTION_ID_RE.finditer(text))
    if not matches:
        return [("NO_SECTION", text.strip())]

    sections = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((m.group("section"), text[start:end].strip()))

    return sections


def _split_section_into_blocks(section_text: str) -> List[str]:
    """
    Splits a section into meaningful blocks:
    - bullet blocks
    - table-like blocks
    - paragraphs
    """
    lines = section_text.splitlines()
    blocks = []
    buffer = []

    def flush():
        nonlocal buffer
        if buffer:
            blocks.append("\n".join(buffer).strip())
            buffer = []

    i = 0
    while i < len(lines):
        line = lines[i]

        if not line.strip():
            flush()
            i += 1
            continue

        if BULLET_RE.match(line):
            flush()
            b = [line]
            i += 1
            while i < len(lines) and (lines[i].startswith(" ") or BULLET_RE.match(lines[i])):
                b.append(lines[i])
                i += 1
            blocks.append("\n".join(b).strip())
            continue

        if TABLE_RE.search(line):
            flush()
            t = [line]
            i += 1
            while i < len(lines) and TABLE_RE.search(lines[i]):
                t.append(lines[i])
                i += 1
            blocks.append("\n".join(t).strip())
            continue

        buffer.append(line)
        i += 1

    flush()
    return blocks


# ---------- Main API ----------

def dynamic_chunk_page(
    tokenizer,
    doc_name: str,
    page_num: int,
    page_text: str,
    max_tokens: int,
    overlap: int,
) -> List[Chunk]:
    """
    Full dynamic chunking pipeline for one PDF page.
    """
    chunks: List[Chunk] = []

    sections = _split_by_sections(page_text)

    for section_id, section_text in sections:
        blocks = _split_section_into_blocks(section_text)

        for block in blocks:
            token_windows = _token_windows(
                tokenizer,
                block,
                max_tokens=max_tokens,
                overlap=overlap,
            )

            for text in token_windows:
                chunks.append(
                    Chunk(
                        text=text,
                        metadata={
                            "doc_name": doc_name,
                            "page": page_num,
                            "section": section_id,
                        },
                    )
                )

    return chunks
