"""Recursive text chunking.

The splitter tries the biggest natural break first (paragraph, then line,
then sentence, then clause, then word). Pieces are packed into chunks of at
most `chunk_size` characters. Each new chunk starts with the last
`chunk_overlap` characters of the previous chunk (cut at a word boundary),
so an idea that sits on a boundary appears in both chunks.
"""

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


def _validate(chunk_size: int, chunk_overlap: int) -> None:
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
        raise ValueError("chunk_overlap must be zero or a positive integer")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")


def _split_keep_separator(text: str, sep: str) -> list[str]:
    """Split on `sep` but keep the separator at the end of each piece."""
    parts = text.split(sep)
    pieces = [p + sep for p in parts[:-1]] + [parts[-1]]
    return [p for p in pieces if p]


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    """Last resort for text with no separator (for example a very long token)."""
    step = size - overlap
    chunks, i = [], 0
    while True:
        chunks.append(text[i : i + size])
        if i + size >= len(text):
            break
        i += step
    return chunks


def _overlap_tail(text: str, overlap: int) -> str:
    """Last `overlap` characters of `text`, starting at a word boundary."""
    if overlap <= 0 or not text:
        return ""
    tail = text[-overlap:]
    if len(text) > overlap and not text[-overlap - 1].isspace():
        # The slice starts in the middle of a word: drop the partial word.
        for i, ch in enumerate(tail):
            if ch.isspace():
                return tail[i + 1 :]
        return ""
    return tail


def _split(text: str, size: int, overlap: int, separators: list[str]) -> list[str]:
    if len(text) <= size:
        return [text]

    sep, rest = None, []
    for i, candidate in enumerate(separators):
        if candidate in text:
            sep, rest = candidate, separators[i + 1 :]
            break
    if sep is None:
        return _hard_split(text, size, overlap)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for piece in _split_keep_separator(text, sep):
        if len(piece) > size:
            # Too big on its own: close the current chunk, split it further.
            if current:
                chunks.append("".join(current))
                current, current_len = [], 0
            chunks.extend(_split(piece, size, overlap, rest))
            continue

        if current and current_len + len(piece) > size:
            finished = "".join(current)
            chunks.append(finished)
            tail = _overlap_tail(finished, overlap)
            if tail and len(tail) + len(piece) <= size:
                current, current_len = [tail], len(tail)
            else:
                current, current_len = [], 0

        current.append(piece)
        current_len += len(piece)

    if current:
        chunks.append("".join(current))
    return chunks


def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separators: list[str] | None = None,
) -> list[str]:
    """Split one text into chunks of at most `chunk_size` characters."""
    _validate(chunk_size, chunk_overlap)
    if text is None or not str(text).strip():
        return []
    seps = DEFAULT_SEPARATORS if separators is None else list(separators)
    chunks = _split(str(text).strip(), chunk_size, chunk_overlap, seps)
    return [c.strip() for c in chunks if c.strip()]


def chunk_documents(
    docs,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    text_key: str = "text",
    id_key: str = "id",
    metadata_keys: tuple = ("title", "category", "published"),
):
    """Chunk a list of document dicts.

    Returns (chunks, skipped). Each chunk is a dict with:
    id ("<doc_id>::<index>"), doc_id, chunk_index, n_chunks, text, plus the
    metadata keys copied from the document. `skipped` counts documents with
    no id or no text.
    """
    chunks, skipped = [], 0
    for doc in docs:
        doc_id = doc.get(id_key)
        if doc_id is None or str(doc_id).strip() == "":
            skipped += 1
            continue
        pieces = split_text(doc.get(text_key), chunk_size, chunk_overlap)
        if not pieces:
            skipped += 1
            continue
        for i, piece in enumerate(pieces):
            record = {
                "id": f"{doc_id}::{i}",
                "doc_id": str(doc_id),
                "chunk_index": i,
                "n_chunks": len(pieces),
                "text": piece,
            }
            for key in metadata_keys:
                if key in doc:
                    record[key] = doc[key]
            chunks.append(record)
    return chunks, skipped
