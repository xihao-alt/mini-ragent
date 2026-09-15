"""Turn extracted document text into retrievable chunks."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Chunk:
    """A text chunk and the metadata needed to trace it to its source."""

    id: int
    text: str
    source: str

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def chunk_text(text: str, source: str = "", max_chars: int = 500) -> list[Chunk]:
    """Split text on extracted paragraphs/lines, combining only short sections."""
    if max_chars < 1:
        raise ValueError("max_chars must be positive")

    sections = [line.strip() for line in text.splitlines() if line.strip()]
    chunks: list[str] = []
    current = ""

    for section in sections:
        if len(section) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(
                section[start : start + max_chars].strip()
                for start in range(0, len(section), max_chars)
            )
        elif not current:
            current = section
        elif len(current) + 1 + len(section) <= max_chars:
            current = f"{current} {section}"
        else:
            chunks.append(current)
            current = section

    if current:
        chunks.append(current)

    return [Chunk(id=index, text=value, source=source) for index, value in enumerate(chunks)]
