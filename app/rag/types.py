"""Shapes of the data in the knowledge base. Source and page are required: no citation, no answer."""
from typing import Annotated, Literal

from pydantic import BaseModel, PositiveInt, StringConstraints

SourceName = Literal["pfa", "mhgap", "pmplus"]

# Text is trimmed first, so a string of spaces counts as empty and is rejected.
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Chunk(BaseModel):
    """One piece of one page, as stored in Pinecone."""

    chunk_id: NonEmptyText
    source: SourceName
    page: PositiveInt  # printed page number; this is what a citation shows
    pdf_page: PositiveInt  # position in the PDF file
    page_label_source: Literal["printed", "pdf_index"]  # "pdf_index" = printed number unreadable
    section: str = ""  # many pages have no heading
    text: NonEmptyText

    def embed_text(self) -> str:
        """Text sent to Voyage: a short prefix plus the text. The stored `text` has no prefix."""
        parts = [self.source, self.section, f"p.{self.page}"]
        prefix = " · ".join(part for part in parts if part)
        return f"[{prefix}] {self.text}"


class RetrievedChunk(BaseModel):
    """One search result handed to the guide agent."""

    text: NonEmptyText
    source: SourceName
    page: PositiveInt
    pdf_page: PositiveInt
    section: str = ""
    score: float
