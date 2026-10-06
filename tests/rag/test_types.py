import pytest
from pydantic import ValidationError

from app.rag.errors import RetrievalError
from app.rag.types import Chunk, RetrievedChunk


def make_chunk(**overrides) -> Chunk:
    values = {
        "chunk_id": "pfa-p12-c0",
        "source": "pfa",
        "page": 12,
        "pdf_page": 14,
        "page_label_source": "printed",
        "section": "Look, listen, link",
        "text": "Approach the person calmly.",
    }
    values.update(overrides)
    return Chunk(**values)


def make_retrieved(**overrides) -> RetrievedChunk:
    values = {
        "text": "Approach the person calmly.",
        "source": "pfa",
        "page": 12,
        "pdf_page": 14,
        "section": "Look, listen, link",
        "score": 0.83,
    }
    values.update(overrides)
    return RetrievedChunk(**values)


def test_valid_chunk_builds():
    chunk = make_chunk()
    assert chunk.source == "pfa"
    assert chunk.page == 12


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_chunk_rejects_empty_text(text):
    with pytest.raises(ValidationError):
        make_chunk(text=text)


def test_chunk_rejects_unknown_source():
    with pytest.raises(ValidationError):
        make_chunk(source="webmd")


@pytest.mark.parametrize("field", ["page", "pdf_page"])
@pytest.mark.parametrize("value", [0, -1])
def test_chunk_rejects_pages_below_one(field, value):
    with pytest.raises(ValidationError):
        make_chunk(**{field: value})


def test_chunk_rejects_unknown_page_label_source():
    with pytest.raises(ValidationError):
        make_chunk(page_label_source="guessed")


def test_embed_text_has_the_prefix_but_stored_text_does_not():
    chunk = make_chunk()
    assert chunk.embed_text().startswith("[pfa · Look, listen, link · p.12]")
    assert chunk.embed_text().endswith(chunk.text)
    assert "[" not in chunk.text


def test_embed_text_without_a_section_has_no_stray_separator():
    prefix = make_chunk(section="").embed_text().split("]")[0]
    assert prefix == "[pfa · p.12"


def test_retrieved_chunk_builds():
    assert make_retrieved().score == 0.83


@pytest.mark.parametrize("field", ["text", "source"])
def test_retrieved_chunk_rejects_empty_text_or_source(field):
    with pytest.raises(ValidationError):
        make_retrieved(**{field: " "})


def test_retrieved_chunk_requires_a_page():
    values = make_retrieved().model_dump()
    del values["page"]
    with pytest.raises(ValidationError):
        RetrievedChunk(**values)


def test_retrieval_error_is_an_exception_with_its_message():
    error = RetrievalError("Pinecone is unreachable")
    assert isinstance(error, Exception)
    assert "Pinecone is unreachable" in str(error)


@pytest.mark.parametrize("field", ["page", "pdf_page"])
def test_retrieved_chunk_rejects_pages_below_one(field):
    with pytest.raises(ValidationError):
        make_retrieved(**{field: 0})
