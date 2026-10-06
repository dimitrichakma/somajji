import hashlib

import httpx
import pytest
import yaml

from scripts.download_sources import DownloadError, Source, download_all, load_sources

PDF_BYTES = b"%PDF-1.4 fake pdf for tests"
PDF_SHA = hashlib.sha256(PDF_BYTES).hexdigest()


def make_source(name: str = "pfa", sha256: str = PDF_SHA) -> Source:
    return Source(
        name=name,
        title="Test guide",
        url=f"https://example.org/{name}.pdf",
        landing_page=f"https://example.org/{name}",
        sha256=sha256,
        licence="CC BY-NC-SA 3.0 IGO",
    )


def make_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def ok_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=PDF_BYTES)


def test_load_sources_reads_every_field(tmp_path):
    path = tmp_path / "sources.yaml"
    path.write_text(yaml.safe_dump({"sources": [make_source().__dict__]}))
    [source] = load_sources(path)
    assert source == make_source()


def test_load_sources_rejects_an_entry_with_a_missing_field(tmp_path):
    entry = make_source().__dict__ | {}
    del entry["sha256"]
    path = tmp_path / "sources.yaml"
    path.write_text(yaml.safe_dump({"sources": [entry]}))
    with pytest.raises(DownloadError, match="sha256"):
        load_sources(path)


def test_downloads_each_file_named_after_its_source(tmp_path):
    sources = [make_source("pfa"), make_source("mhgap"), make_source("pmplus")]
    with make_client(ok_handler) as client:
        download_all(sources, tmp_path, client)
    for name in ("pfa", "mhgap", "pmplus"):
        assert (tmp_path / f"{name}.pdf").read_bytes() == PDF_BYTES


def test_skips_a_file_that_already_matches(tmp_path):
    (tmp_path / "pfa.pdf").write_bytes(PDF_BYTES)
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url)
        return httpx.Response(200, content=PDF_BYTES)

    with make_client(handler) as client:
        download_all([make_source()], tmp_path, client)
    assert calls == []


def test_replaces_an_existing_file_with_the_wrong_hash(tmp_path):
    (tmp_path / "pfa.pdf").write_bytes(b"old, damaged")
    with make_client(ok_handler) as client:
        download_all([make_source()], tmp_path, client)
    assert (tmp_path / "pfa.pdf").read_bytes() == PDF_BYTES


def test_hash_mismatch_fails_and_leaves_no_file(tmp_path):
    with make_client(ok_handler) as client:
        with pytest.raises(DownloadError, match="sha256"):
            download_all([make_source(sha256="0" * 64)], tmp_path, client)
    assert not (tmp_path / "pfa.pdf").exists()


def test_http_error_says_how_to_download_by_hand(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403)

    with make_client(handler) as client:
        with pytest.raises(DownloadError) as error:
            download_all([make_source()], tmp_path, client)
    message = str(error.value)
    assert "https://example.org/pfa" in message  # the page to download from
    assert "pfa.pdf" in message  # the file name to save as


def test_network_error_says_how_to_download_by_hand(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no network")

    with make_client(handler) as client:
        with pytest.raises(DownloadError, match="by hand"):
            download_all([make_source()], tmp_path, client)
