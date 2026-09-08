import base64

import pytest
import senza

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def test_image_url():
    a = senza.image_url("https://example.com/i.png")
    assert "image_url" in repr(a)
    assert "https://example.com/i.png" in repr(a)


def test_image_base64_encodes():
    a = senza.image_base64(PNG_1X1)
    assert "image_base64" in repr(a)
    assert "image/png" in repr(a)


def test_image_base64_custom_mime():
    a = senza.image_base64(PNG_1X1, mime_type="image/jpeg")
    assert "image/jpeg" in repr(a)


def test_document_url():
    a = senza.document_url("https://example.com/d.pdf", name="d.pdf")
    assert "d.pdf" in repr(a)


def test_document_url_unnamed():
    a = senza.document_url("https://example.com/d.pdf")
    assert "<unnamed>" in repr(a)


def test_document_file(tmp_path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4 test")
    a = senza.document_file(str(p))
    assert "doc.pdf" in repr(a)


def test_document_file_txt(tmp_path):
    p = tmp_path / "note.txt"
    p.write_text("hello")
    a = senza.document_file(str(p))
    assert "note.txt" in repr(a)


def test_document_file_rejects_unknown_ext(tmp_path):
    p = tmp_path / "data.bin"
    p.write_bytes(b"x")
    with pytest.raises(ValueError, match="unsupported document extension"):
        senza.document_file(str(p))
