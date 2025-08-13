import fitz  # PyMuPDF
import pytest

from core.ingest import Ingestor


@pytest.fixture
def sample_data_dir(tmp_path):
    # Create a temporary data/samples directory
    sample_dir = tmp_path / "data" / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Create a dummy text file
    text_file = sample_dir / "sample.txt"
    text_file.write_text("This is a sample text document.\nIt has multiple lines.")

    # Create a dummy PDF file
    pdf_file = sample_dir / "sample.pdf"
    doc = fitz.open()  # new PDF
    page = doc.new_page()  # new page
    page.insert_text(
        (72, 72), "This is a sample PDF document.\nIt has multiple lines and a second page."
    )
    page = doc.new_page()
    page.insert_text((72, 72), "This is the second page of the PDF.")
    doc.save(str(pdf_file))
    doc.close()

    return sample_dir


def test_chunk_integration(sample_data_dir):
    ingestor = Ingestor()

    # Scan the sample data directory
    ingest_items = ingestor.scan(str(sample_data_dir))

    # Filter out image files for chunking, as chunking currently only handles text/PDF
    text_pdf_items = [item for item in ingest_items if item.modality == "text"]

    # Chunk the scanned items
    chunks = ingestor.chunk(text_pdf_items)

    # Assert that non-empty chunks are produced
    assert len(chunks) > 0

    # Assert that chunks have expected attributes
    for chunk in chunks:
        assert isinstance(chunk.doc_id, str)
        assert isinstance(chunk.chunk_id, str)
        assert isinstance(chunk.content, str)
        assert len(chunk.content) > 0
        assert isinstance(chunk.lang, str)
        assert isinstance(chunk.meta, dict)

        if "sample.pdf" in chunk.meta["file_path"]:
            assert chunk.page is not None
            assert chunk.page >= 0
            # bbox might be None if unstructured doesn't provide it for simple text
            # assert chunk.bbox is not None # This might be too strict for simple text

    # Verify that the text file produced chunks
    text_file_chunks = [c for c in chunks if "sample.txt" in c.meta["file_path"]]
    assert len(text_file_chunks) > 0
    assert "sample text document" in text_file_chunks[0].content

    # Verify that the PDF file produced chunks
    pdf_file_chunks = [c for c in chunks if "sample.pdf" in c.meta["file_path"]]
    assert len(pdf_file_chunks) > 0
    assert any("sample PDF document" in c.content for c in pdf_file_chunks)
    assert any(c.page is not None for c in pdf_file_chunks)
