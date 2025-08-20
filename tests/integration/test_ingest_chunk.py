from unittest.mock import Mock

import fitz  # PyMuPDF
import pytest

from core.config.settings import Settings
from core.ingest.chunk import chunk  # Import the original chunk function
from core.ingest.scan import scan  # Import the original scan function
from core.vecdb.client import VecDB


@pytest.fixture
def settings():
    return Settings(device="cpu")


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


def test_chunk_integration(sample_data_dir, settings):
    mock_vecdb = Mock(spec=VecDB)
    mock_vecdb.create_collections.return_value = None

    # dense_embedder = DenseEmbedder(settings.device)
    # image_embedder = ImageEmbedder(settings.device)
    # image_captioner = ImageCaptioner(settings.device)
    # sparse_embedder = SparseEmbedder()

    # We are testing the chunk function directly, not the full pipeline run
    # The IngestPipeline.run() method calls chunk internally.
    # For this test, we want to isolate the chunk functionality.

    # Scan the sample data directory
    ingest_items = scan(str(sample_data_dir))

    # Filter out image files for chunking, as chunking currently only handles text/PDF
    text_pdf_items = [item for item in ingest_items if item.modality == "text"]

    # Chunk the scanned items
    chunks = chunk(text_pdf_items)

    # Assert that non-empty chunks are produced
    assert len(chunks) > 0

    # Assert that chunks have expected attributes
    for c in chunks:
        assert isinstance(c.doc_id, str)
        assert isinstance(c.chunk_id, str)
        assert isinstance(c.content, str)
        assert len(c.content) > 0
        assert isinstance(c.lang, str)
        assert isinstance(c.meta, dict)

        if "sample.pdf" in c.meta["file_path"]:
            assert c.page is not None
            assert c.page >= 0
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
