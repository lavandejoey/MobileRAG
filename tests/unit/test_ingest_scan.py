from unittest.mock import Mock

import pytest

from core.config.settings import Settings
from core.ingest.scan import scan  # Import the original scan function
from core.vecdb.client import VecDB


@pytest.fixture
def settings():
    return Settings(device="cpu")


@pytest.fixture
def temp_ingest_dir(tmp_path):
    # Create a temporary directory structure for testing
    (tmp_path / "text_files").mkdir()
    (tmp_path / "image_files").mkdir()

    # Create dummy text files
    (tmp_path / "text_files" / "doc1.txt").write_text("This is document 1.")
    (tmp_path / "text_files" / "doc2.md").write_text("# Document 2\nSome markdown content.")

    # Create dummy image files (empty for now, pHash will handle it)
    # In a real scenario, you'd use actual small image files
    (tmp_path / "image_files" / "img1.png").touch()
    (tmp_path / "image_files" / "img2.jpg").touch()

    return tmp_path


def test_scan_stability(temp_ingest_dir, settings):
    mock_vecdb = Mock(spec=VecDB)
    mock_vecdb.create_collections.return_value = None

    # dense_embedder = DenseEmbedder(settings.device)
    # image_embedder = ImageEmbedder(settings.device)
    # image_captioner = ImageCaptioner(settings.device)
    # sparse_embedder = SparseEmbedder()

    # We are testing the scan function directly, not the full pipeline run
    # The IngestPipeline.run() method calls scan internally.
    # For this test, we want to isolate the scan functionality.

    import time

    # First scan
    items1 = scan(str(temp_ingest_dir))

    # Add a small delay to ensure mtime changes
    time.sleep(0.1)

    # Second scan without changes
    items2 = scan(str(temp_ingest_dir))

    # Assert that the number of items is the same
    assert len(items1) == len(items2)

    # Sort items by path for consistent comparison
    items1_sorted = sorted(items1, key=lambda x: x.local_path)
    items2_sorted = sorted(items2, key=lambda x: x.local_path)

    # Compare all attributes of IngestItem, especially hashes and mtime
    for i in range(len(items1_sorted)):
        item1 = items1_sorted[i]
        item2 = items2_sorted[i]

        assert item1.path == item2.path
        assert item1.doc_id == item2.doc_id
        assert item1.modality == item2.modality
        assert item1.meta["sha256"] == item2.meta["sha256"]
        assert item1.meta["mtime"] == item2.meta["mtime"]

        if item1.modality == "image":
            # pHash might be empty for dummy files, but should be consistent
            assert item1.meta.get("phash") == item2.meta.get("phash")

    # Verify that adding a new file changes the scan result
    (temp_ingest_dir / "text_files" / "doc3.txt").write_text("New document.")
    items3 = scan(str(temp_ingest_dir))
    assert len(items3) == len(items1) + 1

    # Verify that modifying a file changes its hash and mtime
    time.sleep(0.1)  # Ensure mtime changes
    (temp_ingest_dir / "text_files" / "doc1.txt").write_text("This is document 1 updated.")
    items4 = scan(str(temp_ingest_dir))
    items4_sorted = sorted(items4, key=lambda x: x.local_path)

    # Find the modified item
    modified_item1 = next(item for item in items1_sorted if "doc1.txt" in item.path)
    modified_item4 = next(item for item in items4_sorted if "doc1.txt" in item.path)

    assert modified_item1.meta["sha256"] != modified_item4.meta["sha256"]
