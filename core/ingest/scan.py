import hashlib
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from core.types import IngestItem

# Define common image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}


def calculate_file_hash(file_path: Path) -> str:
    """Calculates the SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def calculate_phash(image_path: Path) -> str:
    """Calculates the perceptual hash (pHash) of an image."""
    try:
        image = Image.open(image_path).convert("L")  # Convert to grayscale
        image = image.resize((8, 8), Image.Resampling.LANCZOS)  # Resize to 8x8
        pixels = list(image.getdata())
        avg = sum(pixels) / len(pixels)
        phash_bits = "".join(["1" if pixel > avg else "0" for pixel in pixels])
        return hex(int(phash_bits, 2))[2:].zfill(16)  # Convert to hex string
    except Exception:
        return ""  # Return empty string if pHash calculation fails


def scan(root_dir: str) -> List[IngestItem]:
    """
    Scans the root directory for files, calculates hashes, and determines modality.
    """
    ingest_items: List[IngestItem] = []
    root_path = Path(root_dir)

    if not root_path.is_dir():
        raise ValueError(f"Root directory not found: {root_dir}")

    for file_path in root_path.rglob("*"):
        if file_path.is_file():
            doc_id = file_path.name  # Simple doc_id for now, can be improved
            modality = "text"
            meta: Dict[str, Any] = {
                "file_path": str(file_path),
                "mtime": int(file_path.stat().st_mtime),
                "sha256": calculate_file_hash(file_path),
            }

            if file_path.suffix.lower() in IMAGE_EXTENSIONS:
                modality = "image"
                meta["phash"] = calculate_phash(file_path)

            ingest_items.append(
                IngestItem(path=str(file_path), doc_id=doc_id, modality=modality, meta=meta)
            )
    return ingest_items
