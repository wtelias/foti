"""Multi-format image loader.

PIL handles JPEG/PNG/WebP/TIFF natively. ``pillow-heif`` adds HEIC/HEIF.
For RAW files (.cr2, .cr3, .nef, .arw, .raf, .dng, .rw2, .orf) we fall
back to ``rawpy``: read the embedded JPEG preview if present (fast), or
demosaic the raw sensor data (slow but always works).

All paths return a PIL Image in RGB mode so the rest of the pipeline
(preprocess for CLIP, thumbnail generation) is format-agnostic.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

from PIL import Image

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    pass

log = logging.getLogger(__name__)

RAW_EXTENSIONS = {".cr2", ".cr3", ".nef", ".arw", ".raf", ".dng", ".rw2", ".orf",
                  ".pef", ".sr2", ".srf", ".srw", ".rwl", ".3fr", ".kdc", ".x3f"}


def _open_raw(path: Path) -> Image.Image:
    """Open a RAW file. Prefer the embedded preview JPEG for speed."""
    try:
        import rawpy  # type: ignore
    except ImportError:
        raise RuntimeError(
            f"RAW file {path.name} requires the `rawpy` package "
            "(install with: uv pip install rawpy or `uv sync --extra raw`)"
        )

    with rawpy.imread(str(path)) as raw:
        # Try the embedded preview first. Most cameras embed a JPEG that's
        # already display-ready, which is ~50× faster than demosaicing.
        try:
            thumb = raw.extract_thumb()
            if thumb.format == rawpy.ThumbFormat.JPEG:
                return Image.open(io.BytesIO(thumb.data)).convert("RGB")
            if thumb.format == rawpy.ThumbFormat.BITMAP:
                return Image.fromarray(thumb.data).convert("RGB")
        except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
            pass

        # Fall back to a full demosaic. Tweak: half-size for speed, since
        # we're going to embed at 224x224 for CLIP anyway.
        rgb = raw.postprocess(use_camera_wb=True, half_size=True,
                              no_auto_bright=False, output_bps=8)
        return Image.fromarray(rgb)


def open_image(path: Path) -> Image.Image:
    """Open *any* supported image format, returning a PIL RGB image."""
    ext = path.suffix.lower()
    if ext in RAW_EXTENSIONS:
        return _open_raw(path)
    # PIL + pillow-heif covers everything else.
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img
