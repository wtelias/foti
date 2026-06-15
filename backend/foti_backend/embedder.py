"""CLIP image + text embedding.

Wraps open_clip with a singleton model so the daemon loads weights once.
GPU is used when ``clip_device='cuda'`` and torch reports CUDA available;
otherwise everything runs on CPU.

Embeddings are L2-normalized so cosine similarity reduces to a dot product.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .config import get_settings
from .imageio import open_image

log = logging.getLogger(__name__)

# Model schema version — bump when the model identifier or pretrained tag
# changes, so the scanner knows to re-embed.
EMBEDDING_VER = 1


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


class ClipEmbedder:
    """Thread-safe wrapper around an open_clip model."""

    def __init__(self) -> None:
        import open_clip

        settings = get_settings()
        self.device = _resolve_device(settings.clip_device)
        self.batch_size = settings.clip_batch_size

        log.info("Loading CLIP %s/%s on %s",
                 settings.clip_model, settings.clip_pretrained, self.device)

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            settings.clip_model,
            pretrained=settings.clip_pretrained,
            cache_dir=str(settings.models_cache_dir),
        )
        self.tokenizer = open_clip.get_tokenizer(settings.clip_model)
        self.model = self.model.to(self.device).eval()

        self._lock = threading.Lock()
        # Parallel image decode + preprocess. JPEG decode + transform is CPU-
        # bound and releases the GIL inside libjpeg / PIL, so threads scale.
        self._pool = ThreadPoolExecutor(
            max_workers=settings.preprocess_workers,
            thread_name_prefix="clip-pre",
        )

    @torch.inference_mode()
    def encode_images(self, paths: Sequence[Path]) -> np.ndarray:
        """Embed a batch of image paths. Returns (N, EMBEDDING_DIM) float32."""
        if not paths:
            return np.zeros((0, 768), dtype=np.float32)

        def _one(p: Path) -> torch.Tensor:
            try:
                img = open_image(p)
                return self.preprocess(img)
            except Exception as exc:
                log.warning("preprocess failed for %s: %s", p, exc)
                # Push a zero-tensor so the batch indices line up; caller
                # filters by the returned mask.
                return torch.zeros(3, 224, 224)

        tensors = list(self._pool.map(_one, paths))

        batch = torch.stack(tensors).to(self.device)
        with self._lock:
            feats = self.model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return feats.cpu().float().numpy()

    @torch.inference_mode()
    def encode_text(self, queries: Sequence[str]) -> np.ndarray:
        """Embed a batch of text queries."""
        tokens = self.tokenizer(list(queries)).to(self.device)
        with self._lock:
            feats = self.model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return feats.cpu().float().numpy()


@lru_cache(maxsize=1)
def get_embedder() -> ClipEmbedder:
    return ClipEmbedder()
