"""Face detection + embedding via InsightFace (buffalo_l pack).

InsightFace ships:
  - SCRFD detector (locates faces, returns bbox + landmarks + det_score)
  - ArcFace recognizer (512-dim L2-normalized embedding per face)

Both run via ONNX Runtime. We prefer the GPU provider on CUDA; if it fails
to load (no CUDA, NVML mismatch, etc.) we fall back to CPU automatically.

Cache directory: ``${FOTI_DATA_DIR}/models/insightface`` so weights are
re-used across daemon restarts.
"""

from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import get_settings
from .imageio import open_image

log = logging.getLogger(__name__)


class FaceModel:
    """Thin wrapper around the InsightFace FaceAnalysis app."""

    def __init__(self) -> None:
        from insightface.app import FaceAnalysis  # heavy import; deferred

        settings = get_settings()
        cache_root = settings.models_cache_dir / "insightface"
        cache_root.mkdir(parents=True, exist_ok=True)
        # InsightFace honors INSIGHTFACE_HOME.
        os.environ.setdefault("INSIGHTFACE_HOME", str(cache_root))

        providers_attempted: list[str] = []
        providers_used: list[str] | None = None
        try:
            providers_attempted = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.app = FaceAnalysis(name="buffalo_l", providers=providers_attempted,
                                    root=str(cache_root))
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            providers_used = self.app.det_model.session.get_providers()
        except Exception as exc:
            log.warning("CUDA face provider failed (%s); falling back to CPU", exc)
            self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"],
                                    root=str(cache_root))
            self.app.prepare(ctx_id=-1, det_size=(640, 640))
            providers_used = ["CPUExecutionProvider"]
        log.info("face model ready (providers=%s)", providers_used)

        self._lock = threading.Lock()

    def detect(self, path: Path) -> list[dict]:
        """Return [{bbox, score, embedding}, ...] for one image."""
        try:
            img = open_image(path)
        except Exception as exc:
            log.warning("face detect: open failed for %s: %s", path, exc)
            return []
        arr = np.array(img)[:, :, ::-1]  # PIL→BGR for insightface
        with self._lock:
            faces = self.app.get(arr)
        out = []
        for f in faces:
            bbox = f.bbox.astype(int).tolist()      # [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox
            embedding = f.normed_embedding.astype("float32")
            out.append({
                "bbox": [x1, y1, x2 - x1, y2 - y1],  # store as [x, y, w, h]
                "score": float(f.det_score),
                "embedding": embedding,
            })
        return out


@lru_cache(maxsize=1)
def get_face_model() -> FaceModel:
    return FaceModel()
