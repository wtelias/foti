"""Runtime configuration for foti.

All paths default to XDG-compatible locations under the user's home.
Override via environment variables prefixed with ``FOTI_``:

    FOTI_DATA_DIR=/mnt/photos-cache FOTI_PORT=8000 foti-backend serve
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _xdg_data_home() -> Path:
    base = os.environ.get("XDG_DATA_HOME")
    if base:
        return Path(base).expanduser()
    return Path.home() / ".local" / "share"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FOTI_", env_file=".env", extra="ignore")

    data_dir: Path = Field(default_factory=lambda: _xdg_data_home() / "foti")
    host: str = "127.0.0.1"
    port: int = 7777

    # CLIP model — open_clip identifier (name, pretrained-tag).
    clip_model: str = "ViT-L-14"
    clip_pretrained: str = "openai"
    clip_device: str = "auto"            # auto | cpu | cuda
    clip_batch_size: int = 16
    preprocess_workers: int = 8           # threads for parallel JPEG decode + transform

    # Scan tuning
    scan_workers: int = 4
    scan_extensions: tuple[str, ...] = (
        ".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff",
        ".heic", ".heif", ".avif",
        ".cr2", ".cr3", ".nef", ".arw", ".raf", ".dng", ".rw2", ".orf",
    )
    thumbnail_size_small: int = 256
    thumbnail_size_large: int = 1024

    @property
    def catalog_path(self) -> Path:
        return self.data_dir / "catalog.sqlite"

    @property
    def thumbs_dir(self) -> Path:
        return self.data_dir / "thumbs"

    @property
    def models_cache_dir(self) -> Path:
        return self.data_dir / "models"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.thumbs_dir, self.models_cache_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
