# foti-backend

The backend daemon for **[Foti](https://github.com/wtelias/foti)** — local,
private, AI-powered photo management for Linux.

This package provides the `foti-backend` CLI and HTTP API: folder scanning,
CLIP content/text search, face clustering, duplicate detection, colour search,
aesthetic ranking, and XMP export over a local SQLite + sqlite-vec catalog.

```bash
pipx install "foti-backend[cpu]"     # or [gpu] for NVIDIA CUDA-12
foti-backend scan ~/Pictures
foti-backend serve                   # http://127.0.0.1:7777
```

Full documentation, the web UI, licensing and third-party notices live in the
[project repository](https://github.com/wtelias/foti). Foti's code is
AGPL-3.0-or-later; some models it downloads at runtime (notably InsightFace face
weights) carry their own licenses — see the repo's `THIRD-PARTY-NOTICES.md`.
