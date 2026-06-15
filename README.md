# Foti

**Local, private, AI-powered photo management for Linux.** Self-hosted. Open source. No cloud, no telemetry — your photos and the catalog never leave your machine.

Foti scans your photo folders, builds a local catalog with content-aware embeddings, and lets you search by **what's in the picture** — not filenames or hand-typed tags.

```
"beach at sunset"        → finds the photo, no tags required
similar to this one      → CLIP image-to-image search
faces / people           → cluster, name, merge, split
duplicates               → perceptual-hash near-dupes
dominant colour          → "photos that are mostly teal"
best of                  → aesthetic ranking
```

It's a Linux-native, open-source take on the workflows commercial tools like Excire Photo offer. Independent implementation — no reverse-engineering, no shared code — built entirely on open models and a local SQLite catalog.

## Status

**Alpha, and genuinely usable.** It runs a real ~97k-photo catalog day to day: content/text search, image similarity, face clustering + naming, duplicate detection, colour search, aesthetic ranking and XMP sidecar export all work end to end. APIs and the on-disk schema may still change between versions.

The UI today is a single-page web app served by the backend (open `http://127.0.0.1:7777` in a browser). A native desktop shell (Tauri) is on the roadmap, not built yet — see [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Features

- **Semantic search** — type a description, get matching photos (OpenCLIP ViT-L/14, vectors in [sqlite-vec](https://github.com/asg017/sqlite-vec)).
- **Find similar** — pick a photo, get visually-similar ones.
- **People** — on-device face detection + embedding (InsightFace), clustering, and naming with merge/rename/unname. No face data ever leaves the machine.
- **Duplicates** — perceptual-hash clustering with LSH banding, so it stays fast on six-figure catalogs.
- **Colour search** — find photos by dominant colour.
- **Aesthetic ranking** — a "best of" sort (CLIP zero-shot scorer; see note below).
- **Export** — XMP sidecars (tags, caption, rating, people) that Lightroom/digiKam read.
- **Formats** — JPEG/PNG/WebP, HEIC/HEIF (pillow-heif), RAW previews (rawpy/LibRaw).
- **Local-first** — FastAPI daemon bound to `127.0.0.1`; the catalog lives in `~/.local/share/foti/`.

> The aesthetic score is currently a CLIP zero-shot scorer (prompt-contrast), not a trained NIMA/LAION head — cheap, dependency-free, and good enough for "show me the good ones." A stronger head is a drop-in upgrade (the `aesthetic` column is the migration handle).

## Install

Foti needs Python **3.12+**. Face detection needs an ONNX Runtime — install **one** of the two extras:

```bash
# CPU — works on any machine
pipx install "foti-backend[cpu]"

# NVIDIA GPU (CUDA 12) — much faster face indexing
pipx install "foti-backend[gpu]"
```

(`pip install` works too; `pipx`/`uv tool` just keep it isolated.) For a GPU install you also need the CUDA-12 runtime libraries on `LD_LIBRARY_PATH` — see [`docs/DEPLOY.md`](docs/DEPLOY.md).

Then:

```bash
foti-backend scan ~/Pictures        # index a folder
foti-backend serve                  # start the daemon → http://127.0.0.1:7777
```

The first run downloads the open models (OpenCLIP + InsightFace) into a local cache. Nothing is bundled in this repo.

### Docker (self-host)

```bash
docker build -t foti backend/
docker run --rm -p 7777:7777 \
  -v "$HOME/Pictures:/photos:ro" \
  -v foti-data:/data \
  foti
```

See [`docs/DEPLOY.md`](docs/DEPLOY.md) for GPU passthrough, reverse-proxy + HTTP Basic auth, and running as a systemd service.

## Develop

```bash
cd backend
uv sync --extra cpu        # or --extra gpu
uv run foti-backend serve
uv run pytest              # test suite
```

Architecture and the data model: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Licensing

Foti's own code is **AGPL-3.0-or-later** ([`LICENSE`](LICENSE)).

Foti depends on third-party models and libraries that it downloads at runtime — it does **not** redistribute any model weights. Their licenses are yours to honour, and one matters for commercial use: **InsightFace's pretrained face models are licensed for non-commercial / research use only.** If you intend to use Foti commercially, review [`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md) and swap in a face model whose license fits your use.

## Why "Foti"

From the Greek *φωτί* (*phos*, light) — what photography is made of.
