# Foti Roadmap

Checkboxes track what's *built and working*, not what's planned. Anything below
the current frontier is subject to change.

## Done — core engine & features

- [x] FastAPI backend, SQLite + sqlite-vec catalog
- [x] OpenCLIP ViT-L/14 embedding pipeline (CPU + NVIDIA GPU)
- [x] CLI: `foti-backend scan / search / similar / serve / info`
- [x] Folder scan + EXIF/metadata extraction
- [x] HEIC/HEIF support (pillow-heif), RAW previews (rawpy)
- [x] Thumbnail cache (small + large)
- [x] Text search (CLIP text → image) and similar-image search
- [x] Web UI (single-page, served by the daemon) — gallery, search, people, duplicates
- [x] Face detection + embedding (InsightFace buffalo_l), online nearest-centroid clustering
- [x] People: name / rename / merge / unname clusters, with an audit trail
- [x] Perceptual-hash duplicate detection — LSH-banded so it scales to 6-figure catalogs
- [x] Near-duplicate review (compare, ignore)
- [x] Aesthetic ranking ("best of" sort) — CLIP zero-shot scorer
- [x] Colour search (dominant-colour backfill + query)
- [x] XMP sidecar export (tags / caption / rating / people)
- [x] Verified at scale on a ~97k-photo catalog (100% faces / colour / aesthetic coverage)

## Next

- [ ] Native desktop shell (Tauri 2 + Svelte) wrapping the web UI
- [ ] One-command packaging: AppImage / Flatpak / `.deb` / AUR
- [ ] Stronger aesthetic head (trained NIMA/LAION MLP) as a drop-in upgrade
- [ ] Smart Collections (saved searches)
- [ ] Composition search (rule of thirds, leading lines — open research)
- [ ] Folder watch (auto re-scan on changes)
- [ ] Multi-catalog / network catalog (one daemon, multiple UIs)
- [ ] Localization (DE first)

## Explicit non-goals (for now)

- Cloud sync — Foti is local-first by design
- RAW editing — Foti is a *finder*, not an editor
- Reverse-engineering or interop with any commercial product's catalog format
