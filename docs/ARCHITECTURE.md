# Foti Architecture

## Two processes

```
┌──────────────────┐     HTTP localhost:7777     ┌──────────────────┐
│   Tauri UI       │  ───────────────────────▶   │  Backend daemon  │
│   (Rust shell +  │  ◀───────────────────────   │   (Python)       │
│    Svelte web)   │       JSON + thumbs         │                  │
└──────────────────┘                             └────────┬─────────┘
                                                          │
                                                          ▼
                                          ┌────────────────────────────┐
                                          │   SQLite + sqlite-vec      │
                                          │  ~/.local/share/foti/      │
                                          │     catalog.sqlite         │
                                          │     thumbs/                │
                                          │     models/                │
                                          └────────────────────────────┘
```

The daemon is a plain FastAPI server bound to `127.0.0.1`. The UI talks to it over HTTP. This split is intentional:

1. **Lifecycle.** The daemon can keep running scanning + indexing while the UI is closed.
2. **Headless mode.** CLI tools and future automations (e.g. cron-based rescans) hit the same API.
3. **Language fit.** The vision/ML stack lives in Python where it's strongest; the desktop shell lives in a fast native language.

## Data model

SQLite, three logical layers:

```sql
-- 1. Files & metadata (what's on disk)
CREATE TABLE photo (
    id              INTEGER PRIMARY KEY,
    path            TEXT NOT NULL UNIQUE,
    mtime           REAL NOT NULL,
    size_bytes      INTEGER NOT NULL,
    sha256          TEXT,
    width           INTEGER,
    height          INTEGER,
    orientation     INTEGER,
    exif            TEXT,           -- JSON blob
    captured_at     TEXT,           -- ISO 8601, from EXIF when available
    indexed_at      TEXT NOT NULL,
    embedding_ver   INTEGER         -- which model produced the embedding row
);

CREATE INDEX photo_captured_idx ON photo(captured_at);
CREATE INDEX photo_mtime_idx ON photo(mtime);

-- 2. Vector embeddings (what the picture means)
-- sqlite-vec virtual table; one row per photo per model.
CREATE VIRTUAL TABLE photo_embedding USING vec0(
    photo_id        INTEGER PRIMARY KEY,
    embedding       FLOAT[768]      -- ViT-L/14 dim
);

-- Future:
CREATE VIRTUAL TABLE face_embedding USING vec0(
    id              INTEGER PRIMARY KEY,
    photo_id        INTEGER,
    bbox            TEXT,           -- JSON [x,y,w,h]
    embedding       FLOAT[512]      -- InsightFace dim
);

-- 3. User-curated layer (tags, collections, decisions)
CREATE TABLE tag (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE photo_tag (
    photo_id INTEGER NOT NULL,
    tag_id INTEGER NOT NULL,
    PRIMARY KEY (photo_id, tag_id)
);

CREATE TABLE collection (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    is_smart INTEGER NOT NULL DEFAULT 0,
    query_json TEXT                  -- when is_smart=1
);
```

## Scan pipeline

```
folder watcher  ──▶  candidate files (mtime > last_seen OR new)
                          │
                          ▼
                  read EXIF + decode preview
                          │
                          ▼
                  thumbnail cache  ──▶  ~/.local/share/foti/thumbs/<id>.webp
                          │
                          ▼
                  CLIP image encoder (batched)
                          │
                          ▼
                  UPSERT photo + REPLACE INTO photo_embedding
```

Batches of 32 images per GPU call (16 on CPU). Crashes/restarts: the `embedding_ver` column lets us re-embed when the model version bumps.

## Search

Text search: encode query with CLIP text encoder → cosine-rank against `photo_embedding` via `vec_distance_cosine`.

Similar-image search: take the target's embedding directly → same cosine-rank.

Both return `(photo_id, score)` lists; the UI joins back to `photo` for path, thumb URL, and metadata.

## Why these choices

**sqlite-vec over Qdrant/Chroma/pgvector.** No separate process, no migration story, no schema drift between metadata and vectors. The catalog *is* the database. Trade-off: vec0 doesn't shard — fine up to a few million photos.

**OpenCLIP ViT-L/14 over BLIP-2/SigLIP.** Best price/perf on Linux without a server-class GPU. SigLIP could be a v2 swap; the `embedding_ver` column is the migration handle.

**Tauri over Electron.** Linux desktop integration is better, bundle ~10× smaller, no Chromium memory tax.

**FastAPI over Flask.** Async-native: image decode + GPU batch can pipeline cleanly. Pydantic schemas double as API contract for the Tauri client.
