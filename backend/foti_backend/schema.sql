-- Foti catalog schema, applied idempotently on every startup.
-- Schema versioning lives in the `meta` table.

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', '1');

CREATE TABLE IF NOT EXISTS photo (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    path            TEXT    NOT NULL UNIQUE,
    mtime           REAL    NOT NULL,
    size_bytes      INTEGER NOT NULL,
    sha256          TEXT,
    width           INTEGER,
    height          INTEGER,
    orientation     INTEGER,
    exif_json       TEXT,
    captured_at     TEXT,
    indexed_at      TEXT    NOT NULL,
    embedding_ver   INTEGER,
    thumb_small     TEXT,
    thumb_large     TEXT,
    phash           TEXT,          -- 64-bit perceptual hash (hex) for duplicate detection
    face_count      INTEGER,       -- cached count from face_detection
    aesthetic       REAL           -- NIMA score 1..10 (null when not scored)
);

CREATE INDEX IF NOT EXISTS photo_captured_idx ON photo(captured_at);
CREATE INDEX IF NOT EXISTS photo_mtime_idx ON photo(mtime);

-- Scan roots (folders the user has added to the catalog).
CREATE TABLE IF NOT EXISTS scan_root (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    path        TEXT NOT NULL UNIQUE,
    added_at    TEXT NOT NULL,
    last_scan   TEXT,
    enabled     INTEGER NOT NULL DEFAULT 1
);

-- User-curated tags.
CREATE TABLE IF NOT EXISTS tag (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS photo_tag (
    photo_id INTEGER NOT NULL REFERENCES photo(id) ON DELETE CASCADE,
    tag_id   INTEGER NOT NULL REFERENCES tag(id)   ON DELETE CASCADE,
    PRIMARY KEY (photo_id, tag_id)
);

-- Collections (manual + smart).
CREATE TABLE IF NOT EXISTS collection (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    is_smart    INTEGER NOT NULL DEFAULT 0,
    query_json  TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS collection_photo (
    collection_id INTEGER NOT NULL REFERENCES collection(id) ON DELETE CASCADE,
    photo_id      INTEGER NOT NULL REFERENCES photo(id) ON DELETE CASCADE,
    added_at      TEXT NOT NULL,
    PRIMARY KEY (collection_id, photo_id)
);

-- Per-photo face detections. One row per face. Face embedding lives in the
-- separate face_embedding vec0 table; this row carries bbox + cluster info.
CREATE TABLE IF NOT EXISTS face_detection (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id    INTEGER NOT NULL REFERENCES photo(id) ON DELETE CASCADE,
    bbox_json   TEXT NOT NULL,         -- JSON [x, y, w, h] in pixels
    det_score   REAL NOT NULL,         -- detector confidence 0..1
    cluster_id  INTEGER,               -- assigned after clustering (NULL = unclustered)
    person_id   INTEGER REFERENCES person(id) ON DELETE SET NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS face_photo_idx ON face_detection(photo_id);
CREATE INDEX IF NOT EXISTS face_cluster_idx ON face_detection(cluster_id);
CREATE INDEX IF NOT EXISTS face_person_idx ON face_detection(person_id);

CREATE TABLE IF NOT EXISTS person (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    created_at  TEXT NOT NULL
);

-- Zero-shot / imported auto-tags (also created lazily by tagging.py for
-- catalogs predating this schema entry).
CREATE TABLE IF NOT EXISTS photo_tag_auto (
    photo_id INTEGER NOT NULL REFERENCES photo(id) ON DELETE CASCADE,
    label    TEXT NOT NULL,
    score    REAL NOT NULL,
    PRIMARY KEY (photo_id, label)
);
CREATE INDEX IF NOT EXISTS photo_tag_auto_label_idx ON photo_tag_auto(label);

-- Duplicate clusters the user dismissed ("these are not duplicates" / "leave
-- them be"). Keyed by sha256 over the sorted member photo-ids.
CREATE TABLE IF NOT EXISTS dupe_ignore (
    cluster_key TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL
);

-- Person rename/merge audit trail (repudiation guard — merges rewrite
-- cluster ids, this records what happened).
CREATE TABLE IF NOT EXISTS person_audit (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    action     TEXT NOT NULL,          -- name | rename | merge | delete
    detail     TEXT NOT NULL,          -- JSON payload
    created_at TEXT NOT NULL
);

-- Scan progress journal: lets the UI report live status, and lets us resume
-- after a crash without re-embedding clean photos.
CREATE TABLE IF NOT EXISTS scan_job (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    root_id     INTEGER REFERENCES scan_root(id) ON DELETE SET NULL,
    state       TEXT    NOT NULL,        -- queued | running | done | failed
    started_at  TEXT,
    finished_at TEXT,
    total       INTEGER NOT NULL DEFAULT 0,
    processed   INTEGER NOT NULL DEFAULT 0,
    error       TEXT
);
