"""SQLite storage: chunks + FTS5 (BM25) + sqlite-vec (embeddings)."""

import sqlite3
import struct
from pathlib import Path

import sqlite_vec

DB_PATH = Path(__file__).resolve().parents[2] / "index.db"
DIM = 1024  # bge-m3


def connect(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS files(
            path TEXT PRIMARY KEY, mtime REAL, source TEXT
        );
        CREATE TABLE IF NOT EXISTS meta(
            key TEXT PRIMARY KEY, value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chunks(
            id INTEGER PRIMARY KEY,
            file TEXT, source TEXT, project TEXT, session TEXT, ts TEXT,
            text TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file);
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text, content='chunks', content_rowid='id', tokenize='unicode61'
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vec USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding float[{DIM}] distance_metric=cosine
        );
        """
    )
    return conn


def serialize(vec):
    return struct.pack(f"{len(vec)}f", *vec)


def get_meta(conn, key: str):
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def set_meta(conn, key: str, value: str):
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", (key, value)
    )


def clear_index(conn):
    """Remove all indexed data while preserving the schema and metadata."""
    conn.execute("DELETE FROM chunk_vec")
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('delete-all')")
    conn.execute("DELETE FROM chunks")
    conn.execute("DELETE FROM files")
    conn.commit()


def get_embeddings_by_text(conn, path=None):
    """Map chunk text -> stored embedding blob for a file's current chunks."""
    return {
        text: emb
        for text, emb in conn.execute(
            "SELECT c.text, v.embedding FROM chunks c "
            "JOIN chunk_vec v ON v.chunk_id = c.id" + (" WHERE c.file=?" if path is not None else ""),
            (path,) if path is not None else (),
        )
    }


def delete_file_chunks(conn, path: str):
    ids = [r[0] for r in conn.execute("SELECT id FROM chunks WHERE file=?", (path,))]
    for cid in ids:
        conn.execute("DELETE FROM chunk_vec WHERE chunk_id=?", (cid,))
        conn.execute(
            "INSERT INTO chunks_fts(chunks_fts, rowid, text) "
            "SELECT 'delete', id, text FROM chunks WHERE id=?",
            (cid,),
        )
    conn.execute("DELETE FROM chunks WHERE file=?", (path,))


def insert_chunks(conn, path, source, project, session, chunks, embeddings):
    for c, emb in zip(chunks, embeddings):
        cur = conn.execute(
            "INSERT INTO chunks(file, source, project, session, ts, text) "
            "VALUES(?,?,?,?,?,?)",
            (path, source, project, session, c["ts"], c["text"]),
        )
        cid = cur.lastrowid
        conn.execute(
            "INSERT INTO chunks_fts(rowid, text) VALUES(?,?)", (cid, c["text"])
        )
        blob = emb if isinstance(emb, (bytes, memoryview)) else serialize(emb)
        conn.execute(
            "INSERT INTO chunk_vec(chunk_id, embedding) VALUES(?,?)",
            (cid, blob),
        )
