"""Incremental indexing of Claude and Codex session files."""

import sys
import os
import sqlite3
from datetime import datetime
from pathlib import Path

from . import db as dbm
from .chunk import MAX_CHARS, OVERLAP, build_chunks
from .embed import MODEL, embed_texts
from .parsers import (
    CLAUDE_ROOT,
    CODEX_ROOT,
    DOCS_ROOT,
    OBSIDIAN_ROOT,
    parse_claude_file,
    parse_claude_memory_file,
    parse_codex_file,
    parse_docs_file,
    parse_obsidian_file,
)

SOURCES = {
    "claude": (CLAUDE_ROOT, "*.jsonl", parse_claude_file),
    "codex": (CODEX_ROOT, "*.jsonl", parse_codex_file),
    "obsidian": (OBSIDIAN_ROOT, "*.md", parse_obsidian_file),
    "docs": (DOCS_ROOT, "*.md", parse_docs_file),
    "memory": (CLAUDE_ROOT, "memory/*.md", parse_claude_memory_file),
}
# 원본이 지워져도 색인을 유지하는 소스(세션 기록은 도구가 자동 삭제함).
ARCHIVE_SOURCES = {"claude", "codex"}

# 색인에서 제외할 잡음 폴더(숨김 폴더는 이미 아래에서 스킵됨).
_SKIP_DIRS = {"node_modules", ".venv", "venv", "__pycache__", "dist", "target", "build"}
INDEX_FORMAT_VERSION = "2"
INDEX_SIGNATURE_KEY = "index_signature"


def index_signature():
    """Fingerprint settings that require rebuilding stored chunks or vectors."""
    return f"{INDEX_FORMAT_VERSION}:{MODEL}:{dbm.DIM}:{MAX_CHARS}:{OVERLAP}"


def prepare_index(conn, rebuild=False):
    """Adopt, validate, or explicitly rebuild the persisted index format."""
    expected = index_signature()
    current = dbm.get_meta(conn, INDEX_SIGNATURE_KEY)
    if rebuild:
        # Reparse every source, but keep vectors for byte-identical text when
        # the embedding model/dimension contract has not changed.
        if not current or current.split(":")[1:3] != expected.split(":")[1:3]:
            archived = _archived_orphans(conn)
            dbm.clear_index(conn)
            _reembed_archived(conn, archived)
        return
    if current is None:
        if conn.execute("SELECT 1 FROM chunks LIMIT 1").fetchone():
            raise SystemExit("Unversioned index requires: uv run localrag index --rebuild")
        dbm.set_meta(conn, INDEX_SIGNATURE_KEY, expected)
        conn.commit()
    elif current != expected:
        raise SystemExit(
            "Index format or embedding settings changed. "
            "Rebuild with: uv run localrag index --rebuild"
        )


def _archived_orphans(conn):
    """Archive-source files whose originals are gone, with their stored chunks."""
    marks = ",".join("?" * len(ARCHIVE_SOURCES))
    files = conn.execute(
        f"SELECT path, mtime, source FROM files WHERE source IN ({marks})", sorted(ARCHIVE_SOURCES)
    ).fetchall()
    return [
        (path, mtime, source, conn.execute(
            "SELECT project, session, ts, text FROM chunks WHERE file=? ORDER BY id", (path,)
        ).fetchall())
        for path, mtime, source in files
        if not Path(path).exists()
    ]


def _reembed_archived(conn, archived):
    """Re-embed archived chunks from stored text, since originals cannot be reparsed."""
    for path, mtime, source, rows in archived:
        if rows:
            embeddings = embed_texts([text for _, _, _, text in rows])
            for (project, session, ts, text), emb in zip(rows, embeddings):
                dbm.insert_chunks(conn, path, source, project, session, [{"ts": ts, "text": text}], [emb])
        conn.execute("INSERT INTO files(path, mtime, source) VALUES(?,?,?)", (path, mtime, source))
        conn.commit()


def discover_files(source=None):
    """Return eligible source files grouped with their parser source name."""
    found = []
    for name, (root, pattern, _) in SOURCES.items():
        if source and name != source:
            continue
        if not root.exists():
            continue
        for p in root.rglob(pattern):
            rel_parts = p.relative_to(root).parts
            if any(part.startswith(".") for part in rel_parts):
                continue
            if any(part in _SKIP_DIRS for part in rel_parts):
                continue
            found.append((p, name))
    return found


def find_stale(conn, source=None, force=False):
    """Return [(path, mtime, source)] of source files new or changed since last index."""
    known = dict(conn.execute("SELECT path, mtime FROM files"))
    stale = []
    for p, name in discover_files(source):
        mtime = p.stat().st_mtime
        if force or known.get(str(p)) != mtime:
            stale.append((p, mtime, name))
    return stale


def find_deleted(conn, source=None):
    """Return indexed paths whose source files no longer exist or are excluded.

    Session sources are an archive: Claude/Codex delete old transcripts on their
    own, so their chunks stay indexed after the original file disappears.
    """
    active_sources = {
        name
        for name, (root, _, _) in SOURCES.items()
        if (not source or name == source) and root.exists() and name not in ARCHIVE_SOURCES
    }
    if not active_sources:
        return []
    present = {str(p) for p, name in discover_files(source) if name in active_sources}
    marks = ",".join("?" * len(active_sources))
    rows = conn.execute(
        f"SELECT path FROM files WHERE source IN ({marks})", sorted(active_sources)
    ).fetchall()
    return [path for (path,) in rows if path not in present]


def prune_deleted(conn, source=None):
    deleted = find_deleted(conn, source)
    for path in deleted:
        dbm.delete_file_chunks(conn, path)
        conn.execute("DELETE FROM files WHERE path=?", (path,))
    if deleted:
        conn.commit()
    return len(deleted)


def index(conn, source=None, quiet=False, rebuild=False):
    if rebuild and source:
        raise ValueError("A full rebuild cannot be combined with a source filter")
    prepare_index(conn, rebuild=rebuild)
    pruned = prune_deleted(conn, source)
    if pruned and not quiet:
        print(f"pruned {pruned} deleted files", file=sys.stderr)
    stale = find_stale(conn, source, force=rebuild)
    if not stale:
        if rebuild:
            dbm.set_meta(conn, INDEX_SIGNATURE_KEY, index_signature())
            conn.commit()
        return 0
    parsers = {name: fn for name, (_, _, fn) in SOURCES.items()}
    shared_embeddings = dbm.get_embeddings_by_text(conn) if rebuild else {}
    done = 0
    for p, mtime, name in stale:
        project, session, turns = parsers[name](p)
        chunks = list(build_chunks(turns))
        # chunk-level incremental: reuse embeddings of unchanged chunks
        # from the previous version of this file, embed only new texts
        emb_by_text = dbm.get_embeddings_by_text(conn, str(p))
        emb_by_text.update({c['text']: shared_embeddings[c['text']] for c in chunks
                            if c['text'] in shared_embeddings})
        new_texts = [
            t for t in dict.fromkeys(c["text"] for c in chunks) if t not in emb_by_text
        ]
        if new_texts:
            emb_by_text.update(zip(new_texts, embed_texts(new_texts)))
            shared_embeddings.update({t: emb_by_text[t] for t in new_texts})
        embeddings = [emb_by_text[c["text"]] for c in chunks]
        dbm.delete_file_chunks(conn, str(p))
        dbm.insert_chunks(conn, str(p), name, project or "", session, chunks, embeddings)
        conn.execute(
            "INSERT OR REPLACE INTO files(path, mtime, source) VALUES(?,?,?)",
            (str(p), mtime, name),
        )
        conn.commit()
        done += 1
        if not quiet:
            print(
                f"\r[{done}/{len(stale)}] {name}: {p.name[:50]} "
                f"({len(chunks)} chunks, {len(new_texts)} new)   ",
                end="",
                file=sys.stderr,
                flush=True,
            )
    if not quiet:
        print(file=sys.stderr)
    if rebuild:
        dbm.set_meta(conn, INDEX_SIGNATURE_KEY, index_signature())
        conn.commit()
    return done


def rebuild_index(db_path=dbm.DB_PATH):
    """Build and verify a sibling database before atomically publishing it.

    Keep the original as a timestamped backup. An interrupted build can resume
    from its sibling as long as the live index has not changed.
    """
    db_path = Path(db_path)
    stamp = str(db_path.stat().st_mtime_ns) if db_path.exists() else "new"
    candidate = db_path.with_suffix(".rebuild.db")
    new_candidate = not candidate.exists()
    if new_candidate and db_path.exists():
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as source:
            with sqlite3.connect(candidate) as dest:
                source.backup(dest)
    conn = dbm.connect(candidate)
    try:
        origin = dbm.get_meta(conn, "rebuild_origin")
        if not new_candidate and origin is not None and origin != stamp:
            raise SystemExit(f"Live index changed; preserve/inspect {candidate} before restarting rebuild.")
        dbm.set_meta(conn, "rebuild_origin", stamp)
        conn.commit()
        done = index(conn, rebuild=True)
        if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("Rebuilt database failed integrity_check")
        if conn.execute("SELECT count(*) FROM chunks").fetchone()[0] != conn.execute(
                "SELECT count(*) FROM chunk_vec").fetchone()[0]:
            raise RuntimeError("Rebuilt database has a vector count mismatch")
        if conn.execute("SELECT 1 FROM chunks c LEFT JOIN chunk_vec v ON v.chunk_id=c.id "
                        "WHERE v.chunk_id IS NULL LIMIT 1").fetchone():
            raise RuntimeError("Rebuilt database has a missing vector")
        conn.execute("INSERT INTO chunks_fts(chunks_fts,rank) VALUES('integrity-check',1)")
        conn.commit()
    finally:
        conn.close()
    now_stamp = str(db_path.stat().st_mtime_ns) if db_path.exists() else "new"
    if now_stamp != stamp:
        raise SystemExit(f"Live index changed during rebuild; candidate kept at {candidate}")
    backup = db_path.with_name(f"{db_path.stem}.backup-{datetime.now():%Y%m%d-%H%M%S-%f}.db")
    if db_path.exists():
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as source:
            with sqlite3.connect(backup) as dest:
                source.backup(dest)
    if (str(db_path.stat().st_mtime_ns) if db_path.exists() else "new") != stamp:
        raise SystemExit(f"Live index changed during backup; candidate kept at {candidate}")
    os.replace(candidate, db_path)
    return done, backup if stamp != "new" else None
