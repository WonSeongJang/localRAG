"""Hybrid search: FTS5 BM25 + vector similarity, fused with RRF."""

import math
import re

from . import db as dbm
from .embed import embed_texts

CANDIDATES = 80
RRF_K = 60
MIN_COVERAGE = 0.65
MIN_SEMANTIC_SIMILARITY = 0.68
_SUFFIXES = ("으로부터", "에서는", "에서", "에게", "으로", "은", "는", "이", "가",
             "을", "를", "에", "의", "과", "와", "도", "로")
_STOP = set("예전에 전에 그때 어떻게 했더라 했지 해줘 찾아줘 찾아주세요 알려줘 알려주세요 "
            "알려줄래 정리해줘 좀 대한 관련 어떤 무엇 있나 있는 있던 그거 이거 것을".split())


def _tokens(text):
    return re.findall(r"[^\W_]+", text.casefold())


def _query_terms(conn, query):
    """Conservative suffix expansion: only accept a stem present in the corpus."""
    groups, weights = [], []
    total = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    for token in dict.fromkeys(_tokens(query)):
        if token in _STOP:
            continue
        forms = {token}
        for suffix in _SUFFIXES:
            if token.endswith(suffix) and len(token) - len(suffix) >= 2:
                stem = token[:-len(suffix)]
                if conn.execute("SELECT 1 FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT 1",
                                (f'"{stem}"',)).fetchone():
                    forms.add(stem)
                    break
        if any(forms & group for group in groups):
            continue
        count = conn.execute("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH ?",
                             (" OR ".join(f'"{t}"' for t in sorted(forms)),)).fetchone()[0]
        groups.append(forms)
        # Unknown words remain significant: never silently discard the subject.
        weights.append(1 + math.log((total + 1) / (count + 1)))
    return groups, weights


def _coverage(text, groups, weights):
    tokens = set(_tokens(text))
    expanded = tokens | {t[:-len(s)] for t in tokens for s in _SUFFIXES
                         if t.endswith(s) and len(t) - len(s) >= 2}
    return sum(w for g, w in zip(groups, weights) if g & expanded) / sum(weights)


def _content_key(text):
    return " ".join(text.split())


def _unique_candidates(fetch, total):
    """Overfetch before ranking until we have distinct content, not duplicate rows."""
    limit = min(CANDIDATES * 2, total)
    while limit:
        rows = fetch(limit)
        unique = {}
        for row in rows:
            unique.setdefault(_content_key(row[1]), row)
        if len(unique) >= CANDIDATES or len(rows) < limit or limit >= total:
            return list(unique.values())[:CANDIDATES]
        limit = min(limit * 2, total)
    return []


def _fts_query(query: str) -> str:
    # Quote each token so FTS5 syntax characters in the query can't break it.
    tokens = re.findall(r"\w+", query, re.UNICODE)
    return " OR ".join(f'"{t}"' for t in tokens)


def _filter_conditions(project, source):
    cond, params = [], []
    if project:
        cond.append("c.project LIKE ?")
        params.append(f"%{project}%")
    if source:
        cond.append("c.source = ?")
        params.append(source)
    return cond, params


def search(conn, query, n=8, project=None, source=None):
    if n <= 0 or not query.strip():
        return []
    conditions, filter_params = _filter_conditions(project, source)
    where_filter = " AND ".join(conditions) or "1"
    total = conn.execute(f"SELECT count(*) FROM chunks c WHERE {where_filter}",
                         filter_params).fetchone()[0]
    if not total:
        return []
    groups, weights = _query_terms(conn, query)
    if not groups:
        return []

    fts_ranks = {}
    fts_q = " OR ".join(f'"{t}"' for t in sorted(set().union(*groups)))
    texts = {}
    if fts_q:
        where = ["chunks_fts MATCH ?", *conditions]
        def fetch_fts(limit):
            return conn.execute(
                "SELECT c.id,c.text FROM chunks_fts JOIN chunks c ON c.id=chunks_fts.rowid "
                f"WHERE {' AND '.join(where)} ORDER BY rank,c.id LIMIT ?",
                (fts_q, *filter_params, limit),
            ).fetchall()
        rows = _unique_candidates(fetch_fts, total)
        fts_ranks = {_content_key(r[1]): i for i, r in enumerate(rows)}
        texts.update({_content_key(r[1]): r[0] for r in rows})

    qvec = embed_texts([query])[0]
    serialized = dbm.serialize(qvec)
    def fetch_vec(limit):
        if not conditions and limit <= 4096:
            return conn.execute(
                "SELECT c.id,c.text,v.distance FROM chunk_vec v "
                "JOIN chunks c ON c.id=v.chunk_id "
                "WHERE v.embedding MATCH ? AND k=? ORDER BY v.distance,c.id",
                (serialized, limit),
            ).fetchall()
        return conn.execute(
            "SELECT c.id,c.text,vec_distance_cosine(v.embedding, ?) AS distance FROM chunk_vec v "
            "JOIN chunks c ON c.id = v.chunk_id "
            f"WHERE {where_filter} ORDER BY distance,c.id LIMIT ?",
            (serialized, *filter_params, limit),
        ).fetchall()
    rows = _unique_candidates(fetch_vec, total)
    vec_ranks = {_content_key(r[1]): i for i, r in enumerate(rows)}
    similarities = {_content_key(r[1]): 1 - r[2] for r in rows}
    for r in rows:
        texts.setdefault(_content_key(r[1]), r[0])

    scores = {}
    evidence = {}
    for cid in set(fts_ranks) | set(vec_ranks):
        coverage = _coverage(cid, groups, weights)
        similarity = similarities.get(cid)
        lexical = coverage >= MIN_COVERAGE
        semantic = similarity is not None and similarity >= MIN_SEMANTIC_SIMILARITY
        if not (lexical or semantic):
            continue
        s = 0.0
        if cid in fts_ranks:
            s += 1.0 / (RRF_K + fts_ranks[cid])
        if cid in vec_ranks:
            s += 1.0 / (RRF_K + vec_ranks[cid])
        scores[cid] = s
        evidence[cid] = (coverage, similarity, "lexical" if lexical else "semantic")

    ranked = sorted(scores, key=lambda key: (-scores[key], texts[key]))
    if not ranked:
        return []
    marks = ",".join("?" * len(ranked))
    rows = conn.execute(
        f"SELECT id, source, project, session, ts, text FROM chunks WHERE id IN ({marks})",
        [texts[key] for key in ranked],
    ).fetchall()
    by_id = {r[0]: r for r in rows}

    # Diversify: drop duplicate texts, cap results per session.
    results, seen_text, per_session = [], set(), {}
    for cid in ranked:
        if texts[cid] not in by_id:
            continue
        _, source, project, session, ts, text = by_id[texts[cid]]
        key = text.strip()
        session_key = (source, project, session)
        if key in seen_text or per_session.get(session_key, 0) >= 2:
            continue
        seen_text.add(key)
        per_session[session_key] = per_session.get(session_key, 0) + 1
        results.append(
            {
                "score": scores[cid],
                "coverage": evidence[cid][0],
                "similarity": evidence[cid][1],
                "match_type": evidence[cid][2],
                "source": source,
                "project": project,
                "session": session,
                "ts": ts,
                "text": text,
            }
        )
        if len(results) >= n:
            break
    return results
