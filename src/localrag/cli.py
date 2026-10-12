"""localrag CLI: index and search Claude/Codex session history."""

import argparse

from . import db as dbm
from .embed import check_ollama
from .indexer import index, prepare_index, rebuild_index
from .search import search

SNIPPET = 700


def main():
    ap = argparse.ArgumentParser(prog="localrag")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_idx = sub.add_parser("index", help="index new/changed session files")
    p_idx.add_argument("--source", choices=["claude", "codex", "obsidian", "docs", "memory"])
    p_idx.add_argument(
        "--rebuild", action="store_true", help="rebuild, verify, and replace the index with a backup"
    )

    p_s = sub.add_parser("search", help="search past sessions")
    p_s.add_argument("query")
    p_s.add_argument("-n", type=int, default=8, help="number of results")
    p_s.add_argument("--project", help="filter by project path substring")
    p_s.add_argument("--source", choices=["claude", "codex", "obsidian", "docs", "memory"])
    p_s.add_argument("--full", action="store_true", help="print full chunk text")
    p_s.add_argument("--no-refresh", action="store_true", help="skip incremental indexing")

    sub.add_parser("stats", help="show index statistics")

    args = ap.parse_args()
    if args.cmd == "index" and args.rebuild and args.source:
        ap.error("--rebuild cannot be combined with --source")
    if args.cmd == "search" and args.n <= 0:
        ap.error("-n must be positive")
    if args.cmd == "index" and args.rebuild:
        check_ollama()
        done, backup = rebuild_index()
        print(f"rebuilt {done} files; backup: {backup}")
        return
    conn = dbm.connect()

    if args.cmd == "index":
        check_ollama()
        prepare_index(conn)
        done = index(conn, source=args.source, rebuild=args.rebuild)
        total = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        print(f"indexed {done} files, {total} chunks total")

    elif args.cmd == "search":
        check_ollama()
        prepare_index(conn)
        if not args.no_refresh:
            index(conn, quiet=False)
        results = search(
            conn, args.query, n=args.n, project=args.project, source=args.source
        )
        if not results:
            print("no sufficiently relevant results; try a specific term, --source, or --project")
            return
        for i, r in enumerate(results, 1):
            date = (r["ts"] or "")[:10]
            print(f"\n=== {i}. [{r['source']}] {r['project']} ({date})")
            print(f"    session: {r['session']}")
            similarity = r.get("similarity")
            print(f"    match: {r['match_type']}, coverage: {r['coverage']:.0%}"
                  + (f", similarity: {similarity:.3f}" if similarity is not None else ""))
            text = r["text"] if args.full else r["text"][:SNIPPET]
            if not args.full and len(r["text"]) > SNIPPET:
                text += " …"
            print(text)

    elif args.cmd == "stats":
        for src, files, chunks in conn.execute(
            """SELECT f.source, count(DISTINCT f.path),
                      (SELECT count(*) FROM chunks c WHERE c.source = f.source)
               FROM files f GROUP BY f.source"""
        ):
            print(f"{src}: {files} files, {chunks} chunks")


if __name__ == "__main__":
    main()
