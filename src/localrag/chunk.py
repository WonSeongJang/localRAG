"""Group turns into user/assistant exchange chunks sized for embedding."""

import re

MAX_CHARS = 2500
OVERLAP = 200


def build_chunks(turns):
    """Yield {"text": str, "ts": str} — one per exchange, split if oversized."""
    exchange = []  # list of (role, text)
    ts = ""
    for t in turns:
        if t["role"] == "note":
            if exchange:
                yield from _emit(exchange, ts)
                exchange = []
            yield from _note_chunks(t["text"], t["ts"])
            continue
        if t["role"] == "user" and exchange and exchange[-1][0] == "assistant":
            yield from _emit(exchange, ts)
            exchange = []
        if not exchange:
            ts = t["ts"]
        exchange.append((t["role"], t["text"]))
    if exchange:
        yield from _emit(exchange, ts)


def _emit(exchange, ts):
    text = "\n".join(f"[{role.upper()}]\n{t}" for role, t in exchange)
    if len(text) <= MAX_CHARS:
        yield {"text": text, "ts": ts}
        return
    start = 0
    while start < len(text):
        piece = text[start : start + MAX_CHARS]
        if len(piece) > 200 or start == 0:  # skip tiny tail fragments
            yield {"text": piece, "ts": ts}
        start += MAX_CHARS - OVERLAP


def _note_chunks(text, ts):
    """Keep document/section headings on every Markdown fragment."""
    headings, body = [], []
    fence = None
    emitted = False

    def emit():
        nonlocal emitted
        prefix = "[NOTE]\n" + "\n".join(label for _, label in headings) + "\n"
        content = "\n".join(body)
        if not content.strip() or all(re.match(r"^#{1,6}\s+", line) or not line.strip() for line in body):
            return
        size = max(1, MAX_CHARS - len(prefix))
        start = 0
        while start < len(content):
            end = min(start + size, len(content))
            if end < len(content):
                boundary = content.rfind("\n", start + size // 2, end)
                if boundary > start:
                    end = boundary + 1
            emitted = True
            yield {"text": prefix + content[start:end], "ts": ts}
            if end == len(content):
                break
            start = max(start + 1, end - min(OVERLAP, size // 4))

    for line in text.splitlines():
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            char = marker[1][0]
            fence = None if fence == char else (fence or char)
        heading = re.match(r"^(#{1,2})\s+(.+)", line) if fence is None and not marker else None
        if heading:
            yield from emit()
            body = []
            level = len(heading[1])
            # The first heading is the filename supplied by the parser.
            # Preserve it even if the document itself contains another H1.
            while len(headings) > 1 and headings[-1][0] >= level:
                headings.pop()
            headings.append((level, line[:180]))
        body.append(line)
    yield from emit()
    if not emitted and text.strip():
        yield from _emit([("note", text)], ts)
