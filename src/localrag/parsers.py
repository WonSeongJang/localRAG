"""Parse Claude Code and Codex session JSONL files into normalized turns.

A turn is a dict: {"role": "user"|"assistant", "text": str, "ts": str}.
Each parser returns (project_path, session_id, [turns]).
"""

import json
import os
import re
from collections import Counter
from pathlib import Path


def _load_env_file():
    """Read KEY=VALUE lines from a local, gitignored .env next to the project."""
    path = Path(__file__).resolve().parents[2] / ".env"
    if not path.exists():
        return {}
    pairs = (line.split("=", 1) for line in path.read_text().splitlines() if "=" in line and not line.startswith("#"))
    return {k.strip(): os.path.expanduser(v.strip().strip("\"'")) for k, v in pairs}


_ENV_FILE = _load_env_file()


def _root(env_key, default):
    """Allow each source root to be relocated without touching the code."""
    return Path(os.environ.get(env_key) or _ENV_FILE.get(env_key) or default)


CLAUDE_ROOT = _root("LOCALRAG_CLAUDE_ROOT", Path.home() / ".claude" / "projects")
CODEX_ROOT = _root("LOCALRAG_CODEX_ROOT", Path.home() / ".codex" / "sessions")
OBSIDIAN_ROOT = _root("LOCALRAG_OBSIDIAN_ROOT", Path.home() / "obsidian")
DOCS_ROOT = _root("LOCALRAG_DOCS_ROOT", Path.home() / "docs")

_SYSTEM_REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)
# Harness blocks can be prepended to a real user message. Remove the blocks,
# not the whole message, so text after the instructions remains searchable.
_HARNESS_BLOCKS = re.compile(
    r"^\s*(?:# AGENTS\.md instructions[^\n]*\n\s*)?"
    r"<(recommended_plugins|environment_context|user_instructions|INSTRUCTIONS|"
    r"skills_instructions|permissions instructions|collaboration_mode)>.*?</\1>",
    re.DOTALL,
)

# User-message prefixes that are harness noise, not the human talking.
_NOISE_PREFIXES = (
    "<command-name>",
    "<local-command-stdout>",
    "<local-command-caveat>",
    "<environment_context>",
    "<user_instructions>",
    "<turn_aborted",
    "<task-notification>",
    "Caveat: The messages below",
    "[Request interrupted",
    "# AGENTS.md instructions",
)


def _clean_user_text(text: str) -> str:
    text = _SYSTEM_REMINDER.sub("", text).strip()
    while match := _HARNESS_BLOCKS.match(text):
        text = text[match.end():]
    text = text.strip()
    if not text or text.startswith(_NOISE_PREFIXES):
        return ""
    return text


def _iter_jsonl(path: Path):
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def parse_claude_file(path: Path):
    project = None
    session_id = path.stem
    turns = []
    for d in _iter_jsonl(path):
        if d.get("isSidechain") or d.get("isMeta"):
            continue
        dtype = d.get("type")
        if dtype not in ("user", "assistant"):
            continue
        if project is None and d.get("cwd"):
            project = d["cwd"]
        msg = d.get("message") or {}
        content = msg.get("content")
        if dtype == "user":
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                # Keep only typed-text blocks; tool_result blocks are machine output.
                text = "\n".join(
                    b.get("text", "") for b in content if b.get("type") == "text"
                )
            else:
                continue
            text = _clean_user_text(text)
            if text:
                turns.append({"role": "user", "text": text, "ts": d.get("timestamp", "")})
        elif dtype == "assistant":
            if not isinstance(content, list):
                continue
            text = "\n".join(
                b.get("text", "") for b in content if b.get("type") == "text"
            ).strip()
            if text and not text.startswith("API Error:"):
                turns.append({"role": "assistant", "text": text, "ts": d.get("timestamp", "")})
    # retries after API errors re-append the same user message
    return project, session_id, _dedupe(turns)


def parse_codex_file(path: Path):
    project = None
    session_id = path.stem
    turns = []
    for d in _iter_jsonl(path):
        dtype = d.get("type")
        payload = d.get("payload") or {}
        ts = d.get("timestamp", "")
        if dtype == "session_meta":
            project = payload.get("cwd") or project
            session_id = payload.get("id") or session_id
            source = payload.get("source")
            subagent = source.get("subagent") if isinstance(source, dict) else None
            if isinstance(subagent, dict) and subagent.get("other") == "guardian":
                # Approval reviews replay entire transcripts repeatedly. They
                # are machine assessments, not additional human conversations.
                return project, session_id, []
        elif dtype == "event_msg":
            ptype = payload.get("type")
            if ptype == "user_message":
                text = _clean_user_text(payload.get("message") or "")
                if text:
                    turns.append({"role": "user", "text": text, "ts": ts, "_channel": "event"})
            elif ptype == "agent_message":
                text = (payload.get("message") or "").strip()
                if text:
                    turns.append({"role": "assistant", "text": text, "ts": ts, "_channel": "event"})
        elif dtype == "response_item" and payload.get("type") == "message":
            role = payload.get("role")
            if role == "assistant" and payload.get("channel") == "analysis":
                continue
            items = payload.get("content") or []
            text = "\n".join(
                i.get("text", "")
                for i in items
                if i.get("type") in ("input_text", "output_text")
            )
            if role == "user":
                text = _clean_user_text(text)
            else:
                text = text.strip()
            if text and role in ("user", "assistant"):
                turns.append({"role": role, "text": text, "ts": ts, "_channel": "response"})
    return project, session_id, _merge_codex_turns(turns)


def _merge_codex_turns(turns):
    """Match mirrored channels one-for-one, preserving repeated real messages."""
    mirrored = Counter((t['role'], t['text']) for t in turns if t['_channel'] == 'response')
    out = []
    for t in turns:
        key = (t['role'], t['text'])
        if t['_channel'] == 'event' and mirrored[key]:
            mirrored[key] -= 1
            continue
        out.append({k: v for k, v in t.items() if k != '_channel'})
    return out


def _parse_md(path: Path, root: Path):
    from datetime import datetime

    text = path.read_text(encoding="utf-8", errors="replace").strip()
    project = str(path.parent)
    session = str(path.relative_to(root))
    if not text:
        return project, session, []
    ts = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
    return project, session, [
        {"role": "note", "text": f"# {path.stem}\n{text}", "ts": ts}
    ]


def parse_obsidian_file(path: Path):
    return _parse_md(path, OBSIDIAN_ROOT)


def parse_docs_file(path: Path):
    """추가 마크다운 문서 폴더(LOCALRAG_DOCS_ROOT)를 제자리에서 색인."""
    return _parse_md(path, DOCS_ROOT)


def parse_claude_memory_file(path: Path):
    """Claude 자동 메모리(~/.claude/projects/*/memory/*.md) — 프로젝트별 증류된 결론."""
    _, session, turns = _parse_md(path, CLAUDE_ROOT)
    # .../projects/<인코딩된-프로젝트명>/memory/x.md → 인코딩명을 project로 사용
    return path.parent.parent.name, session, turns


def _dedupe(turns):
    out = []
    for t in turns:
        key = (t["role"], t["text"])
        # event_msg/response_item twins and immediate retries are adjacent.
        # A repeated question later in the conversation is a new exchange.
        if out and key == (out[-1]["role"], out[-1]["text"]):
            continue
        out.append(t)
    return out
