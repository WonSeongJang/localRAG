"""Embeddings via local Ollama (bge-m3)."""

import requests

OLLAMA_URL = "http://localhost:11434"
MODEL = "bge-m3"
BATCH = 32


def embed_texts(texts):
    """Return one 1024-dim vector per input text."""
    out = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        r = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": MODEL, "input": batch},
            timeout=300,
        )
        r.raise_for_status()
        out.extend(r.json()["embeddings"])
    return out


def check_ollama():
    """Raise a friendly error if Ollama/bge-m3 is not reachable."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        r.raise_for_status()
    except requests.RequestException:
        raise SystemExit(
            "Ollama is not running. Start it with: ollama serve (or open the Ollama app)"
        )
    models = [m["name"] for m in r.json().get("models", [])]
    if not any(m.startswith(MODEL) for m in models):
        raise SystemExit(f"Model '{MODEL}' not found. Pull it with: ollama pull {MODEL}")
