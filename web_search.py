"""
Phase 4: real web evidence retrieval (Tavily), cached to disk.

Behind a provider-swappable interface: search_web(claim) -> list[WebEvidence].
Every call is cached to .cache/web/ keyed by query, so re-runs during harness
development cost zero credits. The free tier is ~1000 credits/month and a retry
loop can burn it fast -- caching is not optional.

Setup:
    export TAVILY_KEY="tvly-..."

Test:
    python web_search.py "Telemundo is an English-language television network"
    python web_search.py --demo      # one claim per FEVER label type
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import requests

TAVILY_KEY = os.getenv("TAVILY_KEY", "")
TAVILY_ENDPOINT = "https://api.tavily.com/search"
CACHE_DIR = Path(".cache/web")


@dataclass
class WebEvidence:
    text: str
    url: str
    title: str
    score: float          # Tavily's own relevance score, 0-1
    source: str = "web"


def _cache_path(query: str, max_results: int) -> Path:
    key = hashlib.sha256(f"{query}|{max_results}".encode()).hexdigest()[:16]
    return CACHE_DIR / f"{key}.json"


def _raw_search(query: str, max_results: int, use_cache: bool) -> dict:
    path = _cache_path(query, max_results)
    if use_cache and path.exists():
        return json.loads(path.read_text())

    if not TAVILY_KEY:
        raise RuntimeError("TAVILY_KEY is not set")

    resp = requests.post(TAVILY_ENDPOINT, json={
        "api_key": TAVILY_KEY,
        "query": query,
        "search_depth": "basic",      # "advanced" costs more credits
        "max_results": max_results,
        "include_answer": False,       # we want raw evidence, not a synthesized answer
    }, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def search_web(claim: str, max_results: int = 5,
               use_cache: bool = True) -> list[WebEvidence]:
    """Retrieve web evidence for a claim. Provider-swappable: to change search
    backends, reimplement this function's body and keep the signature."""
    data = _raw_search(claim, max_results, use_cache)
    out = []
    for item in data.get("results", []):
        content = item.get("content", "").strip()
        if not content:
            continue
        out.append(WebEvidence(
            text=content,
            url=item.get("url", ""),
            title=item.get("title", ""),
            score=float(item.get("score", 0.0)),
        ))
    return out


# --------------------------------------------------------------------------
# Demo: one real FEVER claim per label type, to inspect what web search returns
# --------------------------------------------------------------------------

DEMO = [
    ("SUPPORTS",        "Fox 2000 Pictures released the film Soul Food."),
    ("REFUTES",         "Telemundo is an English-language television network."),
    ("NOT ENOUGH INFO", "Colin Kaepernick became a starting quarterback during "
                        "the 49ers 63rd season in the National Football League."),
]


def run_demo() -> None:
    for label, claim in DEMO:
        print("=" * 72)
        print(f"FEVER label: {label}")
        print(f"claim: {claim}")
        print("=" * 72)
        ev = search_web(claim)
        if not ev:
            print("  (no results)")
        for e in ev:
            print(f"  [{e.score:.2f}] {e.title[:60]}")
            print(f"        {e.text[:160]}")
            print(f"        {e.url}")
            print()
        time.sleep(0.3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="*")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    if args.demo:
        run_demo()
        return
    if not args.query:
        print(__doc__)
        sys.exit(1)

    for e in search_web(" ".join(args.query), use_cache=not args.no_cache):
        print(f"[{e.score:.2f}] {e.title}")
        print(f"   {e.text[:200]}")
        print(f"   {e.url}\n")


if __name__ == "__main__":
    main()