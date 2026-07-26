from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

import requests

FACTCHECK_KEY = os.getenv("GOOGLE_FACTCHECK_KEY", "")
ENDPOINT = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
CACHE_DIR = Path(".cache/factcheck")

Verdict = Literal["SUPPORTED", "REFUTED", "UNCLEAR", "NO_MATCH"]


# ---------------------------------------------------------------------------
# Rating normalization -- the one real piece of logic in phase 2
# ---------------------------------------------------------------------------

# Ordered most-specific first. Each pattern maps a free-text rating to a verdict.
# REFUTED patterns are broad because most fact-checks debunk; SUPPORTED patterns
# are kept tight to avoid false confidence. Everything unmatched -> UNCLEAR.
_REFUTED = re.compile(
    r"\b(false|pants[\s-]?on[\s-]?fire|fake|hoax|debunk|no evidence|not true|"
    r"incorrect|misleading|miscaption|mislead|distort|unfounded|baseless|"
    r"unsupported|fabricat|doctored|out of context|misrepresent|flawed|"
    r"do(es)? not|did not|never|inaccurate|wrong|myth|conspiracy)\b",
    re.I,
)
_SUPPORTED = re.compile(
    r"\b(true|correct|accurate|confirmed|verified|legit|substantiat|"
    r"mostly true|supported by evidence)\b",
    re.I,
)
_MIXED = re.compile(
    r"\b(half[\s-]?true|partly|partially|mixture|mixed|misleading|"
    r"exaggerat|needs context|missing context|lacks context|oversimplif|"
    r"barely[\s-]?true|mostly false)\b",
    re.I,
)


def classify_rating(rating: str) -> tuple[Verdict, str]:
    """Map free-text textualRating -> (verdict, reason). Conservative by design."""
    r = rating.strip()
    if not r:
        return "UNCLEAR", "empty rating"

    # Mixed checked before the others: "mostly false" contains "false" but is
    # genuinely a mixed verdict, so it must win.
    if _MIXED.search(r):
        return "UNCLEAR", "mixed / needs-context rating"
    if _REFUTED.search(r):
        return "REFUTED", "rating indicates the claim is false"
    if _SUPPORTED.search(r):
        return "SUPPORTED", "rating indicates the claim is true"
    return "UNCLEAR", "rating text not recognized"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Review:
    claim_text: str
    claimant: str
    claim_date: str
    publisher: str
    rating_raw: str
    verdict: Verdict
    reason: str
    url: str
    title: str
    review_date: str


@dataclass
class Result:
    query: str
    verdict: Verdict
    confidence: float
    reviews: list[Review]

    def render(self) -> str:
        icon = {"SUPPORTED": "[+]", "REFUTED": "[-]",
                "UNCLEAR": "[?]", "NO_MATCH": "[ ]"}[self.verdict]
        lines = [f"{icon} {self.verdict}   (confidence {self.confidence:.2f})",
                 f"    query: {self.query}"]
        if not self.reviews:
            lines.append("    no fact-checks found for this claim")
        for rv in self.reviews[:5]:
            lines.append(f"    - {rv.publisher}: \"{rv.rating_raw}\"  -> {rv.verdict}")
            lines.append(f"      reviewed claim: {_shorten(rv.claim_text, 100)}")
            lines.append(f"      {rv.url}")
        return "\n".join(lines)


def _shorten(s: str, n: int) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "\u2026"


# ---------------------------------------------------------------------------
# Retrieval with on-disk cache
# ---------------------------------------------------------------------------

def _cache_path(query: str, lang: str, size: int) -> Path:
    key = hashlib.sha256(f"{query}|{lang}|{size}".encode()).hexdigest()[:16]
    return CACHE_DIR / f"{key}.json"


def _raw_search(query: str, lang: str, size: int, use_cache: bool) -> dict:
    path = _cache_path(query, lang, size)
    if use_cache and path.exists():
        return json.loads(path.read_text())

    if not FACTCHECK_KEY:
        raise RuntimeError("GOOGLE_FACTCHECK_KEY is not set")

    resp = requests.get(ENDPOINT, params={
        "query": query, "key": FACTCHECK_KEY,
        "languageCode": lang, "pageSize": size,
    }, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def search(query: str, lang: str = "en", size: int = 10,
           use_cache: bool = True) -> Result:
    data = _raw_search(query, lang, size, use_cache)
    reviews: list[Review] = []

    for claim in data.get("claims", []):
        ctext = claim.get("text", "")
        claimant = claim.get("claimant", "")
        cdate = claim.get("claimDate", "")
        for rv in claim.get("claimReview", []):
            raw = rv.get("textualRating", "")
            verdict, reason = classify_rating(raw)
            reviews.append(Review(
                claim_text=ctext,
                claimant=claimant,
                claim_date=cdate,
                publisher=rv.get("publisher", {}).get("name", ""),
                rating_raw=raw,
                verdict=verdict,
                reason=reason,
                url=rv.get("url", ""),
                title=rv.get("title", ""),
                review_date=rv.get("reviewDate", ""),
            ))

    return _aggregate(query, reviews)


def _aggregate(query: str, reviews: list[Review]) -> Result:
    """Majority vote across reviews. Confidence = winning share of decided votes.

    UNCLEAR reviews count toward coverage but not toward the decision. If no
    review lands SUPPORTED or REFUTED, the whole result is UNCLEAR (we found
    fact-checks but none gave a clean verdict) or NO_MATCH (found nothing).
    """
    if not reviews:
        return Result(query, "NO_MATCH", 0.0, reviews)

    refuted = sum(r.verdict == "REFUTED" for r in reviews)
    supported = sum(r.verdict == "SUPPORTED" for r in reviews)
    decided = refuted + supported

    if decided == 0:
        return Result(query, "UNCLEAR", 0.0, reviews)

    if refuted >= supported:
        return Result(query, "REFUTED", refuted / decided, reviews)
    return Result(query, "SUPPORTED", supported / decided, reviews)


# ---------------------------------------------------------------------------
# Demo: the Phase 1 adversarial claims, now sent to real fact-checkers
# ---------------------------------------------------------------------------

DEMO_CLAIMS = [
    "drinking bleach cures infections",
    "vaccines cause autism",
    "the Great Wall of China is visible from the Moon",
    "the Earth orbits the Sun",
    "5G towers spread coronavirus",
    "the 2020 US election was stolen through voter fraud",
]


def run_demo() -> None:
    for claim in DEMO_CLAIMS:
        print("=" * 70)
        print(search(claim).render())
        print()
        time.sleep(0.3)   # be polite to the API


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="*", help="claim to check")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--json", action="store_true", help="emit raw JSON")
    args = ap.parse_args()

    if args.demo:
        run_demo()
        return

    if not args.query:
        print(__doc__)
        sys.exit(1)

    result = search(" ".join(args.query), use_cache=not args.no_cache)
    if args.json:
        print(json.dumps({**asdict(result)}, indent=2, default=str))
    else:
        print(result.render())


if __name__ == "__main__":
    main()