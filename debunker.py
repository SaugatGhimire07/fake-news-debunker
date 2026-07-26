"""
Evidence-based fake news debunker.

Pipeline:  input -> claim extraction -> evidence retrieval -> rerank
           -> stance (NLI) -> aggregated verdict with citations

Environment variables expected:
    GOOGLE_FACTCHECK_KEY   https://console.cloud.google.com  (enable Fact Check Tools API)
    TAVILY_KEY             https://tavily.com                (1000 free credits/month)

Install:  pip install -r requirements.txt
Run:      python debunker.py "The Eiffel Tower was sold for scrap in 1925."
"""

from __future__ import annotations

import os
import re
import textwrap
from dataclasses import dataclass, field
from typing import Iterable, Literal

import requests

FACTCHECK_KEY = os.getenv("GOOGLE_FACTCHECK_KEY", "")
TAVILY_KEY = os.getenv("TAVILY_KEY", "")

Label = Literal["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class Evidence:
    text: str
    url: str
    title: str = ""
    source_type: str = "web"          # "factcheck" | "web"
    relevance: float = 0.0            # from the reranker
    stance: str = "neutral"           # entailment | contradiction | neutral
    stance_score: float = 0.0


@dataclass
class Verdict:
    claim: str
    label: Label = "NOT_ENOUGH_INFO"
    confidence: float = 0.0
    evidence: list[Evidence] = field(default_factory=list)

    def render(self) -> str:
        icon = {"SUPPORTED": "[+]", "REFUTED": "[-]", "NOT_ENOUGH_INFO": "[?]"}[self.label]
        lines = [f"{icon} {self.label}  (confidence {self.confidence:.2f})",
                 f"    Claim: {textwrap.shorten(self.claim, 140)}"]
        for e in self.evidence[:4]:
            lines.append(f"    - [{e.stance}] {textwrap.shorten(e.text, 160)}")
            lines.append(f"      {e.title or e.source_type} :: {e.url}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# 1. Article extraction
# --------------------------------------------------------------------------

def load_input(source: str) -> str:
    """Accepts raw text or a URL. Returns clean article body text."""
    if not source.startswith("http"):
        return source
    # For URL input, fetch the page and strip boilerplate before extracting claims.
    import trafilatura
    downloaded = trafilatura.fetch_url(source)
    if downloaded is None:
        raise RuntimeError(f"Could not fetch {source}")
    return trafilatura.extract(downloaded, include_comments=False) or ""


# --------------------------------------------------------------------------
# 2. Claim extraction (check-worthiness)
# --------------------------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")

# Cheap heuristic proxy for ClaimBuster-style check-worthiness. Replace with a
# trained classifier or an LLM call once the rest of the pipeline works.
_FACTUAL_CUES = re.compile(
    r"\b(\d[\d,.]*%?|percent|million|billion|study|report|according to|announced|"
    r"confirmed|banned|approved|died|killed|increase[sd]?|decrease[sd]?)\b",
    re.I,
)
_HEDGE = re.compile(r"\b(i think|maybe|perhaps|in my opinion|should|ought to)\b", re.I)


def extract_claims(text: str, max_claims: int = 5) -> list[str]:
    # Split the article into sentences and keep the ones that look check-worthy.
    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if len(s.split()) >= 6]
    scored = []
    for s in sentences:
        if _HEDGE.search(s):
            continue
        score = len(_FACTUAL_CUES.findall(s))
        if score:
            scored.append((score, s))
    scored.sort(key=lambda x: -x[0])
    return [s for _, s in scored[:max_claims]] or sentences[:1]


# --------------------------------------------------------------------------
# 3. Evidence retrieval
# --------------------------------------------------------------------------

def search_factcheck_api(claim: str, limit: int = 5) -> list[Evidence]:
    """Claims already reviewed by professional fact-checkers. Cheap, high precision."""
    if not FACTCHECK_KEY:
        return []
    r = requests.get(
        "https://factchecktools.googleapis.com/v1alpha1/claims:search",
        params={"query": claim, "key": FACTCHECK_KEY, "pageSize": limit, "languageCode": "en"},
        timeout=20,
    )
    r.raise_for_status()
    out: list[Evidence] = []
    for item in r.json().get("claims", []):
        for review in item.get("claimReview", []):
            publisher = review.get("publisher", {}).get("name", "")
            rating = review.get("textualRating", "")
            out.append(Evidence(
                text=f"{publisher} rated this claim: {rating}. Claim reviewed: {item.get('text', '')}",
                url=review.get("url", ""),
                title=review.get("title", publisher),
                source_type="factcheck",
            ))
    return out


def search_web(claim: str, limit: int = 6) -> list[Evidence]:
    """General web evidence. Swap this body to change search provider."""
    if not TAVILY_KEY:
        return []
    r = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": TAVILY_KEY,
            "query": claim,
            "search_depth": "advanced",
            "max_results": limit,
        },
        timeout=30,
    )
    r.raise_for_status()
    return [
        Evidence(text=item.get("content", ""), url=item.get("url", ""),
                 title=item.get("title", ""), source_type="web")
        for item in r.json().get("results", [])
        if item.get("content")
    ]


def retrieve(claim: str) -> list[Evidence]:
    return search_factcheck_api(claim) + search_web(claim)


# --------------------------------------------------------------------------
# 4 + 5. Rerank and stance classification
# --------------------------------------------------------------------------

class Verifier:
    """Lazily loads the cross-encoder reranker and the NLI stance model."""

    RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    NLI = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"   # swap to -large for accuracy

    def __init__(self) -> None:
        from sentence_transformers import CrossEncoder
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.reranker = CrossEncoder(self.RERANKER)
        self.tok = AutoTokenizer.from_pretrained(self.NLI)
        self.nli = AutoModelForSequenceClassification.from_pretrained(self.NLI)
        self.nli.eval()
        self.id2label = {i: l.lower() for i, l in self.nli.config.id2label.items()}

    def rerank(self, claim: str, evidence: list[Evidence], keep: int = 6) -> list[Evidence]:
        if not evidence:
            return []
        scores = self.reranker.predict([(claim, e.text) for e in evidence])
        for e, s in zip(evidence, scores):
            e.relevance = float(s)
        evidence.sort(key=lambda e: -e.relevance)
        return evidence[:keep]

    def stance(self, claim: str, evidence: Iterable[Evidence]) -> None:
        """NLI: premise = evidence, hypothesis = claim. Mutates in place."""
        import torch

        for e in evidence:
            inputs = self.tok(e.text, claim, return_tensors="pt",
                              truncation=True, max_length=512)
            with torch.no_grad():
                probs = torch.softmax(self.nli(**inputs).logits[0], dim=-1)
            idx = int(probs.argmax())
            e.stance = self.id2label[idx]
            e.stance_score = float(probs[idx])


# --------------------------------------------------------------------------
# 6. Verdict aggregation
# --------------------------------------------------------------------------

def aggregate(claim: str, evidence: list[Evidence], threshold: float = 0.55) -> Verdict:
    support = sum(e.stance_score * (2.0 if e.source_type == "factcheck" else 1.0)
                  for e in evidence if e.stance == "entailment")
    refute = sum(e.stance_score * (2.0 if e.source_type == "factcheck" else 1.0)
                 for e in evidence if e.stance == "contradiction")
    total = support + refute

    if total == 0:
        return Verdict(claim, "NOT_ENOUGH_INFO", 0.0, evidence)

    if support >= refute:
        label, conf = "SUPPORTED", support / total
    else:
        label, conf = "REFUTED", refute / total

    if conf < threshold:
        label = "NOT_ENOUGH_INFO"
    return Verdict(claim, label, conf, evidence)


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------

def debunk(source: str, max_claims: int = 3) -> list[Verdict]:
    text = load_input(source)
    claims = extract_claims(text, max_claims=max_claims)
    verifier = Verifier()

    verdicts = []
    for claim in claims:
        ev = verifier.rerank(claim, retrieve(claim))
        verifier.stance(claim, ev)
        verdicts.append(aggregate(claim, ev))
    return verdicts


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)

    for v in debunk(sys.argv[1]):
        print(v.render())
        print()