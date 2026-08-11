"""
Phase 5: pipeline orchestration for the UI.

One entry point -- debunk(claim) -- that runs the whole system and returns a
structured result the interface can render: the verdict, confidence, and every
piece of evidence with its source URL, relevance, and stance.

This is the same pipeline the FEVER harness evaluated (retrieve -> gate ->
stance -> aggregate), packaged for interactive use rather than batch scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import factcheck
from stance import Verifier, aggregate, StanceResult
from web_search import search_web


@dataclass
class EvidenceView:
    """One piece of evidence, with everything the UI needs to show it."""
    text: str
    url: str
    title: str
    source: str          # "fact-check" | "web"
    relevance: float
    relevant: bool
    stance: str
    stance_conf: float
    verdict: str         # SUPPORTS | REFUTES | NEUTRAL | IRRELEVANT


@dataclass
class DebunkResult:
    claim: str
    verdict: str         # SUPPORTED | REFUTED | NOT_ENOUGH_INFO | CONFLICTING
    confidence: float
    n_voted: int
    evidence: list[EvidenceView]

    @property
    def headline(self) -> str:
        return {
            "SUPPORTED": "Likely true",
            "REFUTED": "Likely false",
            "NOT_ENOUGH_INFO": "Not enough evidence",
            "CONFLICTING": "Evidence conflicts",
        }.get(self.verdict, self.verdict)


def _gather(claim: str, max_web: int) -> list[EvidenceView]:
    """Collect evidence from both sources, tagged with provenance."""
    raw: list[tuple[str, str, str, str]] = []   # (text, url, title, source)

    try:
        fc = factcheck.search(claim)
        for rv in fc.reviews:
            if rv.claim_text:
                raw.append((
                    f"{rv.claim_text} (rated {rv.rating_raw} by {rv.publisher})",
                    rv.url, rv.title or rv.publisher, "fact-check"))
    except Exception:  # noqa: BLE001 -- fact-check is best-effort
        pass

    for e in search_web(claim, max_results=max_web):
        if len(e.text.strip()) >= 25:
            raw.append((e.text, e.url, e.title, "web"))

    # Dedup by leading text.
    seen, deduped = set(), []
    for text, url, title, source in raw:
        k = text[:120]
        if k not in seen:
            seen.add(k)
            deduped.append((text, url, title, source))

    return [EvidenceView(text=t, url=u, title=ti, source=s,
                         relevance=0.0, relevant=False,
                         stance="", stance_conf=0.0, verdict="")
            for t, u, ti, s in deduped]


def debunk(claim: str, verifier: Verifier, max_web: int = 5,
           min_votes: int = 2) -> DebunkResult:
    """Run the full pipeline on a single claim."""
    views = _gather(claim, max_web)

    if not views:
        return DebunkResult(claim, "NOT_ENOUGH_INFO", 0.0, 0, [])

    stance_results: list[StanceResult] = verifier.check_many(
        claim, [v.text for v in views])

    # Fold stance output back into the evidence views.
    for v, sr in zip(views, stance_results):
        v.relevance = sr.relevance_score
        v.relevant = sr.relevant
        v.stance = sr.stance
        v.stance_conf = sr.stance_probs[sr.stance]
        v.verdict = sr.verdict

    verdict, conf, n_voted = aggregate(stance_results, min_votes=min_votes)

    # Show the most decision-relevant evidence first: voting pieces, then by
    # relevance.
    order = {"SUPPORTS": 0, "REFUTES": 0, "NEUTRAL": 1, "IRRELEVANT": 2}
    views.sort(key=lambda v: (order.get(v.verdict, 3), -v.relevance))

    return DebunkResult(claim, verdict, conf, n_voted, views)