"""
Phase 3: relevance-gated stance classification.

Today's finding, learned from real output: an NLI model does NOT answer "is this
evidence about this claim?" It answers "if the premise is true, is the hypothesis
true?" Off-topic evidence about a claim the model has a pretraining opinion on
gets pulled toward entailment/contradiction by that opinion, not toward neutral.
(Five off-topic climate reviews all scored contradiction ~0.99 against "climate
change is a hoax" because the model knows climate change is real.)

Therefore relevance and stance are SEPARATE questions and need SEPARATE models:

    1. RelevanceGate (cross-encoder)  -- is this evidence topically about the
       claim at all? Drop everything below a topicality threshold.
    2. StanceModel (NLI)              -- for evidence that passed the gate, does
       it support (entailment) or refute (contradiction) the claim?

Only evidence that is both relevant AND has a clear stance gets to vote.

Models:
    relevance: cross-encoder/ms-marco-MiniLM-L-6-v2   (query-doc relevance score)
    stance:    MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
               label map confirmed on-device: {0: entailment, 1: neutral, 2: contradiction}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

StanceLabel = Literal["entailment", "neutral", "contradiction"]
EvidenceVerdict = Literal["SUPPORTS", "REFUTES", "NEUTRAL", "IRRELEVANT"]

RELEVANCE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
STANCE_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

# Cross-encoder ms-marco scores are logits, not 0-1. Higher = more relevant.
#
# Calibration finding (calibrate_relevance.py, 18 labeled pairs): NO single
# signal separates on-topic from off-topic cleanly. The cross-encoder peaked at
# 0.83 labelling accuracy; NLI-neutral was worse at 0.67; they err on disjoint
# subsets. Rather than fit a threshold to 18 hand-picked examples (the Phase 1
# overfitting trap), the interim gate is deliberately PERMISSIVE -- it only
# rejects evidence the cross-encoder scores clearly off-topic -- and the verdict
# leans on stance confidence plus corroboration (REQUIRE_MIN_VOTES) instead of
# trusting the gate to be surgical. The principled gate is settled in Phase 4
# against FEVER, where claims are diverse and the model has no priors.
#
# -6.0 sits below every genuinely-relevant pair in the labeled set except two
# reframed rebuttals, and above the clearly off-topic cluster (-8 to -11).
RELEVANCE_THRESHOLD = -6.0

# NLI must be reasonably confident before a piece of evidence is allowed to vote.
STANCE_MIN_CONFIDENCE = 0.60

# Because the gate is permissive, a single piece is not allowed to decide a
# verdict on its own -- require at least this many agreeing votes, or fall back
# to NOT_ENOUGH_INFO. Set to 1 to disable corroboration.
REQUIRE_MIN_VOTES = 2


@dataclass
class StanceResult:
    evidence: str
    claim: str
    relevance_score: float
    relevant: bool
    stance: StanceLabel
    stance_probs: dict[str, float]
    verdict: EvidenceVerdict

    def render(self) -> str:
        return (f"[{self.verdict:10}] rel={self.relevance_score:+.2f} "
                f"stance={self.stance}({self.stance_probs[self.stance]:.2f})\n"
                f"             claim:    {_short(self.claim, 70)}\n"
                f"             evidence: {_short(self.evidence, 70)}")


def _short(s: str, n: int) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "\u2026"


# ---------------------------------------------------------------------------
# Verdict logic -- pure function, unit-tested below without any model
# ---------------------------------------------------------------------------

def decide_verdict(relevant: bool, stance: StanceLabel,
                   stance_conf: float,
                   min_conf: float = STANCE_MIN_CONFIDENCE) -> EvidenceVerdict:
    """Combine a relevance decision and an NLI stance into one evidence verdict.

    The gate is the whole point of the phase: irrelevant evidence NEVER votes,
    regardless of how confident its stance score is. This is what would have
    stopped the five off-topic climate reviews from being counted.
    """
    if not relevant:
        return "IRRELEVANT"
    if stance == "neutral" or stance_conf < min_conf:
        return "NEUTRAL"
    if stance == "entailment":
        return "SUPPORTS"
    return "REFUTES"


def aggregate(results: Sequence[StanceResult],
              min_votes: int = REQUIRE_MIN_VOTES) -> tuple[str, float, int]:
    """Aggregate evidence verdicts into a claim-level verdict.

    Only SUPPORTS / REFUTES count. Returns (verdict, confidence, n_voting).
    NEUTRAL and IRRELEVANT evidence contributes to coverage but not the vote,
    so a claim backed only by off-topic hits comes back NOT_ENOUGH_INFO -- the
    Phase 2 failure mode, now fixed.

    Because the relevance gate is permissive (see RELEVANCE_THRESHOLD), the
    winning side must reach min_votes agreeing pieces; otherwise NOT_ENOUGH_INFO.
    This trades recall for precision while the gate stays uncalibrated.
    """
    supports = sum(r.verdict == "SUPPORTS" for r in results)
    refutes = sum(r.verdict == "REFUTES" for r in results)
    voting = supports + refutes

    if voting == 0:
        return "NOT_ENOUGH_INFO", 0.0, 0
    if supports == refutes:
        return "CONFLICTING", 0.5, voting

    if refutes > supports:
        winner, verdict = refutes, "REFUTED"
    else:
        winner, verdict = supports, "SUPPORTED"

    if winner < min_votes:
        return "NOT_ENOUGH_INFO", 0.0, voting
    return verdict, winner / voting, voting


# ---------------------------------------------------------------------------
# Model wrapper -- lazy, so importing this file costs nothing
# ---------------------------------------------------------------------------

class Verifier:
    def __init__(self,
                 relevance_threshold: float = RELEVANCE_THRESHOLD,
                 stance_min_confidence: float = STANCE_MIN_CONFIDENCE) -> None:
        from sentence_transformers import CrossEncoder
        from transformers import (AutoModelForSequenceClassification,
                                   AutoTokenizer)
        import torch

        self._torch = torch
        self.relevance_threshold = relevance_threshold
        self.stance_min_confidence = stance_min_confidence

        self.reranker = CrossEncoder(RELEVANCE_MODEL)
        self.tok = AutoTokenizer.from_pretrained(STANCE_MODEL)
        self.nli = AutoModelForSequenceClassification.from_pretrained(STANCE_MODEL)
        self.nli.eval()
        self.id2label = {i: l.lower() for i, l in self.nli.config.id2label.items()}

    def _relevance(self, claim: str, evidence: str) -> float:
        return float(self.reranker.predict([(claim, evidence)])[0])

    def _stance(self, evidence: str, claim: str) -> tuple[StanceLabel, dict[str, float]]:
        # premise = evidence, hypothesis = claim
        inputs = self.tok(evidence, claim, return_tensors="pt",
                          truncation=True, max_length=512)
        with self._torch.no_grad():
            probs = self._torch.softmax(self.nli(**inputs).logits[0], dim=-1)
        dist = {self.id2label[i]: float(p) for i, p in enumerate(probs)}
        top = max(dist, key=dist.get)
        return top, dist  # type: ignore[return-value]

    def check(self, claim: str, evidence: str) -> StanceResult:
        rel = self._relevance(claim, evidence)
        relevant = rel >= self.relevance_threshold
        stance, dist = self._stance(evidence, claim)
        verdict = decide_verdict(relevant, stance, dist[stance],
                                 self.stance_min_confidence)
        return StanceResult(evidence, claim, rel, relevant, stance, dist, verdict)

    def check_many(self, claim: str, evidences: Sequence[str]) -> list[StanceResult]:
        return [self.check(claim, e) for e in evidences]


# ---------------------------------------------------------------------------
# The Phase 2 failure cases, as a runnable demo
# ---------------------------------------------------------------------------

EARTH_SUN = (
    "The Earth orbits the Sun.",
    ["Photos appearing to show clouds behind the sun do not prove the sun "
     "orbits the earth; the earth orbits the sun, millions of miles away."],
)

CLIMATE_HOAX = (
    "Climate change is a hoax.",
    ["The wind doesn't work, it's the most expensive energy.",
     "Coal is okay, they have methods now where coal becomes clean coal.",
     "Years ago they used to call it global cooling.",
     "Scientists predicted we have 12 years to live because of climate change.",
     "They don't call it global warming anymore, they call it climate change.",
     "Multiple independent temperature records confirm the planet is warming "
     "due to human greenhouse gas emissions."],
)


def run_demo() -> None:
    v = Verifier()
    for title, (claim, evidences) in [("EARTH / SUN", EARTH_SUN),
                                      ("CLIMATE HOAX", CLIMATE_HOAX)]:
        print("=" * 72)
        print(title)
        print("=" * 72)
        results = v.check_many(claim, evidences)
        for r in results:
            print(r.render())
            print()
        verdict, conf, n = aggregate(results)
        print(f">>> claim verdict: {verdict}  (confidence {conf:.2f}, "
              f"{n} of {len(results)} pieces voted)\n")


if __name__ == "__main__":
    run_demo()