"""
Phase 3 calibration: find the relevance threshold from labeled data, not a guess.

Method
------
1. A hand-labeled set of (claim, evidence, is_relevant) triples. RELEVANCE is
   labeled independently of stance: "is this evidence ABOUT this claim?" A
   refuting-but-on-topic sentence is relevant; an off-topic sentence is not,
   no matter what it says.
2. Score every pair with the cross-encoder.
3. Look at how relevant vs irrelevant scores separate:
     - best threshold (maximizes labelling accuracy)
     - the overlap zone (max irrelevant score vs min relevant score)
   If min(relevant) > max(irrelevant), a clean threshold exists -> use the
   midpoint. If they overlap, no threshold separates them and the cross-encoder
   is the wrong tool -> that is a reportable result, not a bug to hide.

Run:
    python calibrate_relevance.py

Add your own pairs to LABELED. ~8-10 relevant and ~8-10 irrelevant is enough to
see whether separation exists; more tightens the threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

RELEVANCE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# (claim, evidence, is_relevant)
# is_relevant = is this evidence ABOUT this claim, regardless of support/refute.
LABELED: list[tuple[str, str, bool]] = [
    # --- climate hoax: one relevant rebuttal, several off-topic distractors ---
    ("Climate change is a hoax.",
     "Multiple independent temperature records confirm the planet is warming due to human greenhouse gas emissions.",
     True),
    ("Climate change is a hoax.",
     "97% of actively publishing climate scientists agree that warming trends are human-caused.",
     True),
    ("Climate change is a hoax.",
     "The claim that global warming is a fabrication has been contradicted by NASA and NOAA datasets.",
     True),
    ("Climate change is a hoax.",
     "The wind doesn't work, it's the most expensive energy.",
     False),
    ("Climate change is a hoax.",
     "Coal is okay, they have methods now where coal becomes clean coal.",
     False),
    ("Climate change is a hoax.",
     "Scientists predicted we have 12 years to live because of climate change.",
     False),

    # --- earth/sun ---
    ("The Earth orbits the Sun.",
     "The earth orbits the sun once every 365.25 days at a distance of about 93 million miles.",
     True),
    ("The Earth orbits the Sun.",
     "Photos appearing to show clouds behind the sun do not prove the sun orbits the earth.",
     True),
    ("The Earth orbits the Sun.",
     "The moon completes one orbit of the earth roughly every 27 days.",
     False),
    ("The Earth orbits the Sun.",
     "Jupiter is the largest planet in the solar system.",
     False),

    # --- vaccines / autism ---
    ("Vaccines cause autism.",
     "Large cohort studies covering over a million children found no association between the MMR vaccine and autism.",
     True),
    ("Vaccines cause autism.",
     "The 1998 study alleging a vaccine-autism link was retracted and its author lost his medical license.",
     True),
    ("Vaccines cause autism.",
     "Vaccination rates in the United States dropped during the 2020 pandemic.",
     False),
    ("Vaccines cause autism.",
     "The flu vaccine is reformulated each year to match circulating strains.",
     False),

    # --- election ---
    ("The 2020 US election was stolen through voter fraud.",
     "Over 60 courts, including judges appointed by Trump, rejected claims of widespread 2020 election fraud.",
     True),
    ("The 2020 US election was stolen through voter fraud.",
     "Audits and recounts in contested states confirmed the certified 2020 results.",
     True),
    ("The 2020 US election was stolen through voter fraud.",
     "Voter turnout in 2020 was the highest as a share of eligible voters in over a century.",
     False),
    ("The 2020 US election was stolen through voter fraud.",
     "The US president is inaugurated on January 20th following the election.",
     False),
]


@dataclass
class Scored:
    claim: str
    evidence: str
    relevant: bool
    score: float


def score_all(pairs: list[tuple[str, str, bool]]) -> list[Scored]:
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(RELEVANCE_MODEL)
    inputs = [(c, e) for c, e, _ in pairs]
    scores = model.predict(inputs)
    return [Scored(c, e, r, float(s))
            for (c, e, r), s in zip(pairs, scores)]


def analyze(scored: list[Scored]) -> None:
    rel = sorted((s.score for s in scored if s.relevant))
    irr = sorted((s.score for s in scored if not s.relevant))

    print(f"relevant pairs:   {len(rel)}   "
          f"range [{min(rel):+.2f}, {max(rel):+.2f}]")
    print(f"irrelevant pairs: {len(irr)}   "
          f"range [{min(irr):+.2f}, {max(irr):+.2f}]")
    print()

    # Best threshold by sweeping every midpoint between adjacent sorted scores.
    all_scores = sorted(s.score for s in scored)
    candidates = [(a + b) / 2 for a, b in zip(all_scores, all_scores[1:])]
    candidates += [all_scores[0] - 1, all_scores[-1] + 1]

    best_thr, best_acc = 0.0, -1.0
    for thr in candidates:
        acc = sum((s.score >= thr) == s.relevant for s in scored) / len(scored)
        if acc > best_acc:
            best_acc, best_thr = acc, thr

    # Separation check.
    clean = min(rel) > max(irr)
    print(f"max irrelevant score: {max(irr):+.2f}")
    print(f"min relevant score:   {min(rel):+.2f}")
    if clean:
        gap_lo, gap_hi = max(irr), min(rel)
        print(f"\nCLEAN SEPARATION. Any threshold in ({gap_lo:+.2f}, "
              f"{gap_hi:+.2f}) separates perfectly.")
        print(f"Recommended threshold: {(gap_lo + gap_hi) / 2:+.2f}")
    else:
        overlap = [s for s in scored
                   if (s.relevant and s.score <= max(irr))
                   or (not s.relevant and s.score >= min(rel))]
        print(f"\nOVERLAP. No threshold separates cleanly.")
        print(f"Best achievable labelling accuracy: {best_acc:.2f} "
              f"at threshold {best_thr:+.2f}")
        print(f"{len(overlap)} pairs fall in the overlap zone:")
        for s in sorted(overlap, key=lambda x: x.score):
            tag = "REL" if s.relevant else "irr"
            print(f"   {tag} {s.score:+.2f}  {s.evidence[:64]}")
        print("\nInterpretation: the cross-encoder cannot cleanly tell on-topic")
        print("from off-topic for these claims. A better relevance signal is")
        print("needed (e.g. NLI neutral-score, or an NLI-based entailment gate).")


def score_all_nli(pairs: list[tuple[str, str, bool]]) -> list[Scored]:
    """Relevance signal = 1 - neutral_probability from the NLI model.

    Hypothesis: on-topic evidence (support OR refute) has LOW neutral prob;
    off-topic evidence has HIGH neutral prob. So (1 - neutral) is a relevance
    score on the same 0-1 scale for every pair, from the model already in the
    pipeline. Higher = more relevant.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    model.eval()
    neutral_idx = next(i for i, l in model.config.id2label.items()
                       if l.lower() == "neutral")

    out = []
    for claim, evidence, relevant in pairs:
        inputs = tok(evidence, claim, return_tensors="pt",
                     truncation=True, max_length=512)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits[0], dim=-1)
        relevance = 1.0 - float(probs[neutral_idx])
        out.append(Scored(claim, evidence, relevant, relevance))
    return out


def main() -> None:
    print("#" * 72)
    print("# SIGNAL 1: cross-encoder (ms-marco passage relevance)")
    print("#" * 72)
    scored = score_all(LABELED)
    for s in sorted(scored, key=lambda x: -x.score):
        tag = "REL" if s.relevant else "irr"
        print(f"  {tag}  {s.score:+7.2f}  [{s.claim[:28]:28}] {s.evidence[:46]}")
    print()
    analyze(scored)

    print()
    print("#" * 72)
    print("# SIGNAL 2: NLI (1 - neutral probability)")
    print("#" * 72)
    scored_nli = score_all_nli(LABELED)
    for s in sorted(scored_nli, key=lambda x: -x.score):
        tag = "REL" if s.relevant else "irr"
        print(f"  {tag}  {s.score:6.3f}  [{s.claim[:28]:28}] {s.evidence[:46]}")
    print()
    analyze(scored_nli)


if __name__ == "__main__":
    main()