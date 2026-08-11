"""
Phase 4: evaluate the full pipeline against FEVER labelled_dev (Option C).

For each sampled FEVER claim:
    retrieve (fact-check API + web) -> clean to prose -> relevance gate + NLI
    stance -> aggregate -> map to FEVER's {SUPPORTS, REFUTES, NOT ENOUGH INFO}
and score against the gold label.

The honest-NEI accounting is the point of this harness. A claim that comes out
NOT_ENOUGH_INFO is split into:
    - NEI (empty)     : retrieval returned nothing -> abstained by default
    - NEI (reasoned)  : evidence was retrieved but gate/stance found it
                        insufficient -> abstained by reasoning
Only the second is a "real" NEI. Reporting them merged would let the pipeline
score well on NEI claims for the wrong reason (obscure claims -> empty retrieval
-> accidental NEI).

Run:
    export GOOGLE_FACTCHECK_KEY=...   TAVILY_KEY=...
    python fever_eval.py --per-label 10          # tiny debug run
    python fever_eval.py --per-label 100         # real run

Reuses: fever_load.py, factcheck.py, web_search.py, stance.py
"""

from __future__ import annotations

import argparse
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from fever_load import load_fever_dev
from web_search import search_web
import factcheck
from stance import Verifier, aggregate, StanceResult

FEVER_LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

# Map the pipeline's verdict vocabulary onto FEVER's three labels.
VERDICT_TO_FEVER = {
    "SUPPORTED": "SUPPORTS",
    "REFUTED": "REFUTES",
    "NOT_ENOUGH_INFO": "NOT ENOUGH INFO",
    "CONFLICTING": "NOT ENOUGH INFO",   # genuine disagreement = can't resolve
}


# --------------------------------------------------------------------------
# Evidence cleanup: NLI wants prose, not infobox pipes or table fragments
# --------------------------------------------------------------------------

def clean_evidence(text: str) -> str:
    # Drop wiki/markdown table rows ("| a | b |"), collapse whitespace.
    lines = [ln for ln in text.splitlines() if ln.count("|") < 2]
    text = " ".join(lines)
    text = re.sub(r"\s+", " ", text).strip()
    # FEVER wiki markup artifacts, in case any leak through a source.
    text = text.replace("-LRB-", "(").replace("-RRB-", ")")
    return text


def gather_evidence(claim: str, max_web: int = 5) -> list[str]:
    """Collect and clean candidate evidence sentences from all sources."""
    pieces: list[str] = []

    # Fact-check API: use the reviewed-claim text as evidence when present.
    try:
        fc = factcheck.search(claim)
        for rv in fc.reviews:
            if rv.claim_text:
                pieces.append(clean_evidence(
                    f"{rv.claim_text} (rated {rv.rating_raw} by {rv.publisher})"))
    except Exception:  # noqa: BLE001 -- fact-check is best-effort here
        pass

    # Web: the content snippets.
    for e in search_web(claim, max_results=max_web):
        c = clean_evidence(e.text)
        if len(c) >= 25:            # drop fragments too short to reason over
            pieces.append(c)

    # Dedup while preserving order.
    seen, out = set(), []
    for p in pieces:
        k = p[:120]
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


# --------------------------------------------------------------------------
# Per-claim evaluation
# --------------------------------------------------------------------------

@dataclass
class ClaimResult:
    claim: str
    gold: str
    predicted: str
    n_evidence: int
    n_voted: int
    nei_kind: str = ""      # "" | "empty" | "reasoned"  (only for NEI predictions)
    results: list = field(default_factory=list)


def evaluate_claim(verifier: Verifier, claim: str, gold: str,
                   min_votes: int = 2) -> ClaimResult:
    evidence = gather_evidence(claim)

    if not evidence:
        return ClaimResult(claim, gold, "NOT ENOUGH INFO", 0, 0, nei_kind="empty")

    results: list[StanceResult] = verifier.check_many(claim, evidence)
    verdict, _conf, n_voted = aggregate(results, min_votes=min_votes)
    predicted = VERDICT_TO_FEVER[verdict]

    nei_kind = ""
    if predicted == "NOT ENOUGH INFO":
        nei_kind = "reasoned"

    return ClaimResult(claim, gold, predicted, len(evidence), n_voted, nei_kind,
                       results)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def per_class_prf(results: list[ClaimResult]) -> None:
    print("\nper-class precision / recall / F1")
    print(f"{'label':18} {'prec':>6} {'rec':>6} {'f1':>6} {'support':>8}")
    for label in FEVER_LABELS:
        tp = sum(r.predicted == label and r.gold == label for r in results)
        fp = sum(r.predicted == label and r.gold != label for r in results)
        fn = sum(r.predicted != label and r.gold == label for r in results)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        support = sum(r.gold == label for r in results)
        print(f"{label:18} {prec:6.3f} {rec:6.3f} {f1:6.3f} {support:8}")

    acc = sum(r.predicted == r.gold for r in results) / len(results)
    print(f"\noverall accuracy: {acc:.3f}  (n={len(results)})")


def confusion(results: list[ClaimResult]) -> None:
    print("\nconfusion matrix (rows=gold, cols=predicted)")
    idx = {l: i for i, l in enumerate(FEVER_LABELS)}
    m = [[0] * 3 for _ in range(3)]
    for r in results:
        m[idx[r.gold]][idx[r.predicted]] += 1
    print(f"{'':18}" + "".join(f"{l[:7]:>9}" for l in FEVER_LABELS))
    for l in FEVER_LABELS:
        row = m[idx[l]]
        print(f"{l:18}" + "".join(f"{v:>9}" for v in row))


def nei_breakdown(results: list[ClaimResult]) -> None:
    """The honesty check: of NEI predictions, how many were empty vs reasoned."""
    nei = [r for r in results if r.predicted == "NOT ENOUGH INFO"]
    kinds = Counter(r.nei_kind for r in nei)
    print("\nNEI honesty breakdown (of all NEI predictions)")
    print(f"   empty retrieval (default abstain): {kinds.get('empty', 0)}")
    print(f"   reasoned (evidence found, judged insufficient): "
          f"{kinds.get('reasoned', 0)}")

    # Of gold-NEI claims we got right, how many for the right reason?
    correct_nei = [r for r in nei if r.gold == "NOT ENOUGH INFO"]
    reasoned = sum(r.nei_kind == "reasoned" for r in correct_nei)
    print(f"\n   correct NEI predictions: {len(correct_nei)}")
    print(f"      of which reasoned (not just empty retrieval): {reasoned}")
    if correct_nei:
        print(f"      honest-NEI rate: {reasoned / len(correct_nei):.2f}")


def coverage(results: list[ClaimResult]) -> None:
    with_ev = sum(r.n_evidence > 0 for r in results)
    voted = sum(r.n_voted > 0 for r in results)
    print("\nretrieval coverage")
    print(f"   claims with any evidence retrieved: {with_ev}/{len(results)}")
    print(f"   claims where evidence actually voted: {voted}/{len(results)}")


# --------------------------------------------------------------------------

def inspect_misses(verifier: Verifier, sample: list[tuple[str, str]],
                   gold_filter: str, pred_filter: str,
                   min_votes: int = 2) -> None:
    """Re-run claims matching gold->pred and dump per-piece evidence + stance."""
    shown = 0
    for claim, gold in sample:
        if gold != gold_filter:
            continue
        r = evaluate_claim(verifier, claim, gold, min_votes=min_votes)
        if r.predicted != pred_filter:
            continue
        shown += 1
        print("=" * 72)
        print(f"gold={gold}  predicted={r.predicted}  "
              f"(voted {r.n_voted}/{r.n_evidence})")
        print(f"claim: {claim}")
        print("-" * 72)
        for sr in r.results:
            gate = "PASS" if sr.relevant else "gate-drop"
            print(f"  [{sr.verdict:10}] {gate:9} rel={sr.relevance_score:+.2f} "
                  f"stance={sr.stance}({sr.stance_probs[sr.stance]:.2f})")
            print(f"     {sr.evidence[:150]}")
        print()
    if not shown:
        print(f"(no claims matched gold={gold_filter} -> pred={pred_filter})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-label", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--inspect", nargs=2, metavar=("GOLD", "PRED"),
                    help="dump evidence+stance for claims with this gold->pred, "
                         "e.g. --inspect 'NOT ENOUGH INFO' REFUTES")
    ap.add_argument("--relevance-threshold", type=float, default=None,
                    help="override stance.RELEVANCE_THRESHOLD for this run")
    ap.add_argument("--min-votes", type=int, default=None,
                    help="override stance.REQUIRE_MIN_VOTES for this run")
    args = ap.parse_args()

    ds = load_fever_dev()

    by_label = defaultdict(list)
    for row in ds:
        by_label[row["label"]].append(row["claim"])
    rng = random.Random(args.seed)
    sample = []
    for label in FEVER_LABELS:
        claims = by_label[label]
        rng.shuffle(claims)
        sample += [(c, label) for c in claims[:args.per_label]]
    rng.shuffle(sample)

    print(f"sampled {len(sample)} claims "
          f"({args.per_label} per label), seed {args.seed}")

    import stance as stance_mod
    rt = args.relevance_threshold if args.relevance_threshold is not None \
        else stance_mod.RELEVANCE_THRESHOLD
    mv = args.min_votes if args.min_votes is not None \
        else stance_mod.REQUIRE_MIN_VOTES
    if args.relevance_threshold is not None:
        print(f"[override] relevance_threshold = {rt}")
    if args.min_votes is not None:
        print(f"[override] min_votes = {mv}")

    print("loading models...")
    verifier = Verifier(relevance_threshold=rt)

    if args.inspect:
        gold_f, pred_f = args.inspect
        inspect_misses(verifier, sample, gold_f, pred_f, min_votes=mv)
        return

    results = []
    for i, (claim, gold) in enumerate(sample, 1):
        r = evaluate_claim(verifier, claim, gold, min_votes=mv)
        results.append(r)
        mark = "OK " if r.predicted == r.gold else "XX "
        print(f"  [{i:3}/{len(sample)}] {mark} gold={r.gold:16} "
              f"pred={r.predicted:16} ev={r.n_evidence} voted={r.n_voted}")
        if args.verbose and r.predicted != r.gold:
            print(f"        claim: {claim[:90]}")

    print("\n" + "=" * 72)
    per_class_prf(results)
    confusion(results)
    coverage(results)
    nei_breakdown(results)


if __name__ == "__main__":
    main()