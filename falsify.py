from __future__ import annotations

import argparse

from sklearn.model_selection import train_test_split

from baseline import build_pipeline, load_isot, load_liar

SEED = 42


def _split(df):
    return train_test_split(df["text"], df["label"], test_size=0.2,
                            random_state=SEED, stratify=df["label"])


def _acc(pipe, X, y) -> float:
    return float((pipe.predict(X) == y).mean())


# ---------------------------------------------------------------------------

def exp1_prefix_probe(isot_dir: str, n_chars: int = 40) -> None:
    """If a 40-char prefix predicts the label, the model isn't reading content."""
    print("\n" + "=" * 62)
    print(f"EXPERIMENT 1 -- prefix probe (first {n_chars} characters only)")
    print("=" * 62)

    df = load_isot(isot_dir)
    df["text"] = df["text"].str[:n_chars]
    X_tr, X_te, y_tr, y_te = _split(df)

    pipe = build_pipeline(max_features=5_000, ngram=(1, 3))
    pipe.fit(X_tr, y_tr)
    acc = _acc(pipe, X_te, y_te)

    print(f"accuracy from {n_chars} characters: {acc:.4f}")
    print("\nsample prefixes:")
    for txt, lab in list(zip(X_te, y_te))[:6]:
        print(f"  [{'fake' if lab else 'real'}] {txt!r}")
    print("\nInterpretation: a real detector cannot work here. Anything well above")
    print("0.50 is the model reading a formatting artifact, not a factual signal.")


def exp2_dateline_ablation(isot_dir: str) -> None:
    print("\n" + "=" * 62)
    print("EXPERIMENT 2 -- dateline ablation")
    print("=" * 62)

    results = {}
    for strip in (False, True):
        df = load_isot(isot_dir, strip_dateline=strip)
        X_tr, X_te, y_tr, y_te = _split(df)
        pipe = build_pipeline()
        pipe.fit(X_tr, y_tr)
        results["stripped" if strip else "intact"] = _acc(pipe, X_te, y_te)

    print(f"dateline intact:   {results['intact']:.4f}")
    print(f"dateline stripped: {results['stripped']:.4f}")
    print(f"delta:             {results['intact'] - results['stripped']:+.4f}")
    print("\nInterpretation: whatever the drop is, that share of the headline")
    print("accuracy was the Reuters prefix. Note the remainder is still style,")
    print("not truth -- experiment 3 shows that.")


def exp3_cross_dataset(isot_dir: str, liar_dir: str) -> None:
    """The decisive test. Same task, different source -- does anything transfer?"""
    print("\n" + "=" * 62)
    print("EXPERIMENT 3 -- cross-dataset transfer")
    print("=" * 62)

    isot = load_isot(isot_dir)
    pipe = build_pipeline()
    pipe.fit(isot["text"], isot["label"])

    liar_te = load_liar(liar_dir, "test")
    acc = _acc(pipe, liar_te["text"], liar_te["label"])
    majority = max(liar_te["label"].mean(), 1 - liar_te["label"].mean())

    print(f"trained on ISOT, tested on LIAR: {acc:.4f}")
    print(f"majority-class baseline on LIAR: {majority:.4f}")
    print(f"lift over always guessing the majority class: {acc - majority:+.4f}")

    liar_tr = load_liar(liar_dir, "train")
    native = build_pipeline().fit(liar_tr["text"], liar_tr["label"])
    print(f"same model architecture trained natively on LIAR: "
          f"{_acc(native, liar_te['text'], liar_te['label']):.4f}")
    print("\nInterpretation: the architecture did not get worse between datasets.")
    print("ISOT is separable by style; LIAR is not. The 99% measured the corpus.")


# Style and truth are deliberately crossed here: true statements written badly,
# false statements written like a wire report.
ADVERSARIAL = [
    ("The Earth orbits the Sun once every 365.25 days.", 0, "true, plain"),
    ("SHOCKING: Scientists CONFIRM the Earth orbits the Sun -- "
     "here's what they don't want you to know!!!", 0, "true, tabloid voice"),
    ("WASHINGTON (Reuters) - The Department of Health confirmed on Tuesday that "
     "drinking bleach cures respiratory infections, according to a spokesman.",
     1, "false, wire voice"),
    ("Vaccines cause autism, share this before they delete it!!!", 1, "false, tabloid"),
    ("The Great Wall of China is visible to the naked eye from the Moon.",
     1, "false, plain"),
    ("BRUSSELS (Reuters) - European officials said the summit would resume Thursday.",
     0, "true, wire voice"),
]


def exp4_adversarial(isot_dir: str) -> None:
    print("\n" + "=" * 62)
    print("EXPERIMENT 4 -- style/truth mismatch")
    print("=" * 62)

    isot = load_isot(isot_dir)
    pipe = build_pipeline().fit(isot["text"], isot["label"])

    texts = [t for t, _, _ in ADVERSARIAL]
    preds = pipe.predict(texts)
    probs = pipe.predict_proba(texts)[:, 1]

    correct = 0
    for (text, gold, note), p, pr in zip(ADVERSARIAL, preds, probs):
        ok = p == gold
        correct += ok
        print(f"\n  {'OK ' if ok else 'MISS'} | {note}")
        print(f"       gold={'fake' if gold else 'real'}  "
              f"pred={'fake' if p else 'real'}  p(fake)={pr:.2f}")
        print(f"       {text[:90]}")

    print(f"\naccuracy on 6 hand-written cases: {correct}/{len(ADVERSARIAL)}")
    print("\nInterpretation: the errors are not random. The model follows voice.")
    print("This is the slide that motivates evidence retrieval.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--isot-dir", required=True)
    ap.add_argument("--liar-dir", default="")
    ap.add_argument("--only", type=int, choices=[1, 2, 3, 4])
    args = ap.parse_args()

    run = {args.only} if args.only else {1, 2, 3, 4}

    if 1 in run:
        exp1_prefix_probe(args.isot_dir)
    if 2 in run:
        exp2_dateline_ablation(args.isot_dir)
    if 3 in run:
        if args.liar_dir:
            exp3_cross_dataset(args.isot_dir, args.liar_dir)
        else:
            print("\n[skipped experiment 3: pass --liar-dir]")
    if 4 in run:
        exp4_adversarial(args.isot_dir)


if __name__ == "__main__":
    main()
