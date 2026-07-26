from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Label convention throughout this project: 1 = fake/false, 0 = real/true.
LABELS = {0: "real", 1: "fake"}

# ISOT's True.csv articles nearly all open with a wire-service dateline like
# "WASHINGTON (Reuters) - ". Fake.csv has no such prefix. This is the leak.
DATELINE = re.compile(r"^\s*[A-Z][A-Za-z\s.,/'-]{0,40}\(Reuters\)\s*[-\u2013]\s*")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_isot(data_dir: str | Path, strip_dateline: bool = False) -> pd.DataFrame:
    d = Path(data_dir)
    true_df = pd.read_csv(d / "True.csv")
    fake_df = pd.read_csv(d / "Fake.csv")
    true_df["label"] = 0
    fake_df["label"] = 1

    df = pd.concat([true_df, fake_df], ignore_index=True)
    # Combine the title and body into one text field so the classifier sees the full article.
    df["text"] = (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()

    if strip_dateline:
        df["text"] = df["text"].str.replace(DATELINE, "", regex=True)
        df["text"] = df["text"].str.replace(r"\(Reuters\)", "", regex=True)

    return df[["text", "label"]].dropna().query("text != ''").reset_index(drop=True)


LIAR_COLS = [
    "id", "label", "statement", "subject", "speaker", "job_title", "state_info",
    "party_affiliation", "barely_true_counts", "false_counts", "half_true_counts",
    "mostly_true_counts", "pants_on_fire_counts", "context",
]

# LIAR ships 6 ordinal labels. Collapsing to binary makes it comparable to ISOT
# -- and makes the accuracy gap between the two datasets impossible to explain
# away as "different number of classes".
LIAR_BINARY = {
    "true": 0, "mostly-true": 0, "half-true": 0,
    "barely-true": 1, "false": 1, "pants-fire": 1,
}


def load_liar(data_dir: str | Path, split: str = "train") -> pd.DataFrame:
    path = Path(data_dir) / f"{split}.tsv"
    df = pd.read_csv(path, sep="\t", header=None, names=LIAR_COLS, quoting=3)
    df["label"] = df["label"].str.strip().map(LIAR_BINARY)
    df = df.rename(columns={"statement": "text"})
    return df[["text", "label"]].dropna().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_pipeline(max_features: int = 50_000, ngram: tuple[int, int] = (1, 2)) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram,
            sublinear_tf=True,
            min_df=2,
            stop_words="english",
        )),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")),
    ])


def top_features(pipe: Pipeline, n: int = 20) -> pd.DataFrame:
    """The single most useful diagnostic in this whole file."""
    names = np.array(pipe.named_steps["tfidf"].get_feature_names_out())
    coefs = pipe.named_steps["clf"].coef_[0]
    order = np.argsort(coefs)
    return pd.DataFrame({
        "pushes_toward_real": names[order[:n]],
        "real_weight": np.round(coefs[order[:n]], 3),
        "pushes_toward_fake": names[order[-n:]][::-1],
        "fake_weight": np.round(coefs[order[-n:]][::-1], 3),
    })


def report(pipe: Pipeline, X_test, y_test, header: str = "") -> float:
    pred = pipe.predict(X_test)
    acc = (pred == y_test).mean()
    if header:
        print(f"\n=== {header} ===")
    print(f"accuracy: {acc:.4f}")
    print(classification_report(y_test, pred, target_names=["real", "fake"], digits=3))
    print("confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, pred))
    return acc


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["isot", "liar"], required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--strip-dateline", action="store_true",
                    help="ISOT only: remove the Reuters dateline before training")
    ap.add_argument("--save", default="")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.dataset == "isot":
        df = load_isot(args.data_dir, strip_dateline=args.strip_dateline)
        # Keep class proportions balanced across train and test splits for a fair evaluation.
        X_tr, X_te, y_tr, y_te = train_test_split(
            df["text"], df["label"], test_size=0.2,
            random_state=args.seed, stratify=df["label"],
        )
    else:
        tr = load_liar(args.data_dir, "train")
        te = load_liar(args.data_dir, "test")
        X_tr, y_tr, X_te, y_te = tr["text"], tr["label"], te["text"], te["label"]

    print(f"train: {len(X_tr):,}   test: {len(X_te):,}   "
          f"fake rate: {y_tr.mean():.3f}")

    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)

    label = f"{args.dataset}" + (" (dateline stripped)" if args.strip_dateline else "")
    report(pipe, X_te, y_te, header=label)

    print("\ntop weighted features:")
    print(top_features(pipe, n=15).to_string(index=False))

    if args.save:
        import joblib
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, args.save)
        print(f"\nsaved -> {args.save}")


if __name__ == "__main__":
    main()
