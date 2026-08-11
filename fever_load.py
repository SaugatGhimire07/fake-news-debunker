"""
Phase 4 step 1: load FEVER and inspect its real structure.

FEVER (fever/fever on HF) still ships a loading script, so on datasets >= 4.5
the normal load_dataset("fever", "v1.0") fails with "Dataset scripts are no
longer supported" -- the same wall LIAR hit. The fix is the auto-generated
parquet revision, which needs no script.

This script ONLY loads a slice and prints structure. Build the harness after
seeing the real field names -- do not trust the docs.

Run:
    pip install datasets
    python fever_load.py
"""

from __future__ import annotations

REPO = "fever/fever"
# The default config collapses splits to train/validation/test, hiding
# labelled_dev. discover() revealed the real parquet layout, so we load the
# file directly by hf:// path and bypass the config machinery entirely.
DEV_PARQUET = f"hf://datasets/{REPO}@refs/convert/parquet/v1.0/labelled_dev/0000.parquet"
WIKI_GLOB = f"hf://datasets/{REPO}@refs/convert/parquet/wiki_pages/partial-wikipedia_pages/*.parquet"


def discover() -> None:
    """Print available parquet files so we stop guessing names."""
    from huggingface_hub import HfApi
    api = HfApi()
    files = api.list_repo_files(REPO, repo_type="dataset",
                                revision="refs/convert/parquet")
    for f in sorted(files):
        if f.endswith(".parquet"):
            print(f"   {f}")


def load_fever_dev(n: int | None = None):
    """Load the labelled_dev split directly from its parquet file."""
    from datasets import load_dataset
    ds = load_dataset("parquet", data_files=DEV_PARQUET, split="train")
    print(f"[loaded {len(ds):,} rows from labelled_dev]")
    return ds.select(range(n)) if n else ds


def load_wiki(n: int | None = None):
    """Load the partial Wikipedia pages (evidence text lives here)."""
    from datasets import load_dataset
    ds = load_dataset("parquet", data_files=WIKI_GLOB, split="train")
    print(f"[loaded {len(ds):,} wiki pages]")
    return ds.select(range(n)) if n else ds


def inspect(ds) -> None:
    print(f"\nrows: {len(ds):,}")
    print(f"columns: {ds.column_names}\n")

    from collections import Counter
    labels = Counter(ds["label"]) if "label" in ds.column_names else {}
    print("label distribution:")
    for k, v in labels.items():
        print(f"   {k:16} {v:,}")

    print("\n--- first Supported claim ---")
    _show_first(ds, "SUPPORTS")
    print("\n--- first Refuted claim ---")
    _show_first(ds, "REFUTES")
    print("\n--- first NotEnoughInfo claim ---")
    _show_first(ds, "NOT ENOUGH INFO")


def _show_first(ds, label: str) -> None:
    for row in ds:
        if row.get("label") == label:
            for k, v in row.items():
                s = repr(v)
                print(f"   {k}: {s[:200]}")
            return
    print(f"   (no {label} row in this slice)")


if __name__ == "__main__":
    ds = load_fever_dev()
    inspect(ds)

    print("\n" + "=" * 60)
    print("WIKI PAGES structure (evidence text source)")
    print("=" * 60)
    wiki = load_wiki(n=3)
    print(f"columns: {wiki.column_names}")
    for row in wiki:
        for k, v in row.items():
            print(f"   {k}: {repr(v)[:180]}")
        print()