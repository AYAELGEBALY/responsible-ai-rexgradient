"""Sanity check: is the data in place and does the encoder run?

    python scripts/check_setup.py

Expected output (numbers may differ slightly if the CSV is updated):
  10000 studies, 17007 image rows
  embedding shape (5, 768)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.data import binary_label, findings_texts, label_counts, load_metadata, study_table
from src.embeddings import EMBEDDING_DIM, embed_texts, get_device, load_encoder


def main() -> None:
    df = load_metadata()
    studies = study_table(df)
    print(f"{len(studies)} studies, {len(df)} image rows, {studies['patient_id'].nunique()} patients")

    for obs in ("edema", "pleural_effusion"):
        counts = label_counts(studies, obs).to_dict()
        clear = int(binary_label(studies, obs).notna().sum())
        print(f"{obs:17s} CheXbert status counts: {counts} -> {clear} clear (positive/negative) studies")

    print(f"device: {get_device()} | loading BioClinicalBERT (downloads ~400 MB the first time)")
    tokenizer, model = load_encoder()
    X = embed_texts(findings_texts(studies)[:5], tokenizer, model, verbose=False)
    print(f"embedding shape {X.shape} (expected (5, {EMBEDDING_DIM}))")
    print("\nSetup looks good." if X.shape == (5, EMBEDDING_DIM) else "\nUnexpected embedding shape!")


if __name__ == "__main__":
    main()
