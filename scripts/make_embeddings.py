"""Embed the Findings text of every study with frozen BioClinicalBERT and cache the result.

    python scripts/make_embeddings.py                       # default: mean pooling, last layer
    python scripts/make_embeddings.py --pooling cls         # optional exploration
    python scripts/make_embeddings.py --layer 6             # optional exploration

Output: data/embeddings/findings_<pooling>_layer<layer>.npy  (shape: n_studies x 768)
        plus a .study_ids.csv file with the row order.
Takes a few minutes on a laptop. Load the result with `src.embeddings.load_embeddings`.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import findings_texts, study_table
from src.embeddings import embed_texts, embedding_path, load_encoder, save_embeddings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    ap.add_argument("--layer", type=int, default=-1, help="-1 = last layer, 0..12 = BERT layers")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    studies = study_table()
    texts = findings_texts(studies)
    print(f"Embedding Findings of {len(texts)} studies (pooling={args.pooling}, layer={args.layer})")

    tokenizer, model = load_encoder()
    X = embed_texts(texts, tokenizer, model, pooling=args.pooling, layer=args.layer, batch_size=args.batch_size)

    path = embedding_path("findings", args.pooling, args.layer)
    save_embeddings(X, studies["study_id"], path)
    print(f"saved {path}  shape={X.shape}")


if __name__ == "__main__":
    main()
