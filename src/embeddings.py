"""Text representations: frozen BioClinicalBERT embeddings.

(The TF-IDF baseline lives in `src/probes.py` as `make_tfidf_probe`, because its
vocabulary must be fitted inside each training fold.)

The encoder is FROZEN: we never train it. A report becomes one vector of
768 numbers. The default is mean pooling over the last layer, which everyone uses
for the main comparison. CLS pooling and other layers are optional explorations.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_ENCODER = "emilyalsentzer/Bio_ClinicalBERT"
EMBEDDING_DIM = 768
EMBEDDINGS_DIR = Path("data/embeddings")


def get_device() -> torch.device:
    """Use the Apple GPU (mps) or an NVIDIA GPU if available, otherwise the CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_encoder(name: str = DEFAULT_ENCODER, device: torch.device | None = None):
    """Load the tokenizer and the frozen encoder. Returns (tokenizer, model)."""
    device = device or get_device()
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name, output_hidden_states=True).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return tokenizer, model


def embed_texts(
    texts: list[str],
    tokenizer=None,
    model=None,
    pooling: str = "mean",
    layer: int = -1,
    batch_size: int = 32,
    max_length: int = 512,
    verbose: bool = True,
) -> np.ndarray:
    """Embed a list of texts with the frozen encoder.

    pooling  "mean": average of all token vectors (default)
             "cls":  the vector of the first [CLS] token
    layer    which hidden layer to read: -1 is the last layer (default),
             0 is the (non-contextual) input embeddings, 1..12 are the BERT layers.

    Returns an array of shape (len(texts), 768).
    """
    if tokenizer is None or model is None:
        tokenizer, model = load_encoder()
    device = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length,
                            return_tensors="pt").to(device)
            hidden = model(**enc).hidden_states[layer]          # (batch, tokens, 768)
            if pooling == "mean":
                mask = enc["attention_mask"].unsqueeze(-1).float()
                vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            elif pooling == "cls":
                vec = hidden[:, 0, :]
            else:
                raise ValueError("pooling must be 'mean' or 'cls'")
            out.append(vec.cpu().numpy())
            if verbose and (start // batch_size) % 25 == 0:
                print(f"  embedded {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    return np.vstack(out).astype(np.float32)


def embedding_path(text: str = "findings", pooling: str = "mean", layer: int = -1) -> Path:
    """Where `scripts/make_embeddings.py` stores an embedding matrix."""
    return EMBEDDINGS_DIR / f"{text}_{pooling}_layer{layer}.npy"


def save_embeddings(X: np.ndarray, study_ids: pd.Series, path: Path) -> None:
    """Save the matrix and, next to it, the study ids in the same row order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, X)
    pd.DataFrame({"study_id": study_ids.values}).to_csv(path.with_suffix(".study_ids.csv"), index=False)


def load_embeddings(path: Path, studies: pd.DataFrame) -> np.ndarray:
    """Load a saved matrix and check that its rows match the given study table."""
    X = np.load(path)
    ids = pd.read_csv(path.with_suffix(".study_ids.csv"))["study_id"]
    if len(ids) != len(studies) or not (ids.values == studies["study_id"].values).all():
        raise ValueError("Embedding rows do not match the study table. Re-run scripts/make_embeddings.py.")
    return X
