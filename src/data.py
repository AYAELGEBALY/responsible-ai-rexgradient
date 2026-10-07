"""Loading the ReXGradient metadata CSV and preparing texts and labels.

The CSV has ONE ROW PER IMAGE. Most of the project works at the STUDY level
(one report per study), so start with `study_table()`.
"""

from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/rexgradient_valid_metadata.csv")

# The 14 observations labelled by CheXbert (column names: chexbert_<observation>)
CHEXBERT_OBSERVATIONS = [
    "enlarged_cardiomediastinum", "cardiomegaly", "lung_opacity", "lung_lesion",
    "edema", "consolidation", "pneumonia", "atelectasis", "pneumothorax",
    "pleural_effusion", "pleural_other", "fracture", "support_devices", "no_finding",
]


def load_metadata(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the image-level CSV. Gives a clear error if the file is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download rexgradient_valid_metadata.csv from the course "
            f"webpage and place it in the data/ folder (see the main README)."
        )
    return pd.read_csv(path)


def study_table(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per study, sorted by study_id, with a few extra convenience columns.

    Extra columns
    -------------
    health_system     the hospital network a study comes from (prefix of `institution`)
    findings_words    number of words in the Findings text
    impression_words  number of words in the Impression text
    """
    if df is None:
        df = load_metadata()
    studies = df.drop_duplicates("study_id").sort_values("study_id").reset_index(drop=True)
    studies["health_system"] = studies["institution"].str.rsplit("-", n=1).str[0]
    studies["findings_words"] = studies["findings"].fillna("").str.split().str.len()
    studies["impression_words"] = studies["impression"].fillna("").str.split().str.len()
    return studies


def findings_texts(studies: pd.DataFrame) -> list[str]:
    """The Findings section of each report. This is the model input in this project."""
    return studies["findings"].fillna("").tolist()


def binary_label(studies: pd.DataFrame, observation: str) -> pd.Series:
    """Silver label for one pathology from CheXbert run on the Impression.

    Returns 1.0 for 'positive', 0.0 for 'negative', and NaN for 'uncertain' or
    'blank' (not mentioned). Studies with NaN should be EXCLUDED from the probe,
    not treated as negatives.
    """
    status = studies[f"chexbert_{observation}"]
    if observation == "no_finding":
        return status.map({"yes": 1.0, "no": 0.0})
    return status.map({"positive": 1.0, "negative": 0.0})


def label_counts(studies: pd.DataFrame, observation: str) -> pd.Series:
    """How many studies are positive / negative / uncertain / blank for one observation."""
    return studies[f"chexbert_{observation}"].value_counts(dropna=False)


def usable_findings(studies: pd.DataFrame, min_words: int = 3) -> pd.Series:
    """True for studies whose Findings text is present and has at least `min_words` words.

    Exclude the other studies from representation experiments: an empty text still
    produces a 768-number vector, but it does not represent a report.
    """
    words = studies["findings"].fillna("").str.split().str.len()
    return studies["findings"].notna() & (words >= min_words)


def top_k_classes(values: pd.Series, k: int = 10) -> pd.Series:
    """Keep the k most frequent classes, set all other classes to NaN (excluded).

    Standard rule for targets with many rare classes, e.g. `institution` (66 sites):
    `top_k_classes(studies["institution"], 10)`.
    """
    keep = values.value_counts().head(k).index
    return values.where(values.isin(keep))
