"""Loading + light cleaning. `text` keeps the original; `clean_text`/`model_input` feed the classifier."""
import re
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
COLUMNS = ["article_id", "title", "source", "url", "publication_date", "text", "collection_status"]


def clean_text(raw) -> str:
    """Strip HTML tags, normalise whitespace. Case, digits, %, $ and negations are deliberately KEPT."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    return re.sub(r"\s+", " ", BeautifulSoup(str(raw), "html.parser").get_text(" ")).strip()


def load_articles(path=ROOT / "data" / "sample_articles.csv") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_json(path) if path.suffix == ".json" else pd.read_csv(path)
    for col in COLUMNS:                                   # tolerate missing columns
        if col not in df.columns:
            df[col] = ""
    df = df.fillna("")
    df["title"] = df["title"].map(clean_text)
    df["clean_text"] = df["text"].map(clean_text)
    # title + body is the classifier input (headline often carries the event)
    df["model_input"] = (df["title"] + ". " + df["clean_text"]).where(df["title"] != "", df["clean_text"])
    df.loc[df["clean_text"] == "", "collection_status"] = "missing_text"
    df = df.drop_duplicates(subset=["title", "clean_text"])                      # exact duplicates
    df = df[~((df["url"] != "") & df.duplicated(subset="url"))]                  # duplicate URLs
    return df.reset_index(drop=True)
