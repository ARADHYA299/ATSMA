"""TF-IDF (1-2 grams) + Logistic Regression: training, evaluation, prediction with uncertainty."""
import json
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "event_model.joblib"
METRICS_PATH = ROOT / "outputs" / "metrics.json"
UNCERTAIN_BELOW, MIN_MARGIN = 0.40, 0.10     # heuristic thresholds, not calibrated guarantees


def _pipe():
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                     ("clf", LogisticRegression(C=10, max_iter=2000, class_weight="balanced"))])


def _scores(y_true, y_pred, labels):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {"n": len(y_true), "accuracy": accuracy_score(y_true, y_pred), "macro_precision": p,
            "macro_recall": r, "macro_f1": f, "labels": labels,
            "per_class": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
            "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist()}


def train(train_df, test_df, model_path=MODEL_PATH, metrics_path=METRICS_PATH):
    """Fit on train_df; evaluate on a SEPARATE hand-written test_df (no template leakage). Saves model + metrics."""
    labels = sorted(set(train_df["label"]) | set(test_df["label"]))
    pipe = _pipe().fit(train_df["model_input"], train_df["label"])
    metrics = {"data_note": "Training data is synthetic (templates); test set is hand-written but also synthetic "
                            "(fictional companies). Scores are indicative only, NOT real-world accuracy.",
               "n_train": len(train_df), "n_test": len(test_df), "test_ids": test_df["article_id"].tolist(),
               "heldout": _scores(test_df["label"], pipe.predict(test_df["model_input"]), labels)}
    Path(model_path).parent.mkdir(exist_ok=True); Path(metrics_path).parent.mkdir(exist_ok=True)
    # vectorizer and model are stored as separate objects in one joblib file
    joblib.dump({"vectorizer": pipe["tfidf"], "model": pipe["clf"], "labels": labels}, model_path)
    Path(metrics_path).write_text(json.dumps(metrics, indent=2, default=float))
    return metrics


def load_model(path=MODEL_PATH):
    if not Path(path).exists():
        raise FileNotFoundError("Model not found. Run:  python train_model.py")
    return joblib.load(path)


def predict(text, bundle):
    vec, clf = bundle["vectorizer"], bundle["model"]
    x = vec.transform([text or ""])
    probs = clf.predict_proba(x)[0]
    order = probs.argsort()[::-1]
    conf, margin = float(probs[order[0]]), float(probs[order[0]] - probs[order[1]])
    contrib = x.multiply(clf.coef_[order[0]]).toarray()[0]          # which n-grams pushed this class up
    names = vec.get_feature_names_out()
    terms = [names[i] for i in contrib.argsort()[::-1][:5] if contrib[i] > 0]
    return {"label": str(clf.classes_[order[0]]), "confidence": conf, "margin": margin,
            "uncertain": x.nnz == 0 or conf < UNCERTAIN_BELOW or margin < MIN_MARGIN,
            "top3": [(str(clf.classes_[i]), float(probs[i])) for i in order[:3]], "key_terms": terms}
