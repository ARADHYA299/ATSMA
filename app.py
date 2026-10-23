"""Gradio app. Run:  python app.py   (after  python train_model.py)"""
import json
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import company_hierarchy as ch
from classifier import METRICS_PATH, load_model, predict
from information_extractor import display_fields, extract
from text_preprocessing import clean_text, load_articles

DISCLAIMER = "*Prediction and extraction are automated outputs, not verified facts.*"


def analyze(title, text, url, bundle=None):
    """Classify -> extract. Returns (prediction dict, extraction dict). Raises FileNotFoundError if model missing."""
    bundle = bundle or load_model()
    text = clean_text(text)
    pred = predict(f"{title}. {text}" if title else text, bundle)
    return pred, extract(text, pred["label"], url or "", title or "")


def _render(pred, ext):
    warn = list(ext["warnings"])
    if pred["uncertain"]:
        warn.insert(0, "Low-confidence prediction: treat the category as a guess.")
    md = (f"## {pred['label']}\nModel confidence score: **{pred['confidence']:.0%}** "
          f"(top-3: {', '.join(f'{l} {p:.0%}' for l, p in pred['top3'])})  \n"
          f"Key n-grams behind the prediction: `{', '.join(pred['key_terms']) or 'none'}`  \n"
          f"Single-label model: an article may contain several events.  \n{DISCLAIMER}")
    if warn:
        md += "\n\n" + "\n".join(f"⚠️ {w}" for w in warn)
    return md, pd.DataFrame(display_fields(ext), columns=["Field", "Value"])


def analyze_ui(title, text, url):
    if not clean_text(text):
        return "Please paste some article text.", pd.DataFrame(columns=["Field", "Value"])
    try:
        return _render(*analyze(title, text, url))
    except FileNotFoundError as e:
        return f"❌ {e}", pd.DataFrame(columns=["Field", "Value"])


def cm_fig(m, title):
    cm, labels = np.array(m["confusion"]), m["labels"]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), [l[:14] for l in labels], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)), [l[:14] for l in labels], fontsize=7)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=9); fig.tight_layout()
    return fig


def build_ui():
    articles = load_articles()
    try:
        bundle = load_model()
        results = {r.article_id: analyze(r.title, r.clean_text, r.url, bundle) for r in articles.itertuples()}
        edges = [e for i, (p, x) in results.items() for e in ch.infer_edges(x, i)]
        model_msg = ""
    except FileNotFoundError as e:
        bundle, results, edges, model_msg = None, {}, [], f"❌ {e}"
    graph = ch.build_graph(inferred=edges)
    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None
    test_ids = set(metrics["test_ids"]) if metrics else set()

    with gr.Blocks(title="Financial News Intelligence") as demo:
        gr.Markdown("# Financial News Classification, Event Extraction & Company Hierarchy\n"
                    "Academic NLP lab project: TF-IDF + Logistic Regression, spaCy NER, regex rules, NetworkX. "
                    "Demo data is **synthetic**. " + model_msg)
        with gr.Tab("Analyze Article"):
            t, x, u = gr.Textbox(label="Title (optional)"), gr.Textbox(label="Article text", lines=6), gr.Textbox(label="Source URL (optional)")
            gr.Examples([["", "Alpha Corp agreed to acquire Beta Ltd for $500 million on 12 March 2024. The deal is subject to regulatory approval.", ""],
                         ["", "Falcon Industries rejected an unsolicited $6 billion takeover offer from Raptor Capital on 5 May 2024.", ""]], [t, x, u])
            o1, o2 = gr.Markdown(), gr.Dataframe(label="Extracted fields", wrap=True)
            gr.Button("Classify and extract", variant="primary").click(analyze_ui, [t, x, u], [o1, o2])
        with gr.Tab("News Dataset"):
            tbl = pd.DataFrame([{"ID": r.article_id, "Date": r.publication_date, "Source": r.source, "Title": r.title,
                                 "True label": r.label, "Predicted": results[r.article_id][0]["label"] if results else "n/a",
                                 "Confidence": f"{results[r.article_id][0]['confidence']:.0%}" if results else "n/a",
                                 "Held-out test": "yes" if r.article_id in test_ids else "no"} for r in articles.itertuples()])
            gr.Dataframe(tbl, wrap=True)
            pick = gr.Dropdown([f"{r.article_id} | {r.title}" for r in articles.itertuples()], label="Select article")
            d1, d2, d3 = gr.Textbox(label="Article text", lines=3), gr.Markdown(), gr.Dataframe(wrap=True)

            def show(choice):
                if not choice or not results:
                    return "", model_msg or "Select an article.", pd.DataFrame()
                aid = choice.split(" | ")[0]
                md, df = _render(*results[aid])
                return articles.set_index("article_id").loc[aid, "clean_text"], md, df
            pick.change(show, pick, [d1, d2, d3])
        with gr.Tab("Company Hierarchy"):
            q = gr.Textbox(label="Company name", placeholder="e.g. Microsoft, Alpha Corp")
            h1, h2 = gr.Markdown(), gr.Plot()
            q.submit(lambda s: ch.lookup(graph, s), q, [h1, h2])
            gr.Button("Search").click(lambda s: ch.lookup(graph, s), q, [h1, h2])
        with gr.Tab("Model Evaluation"):
            if not metrics:
                gr.Markdown("❌ No metrics found. Run `python train_model.py`.")
            else:
                h = metrics["heldout"]
                gr.Markdown(f"**Held-out test set:** {metrics['n_test']} hand-written articles (never used for training; "
                            f"training set: {metrics['n_train']} template articles).  \n"
                            f"Accuracy **{h['accuracy']:.2f}** | macro-precision **{h['macro_precision']:.2f}** | "
                            f"macro-recall **{h['macro_recall']:.2f}** | macro-F1 **{h['macro_f1']:.2f}**  \n⚠️ {metrics['data_note']}")
                per = pd.DataFrame([{"Class": l, "Precision": round(h["per_class"][l]["precision"], 2), "Recall": round(h["per_class"][l]["recall"], 2),
                                     "F1": round(h["per_class"][l]["f1-score"], 2), "Support": int(h["per_class"][l]["support"])} for l in h["labels"]])
                gr.Dataframe(per); gr.Plot(cm_fig(h, "Confusion matrix (held-out test set)"))
    return demo


if __name__ == "__main__":
    build_ui().launch()
