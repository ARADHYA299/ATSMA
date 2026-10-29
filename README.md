# Financial News Classification, Event Extraction & Company Hierarchy (NLP lab project)

TF-IDF + Logistic Regression classifier, spaCy NER + regex extraction rules, NetworkX hierarchy, Gradio UI.
Runs offline on a normal laptop (no GPU, no API keys, no paid services).

## Install and run
Windows:
```
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python train_model.py
python app.py
```
Linux/macOS: replace the activate line with `source venv/bin/activate`. Open the printed local URL.
Only `en_core_web_sm` (~12 MB) is downloaded; if it is missing the app still runs with regex rules and shows a warning.
Tests: `python -m pytest -q tests`. Optional live RSS: `python news_collector.py <rss_url>` (respects robots.txt, 1 s delay; not needed for the app).

## Files
`text_preprocessing.py` load/clean/dedupe | `classifier.py` train, evaluate, predict | `train_model.py` builds template training data, trains | `information_extractor.py` NER + rules | `company_hierarchy.py` graph | `app.py` UI | `data/` CSVs | `models/`, `outputs/` generated artifacts.

## Data (read this)
- `data/sample_articles.csv`: 40 hand-written, SYNTHETIC articles (fictional companies, no URLs). Used ONLY as the held-out test set.
- `data/train_synthetic.csv`: 192 TEMPLATE-generated synthetic sentences (written by `train_model.py`, fixed seed). Used ONLY for training. Disjoint company names; the script asserts no text overlap.
- `data/company_relationships.csv`: well-known real parent/subsidiary links marked `reported_ownership / needs_filing_check`. Verify each against the company's annual report (e.g. 10-K Exhibit 21) before changing it to `confirmed_ownership`. Article-derived links are added at runtime as `inferred_from_article`.

## Measured results (from `python train_model.py`, held-out 40 articles)
accuracy 0.95, macro precision 0.96, macro recall 0.95, macro-F1 0.95. Errors: S003 (acquisition predicted as divestiture), S037 (market news predicted as regulatory).
These are NOT real-world accuracy: all data is synthetic and the same author wrote templates and test articles, so wording overlap likely inflates scores.

## Limitations
Single-label classification (articles can contain several events); rules cover common sentence patterns only; NER can mislabel unfamiliar names; "date mentioned" is the first date in the evidence sentence, not a verified announcement date; confidence is a model score, not a probability of being correct; live RSS/HTML collection was not tested against real sites.

##Vedio link : https://drive.google.com/uc?id=1KxtIiMuQSuz34XCb8U3g-xoFh-Tn9Odv&export=download
