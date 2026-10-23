import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import company_hierarchy as ch
from classifier import MODEL_PATH, load_model, predict, train
from information_extractor import NF, extract
from text_preprocessing import clean_text, load_articles
from train_model import build_training_set


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    d = tmp_path_factory.mktemp("m")
    tr = build_training_set(); p = d / "t.csv"; tr.to_csv(p, index=False)
    train(load_articles(p), load_articles(), d / "m.joblib", d / "m.json")
    return load_model(d / "m.joblib")


def test_load_and_preprocess():
    df = load_articles()
    assert len(df) == 40 and df["model_input"].str.len().min() > 0
    assert clean_text("<p>Alpha  <b>Corp</b>\n bought</p>") == "Alpha Corp bought"


def test_empty_text_safe(bundle):
    assert clean_text(None) == "" and clean_text(float("nan")) == ""
    r = extract("", "Acquisition"); assert r["acquirer"] == NF and r["warnings"]
    assert predict("", bundle)["uncertain"]


def test_train_and_load(bundle):
    assert set(bundle) == {"vectorizer", "model", "labels"} and len(bundle["labels"]) == 8


def test_acquisition_classified(bundle):
    assert predict("Alpha Corp agreed to acquire Beta Ltd for $500 million.", bundle)["label"] == "Acquisition"


@pytest.mark.parametrize("s,a,t", [("Alpha Corp agreed to acquire Beta Ltd for $500 million.", "Alpha Corp", "Beta Ltd"),
                                   ("Gamma Holdings acquired Delta Software.", "Gamma Holdings", "Delta Software"),
                                   ("Company A agreed to buy Company B.", "Company A", "Company B"),
                                   ("Raptor Capital launched a hostile bid for Cobalt Chemicals.", "Raptor Capital", "Cobalt Chemicals")])
def test_acquirer_target(s, a, t):
    r = extract(s, "Hostile Takeover / Attempted Takeover" if "hostile" in s else "Acquisition")
    assert (r["acquirer"], r["target"]) == (a, t)


def test_money_and_date():
    r = extract("Alpha Corp agreed to acquire Beta Ltd for $500 million on 12 March 2024.", "Acquisition")
    assert (r["deal_value"], r["currency"], r["announcement_date"]) == ("500 million", "USD", "12 March 2024")


def test_missing_not_fabricated():
    r = extract("The company reported results.", "Acquisition")
    assert r["acquirer"] == r["target"] == r["deal_value"] == r["announcement_date"] == NF


def test_proposed_or_rejected_is_not_confirmed_ownership():
    rej = extract("Falcon Industries rejected an unsolicited $6 billion takeover offer from Raptor Capital.", "Hostile Takeover / Attempted Takeover")
    assert rej["status"].startswith("Not completed") and ch.infer_edges(rej, "x") == []
    pro = extract("Nimbus Cloud will acquire Quarry AI for $85 million.", "Acquisition")
    e = ch.infer_edges(pro, "x")[0]
    assert e["status"] == "proposed_acquisition" and e["verification"] == "inferred_from_article"


def test_hierarchy_edges():
    G = ch.build_graph()
    assert G.has_edge("Microsoft", "LinkedIn") and G["Microsoft"]["LinkedIn"]["status"] == "reported_ownership"
    assert "unknown" in ch.lookup(G, "Nonexistent Co")[0]


def test_app_starts():
    import requests
    from app import build_ui
    assert MODEL_PATH.exists(), "run: python train_model.py"
    demo = build_ui()
    _, url, _ = demo.launch(prevent_thread_lock=True, quiet=True)
    try:
        assert requests.get(url, timeout=10).status_code == 200
    finally:
        demo.close()
