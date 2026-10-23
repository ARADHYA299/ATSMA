"""Generate synthetic TRAINING data from templates, train, evaluate on the hand-written set.
Run from anywhere:  python train_model.py"""
import random
import pandas as pd
from text_preprocessing import ROOT, load_articles
from classifier import train

STEMS = "Juniper Kinetic Evergreen Sterling Cascade Fulcrum Meridian Aurora Tundra Basalt Cedar Ember Fjord Glacier Horizon Indigo Jasper Keystone Lantern Mosaic Nova Onyx Prairie Quartz Ripple Sable Tidal Vortex Yarrow Zephyr".split()
SUFFIX = "Corp Ltd Holdings Group Industries Systems Partners Bank Energy Foods Pharma Retail Motors Telecom Capital Media".split()
FIRST = "Anita Rohan Maria Daniel Fatima Lucas Sofia Kevin Neha Omar Laura Victor".split()
LAST = "Sharma Okafor Lindqvist Moreau Banerjee Castillo Nakamura Whitfield Petrov Haddad".split()
MONTHS = "January February March April May June July August September October November December".split()
UNITS = "logistics consumer mining software retail chemicals media energy".split()
ROLES = "chief executive officer|chief financial officer|chairman|president".split("|")
T = {
"Acquisition": ["{A} agreed to acquire {B} for {V}, the company said on {D}.", "{A} has completed its acquisition of {B}, a deal valued at {V}.",
  "{A} will buy {B} in a cash transaction worth {V}, subject to regulatory approval.", "{A} acquired {B} on {D}, expanding its presence in the market.",
  "{A} announced it will purchase a {S}% stake in {B} for {V}.", "The acquisition of {B} by {A} was completed on {D} for {V}."],
"Merger": ["{A} and {B} announced a merger on {D}, creating a combined company.", "{A} and {B} agreed to merge in an all-stock deal valued at {V}.",
  "The boards of {A} and {B} approved the proposed merger, which is expected to close next year.", "{A} and {B} are in talks to merge, people familiar with the matter said.",
  "Shareholders of {A} and {B} approved the merger on {D}.", "The merger between {A} and {B} was completed on {D}."],
"Company Sale / Divestiture": ["{A} agreed to sell its {U} unit to {B} for {V}.", "{A} announced plans to divest its {U} division on {D}.",
  "{A} completed the sale of its {U} business to {B} for {V}.", "{A} will spin off its {U} arm to focus on its core operations.",
  "{A} agreed to sell a {S}% stake in its {U} subsidiary to {B}.", "{A} offloaded its {U} unit for {V}, it said on {D}."],
"Hostile Takeover / Attempted Takeover": ["{A} rejected an unsolicited {V} takeover offer from {B}, calling it inadequate.", "{B} launched a hostile bid for {A} on {D}, going directly to shareholders.",
  "{A} adopted a poison pill to block an unwanted takeover attempt by {B}.", "{B} withdrew its takeover proposal for {A} after the board refused to negotiate.",
  "The board of {A} turned down an unsolicited acquisition proposal from {B}.", "{B} abandoned its hostile approach for {A} on {D}."],
"Executive Change": ["{A} appointed {P} as {R}, effective {D}.", "{A} said {R} {P} resigned on {D}.", "{A} named {P} as {R} on {D}.",
  "{A} CEO {P} stepped down on {D}, and the board named {Q} as interim CEO.", "{A} hired {P} as {R}.", "{A} has promoted {P} to president, replacing {Q} who left the firm."],
"Corporate Restructuring": ["{A} announced a restructuring plan that will cut {N} jobs and close two plants.", "{A} filed for Chapter 11 bankruptcy protection on {D} to restructure its debt of {V}.",
  "{A} will reorganize into three business divisions to simplify its operations.", "{A} unveiled a cost-cutting programme targeting savings of {V}, including layoffs.",
  "{A} agreed with creditors to restructure {V} of debt and extend maturities.", "{A} said it will lay off {N} employees as part of a turnaround strategy."],
"Regulatory Action": ["The competition regulator blocked the proposed merger between {A} and {B} on {D}.", "The securities regulator fined {A} {V} for misleading investors.",
  "The antitrust authority approved the acquisition of {B} by {A} on condition of divestments.", "The central bank imposed a penalty of {V} on {A} for violating norms.",
  "The competition watchdog opened an investigation into {A} over alleged abuse of dominance.", "Regulators ordered {A} to pay {V} in fines after an audit on {D}."],
"Other / No Relevant Corporate Event": ["{A} reported a {S}% rise in quarterly profit to {V}, beating analyst estimates.", "Shares of {A} rose {S}% on {D} after a strong earnings report.",
  "Oil prices fell {S}% on {D} as demand concerns weighed on markets.", "{A} launched a new product line on {D} priced from {V}.",
  "{A} declared a quarterly dividend of 0.{S} dollars per share.", "Analysts expect {A} to post higher sales next quarter, citing strong demand."],
}


def build_training_set(per_class=24, seed=7):
    rnd, rows = random.Random(seed), []
    name = lambda: f"{rnd.choice(STEMS)} {rnd.choice(SUFFIX)}"
    person = lambda: f"{rnd.choice(FIRST)} {rnd.choice(LAST)}"
    for label, temps in T.items():
        for i in range(per_class):
            sym, mult = rnd.choice([("$", "million"), ("$", "billion"), ("EUR ", "million"), ("GBP ", "billion"), ("INR ", "crore")])
            text = temps[i % len(temps)].format(
                A=name(), B=name(), P=person(), Q=person(), R=rnd.choice(ROLES), U=rnd.choice(UNITS), S=rnd.randint(5, 80),
                N=f"{rnd.randint(2, 9) * 500:,}", V=f"{sym}{rnd.randint(2, 900)} {mult}",
                D=f"{rnd.randint(1, 28)} {rnd.choice(MONTHS)} {rnd.randint(2021, 2024)}")
            rows.append({"article_id": f"T{len(rows)+1:03d}", "title": "", "source": "synthetic_template", "url": "",
                         "publication_date": "", "text": text, "collection_status": "generated", "label": label,
                         "data_origin": "synthetic_template"})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    build_training_set().to_csv(ROOT / "data" / "train_synthetic.csv", index=False)   # saved for transparency
    tr, te = load_articles(ROOT / "data" / "train_synthetic.csv"), load_articles()
    assert not set(tr["model_input"]) & set(te["model_input"]), "train/test leakage"
    m = train(tr, te); h = m["heldout"]
    print(m["data_note"])
    print(f"train={m['n_train']} (templates) | test={m['n_test']} (hand-written) | acc={h['accuracy']:.2f} "
          f"macroP={h['macro_precision']:.2f} macroR={h['macro_recall']:.2f} macroF1={h['macro_f1']:.2f}")
