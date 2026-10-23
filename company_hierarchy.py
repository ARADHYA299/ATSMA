"""Directed graph (NetworkX): parent -> child. Edge status keeps ownership evidence levels apart."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.lines import Line2D
from information_extractor import NF

ROOT = Path(__file__).resolve().parent
# status -> (colour, line style). Only 'confirmed_ownership' should rest on a filing.
STYLE = {"confirmed_ownership": ("#2ca02c", "solid"), "completed_acquisition": ("#1f77b4", "solid"),
         "reported_ownership": ("#7f7f7f", "solid"), "proposed_acquisition": ("#ff7f0e", "dashed")}


def infer_edges(res, article_id):
    """Turn ONE extraction into a graph edge, only for acquisitions with both parties found.
    Rejected / blocked / withdrawn deals create NO edge. 'Completed' needs completion wording in the article."""
    if res["event_type"] != "Acquisition" or NF in (res["acquirer"], res["target"]):
        return []
    st = res["status"]
    if st.startswith(("Not completed", "Talks")):
        return []
    status = "completed_acquisition" if st.startswith("Reported completed") else "proposed_acquisition"
    return [{"parent": res["acquirer"], "child": res["target"], "status": status,
             "verification": "inferred_from_article", "evidence": f"{article_id}: {res['evidence']}"}]


def build_graph(csv_path=ROOT / "data" / "company_relationships.csv", inferred=()):
    G = nx.DiGraph()
    if Path(csv_path).exists():
        for r in pd.read_csv(csv_path).fillna("").itertuples():
            G.add_edge(r.parent, r.child, status=r.status, verification=r.verification, evidence=r.evidence)
    for e in inferred:
        if not G.has_edge(e["parent"], e["child"]):          # curated data is never overwritten by inference
            G.add_edge(e["parent"], e["child"], **{k: e[k] for k in ("status", "verification", "evidence")})
    return G


def _line(G, a, b):
    d = G[a][b]
    return f"{d['status'].replace('_', ' ')} ({d['verification'].replace('_', ' ')}) | {d['evidence']}"


def lookup(G, query):
    """Return (markdown report, matplotlib figure or None)."""
    q = (query or "").strip().lower()
    if not q:
        return "Enter a company name.", None
    hits = sorted([n for n in G if q in n.lower()], key=len)
    if not hits:
        return f"No relationship for **{query}** in the local data. Relationship status: **unknown** (not the same as 'none').", None
    n = hits[0]
    md = [f"### {n}"]
    parents, kids = list(G.predecessors(n)), list(G.successors(n))
    md.append("**Known parent(s):** " + ("none recorded" if not parents else ""))
    md += [f"- {p}: {_line(G, p, n)}" for p in parents]
    md.append("**Known subsidiaries / acquisition targets:** " + ("none recorded" if not kids else ""))
    md += [f"- {k}: {_line(G, n, k)}" for k in kids]
    md.append("\n*Only `confirmed_ownership` rests on a filing. Inferred and proposed links come from news text and are not proof of ownership.*")
    return "\n".join(md), draw_graph(G, n)


def draw_graph(G, node):
    sub = nx.ego_graph(G, node, radius=2, undirected=True)
    pos = nx.spring_layout(sub, seed=3, k=1.4)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    nx.draw_networkx_nodes(sub, pos, ax=ax, node_color=["#ffd54f" if n == node else "#e3f2fd" for n in sub],
                           node_size=1800, edgecolors="#555")
    nx.draw_networkx_labels(sub, pos, ax=ax, font_size=7)
    for status, (col, style) in STYLE.items():
        edges = [(a, b) for a, b, d in sub.edges(data=True) if d["status"] == status]
        nx.draw_networkx_edges(sub, pos, edgelist=edges, ax=ax, edge_color=col, style=style, arrows=True,
                               arrowsize=18, node_size=1800, width=1.8)
    ax.legend(handles=[Line2D([0], [0], color=c, ls=s, label=k.replace("_", " ")) for k, (c, s) in STYLE.items()],
              fontsize=7, loc="lower left")
    ax.set_title("Arrow: parent -> subsidiary / target", fontsize=9); ax.axis("off"); fig.tight_layout()
    return fig
