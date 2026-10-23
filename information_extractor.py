"""Rule-based corporate information extraction: spaCy NER (candidates) + regex patterns (relationships).
Covers COMMON sentence patterns only. Anything not supported by text is returned as 'Not found'."""
import re
from functools import lru_cache
from text_preprocessing import clean_text

NF = "Not found"
# Capitalised word sequence = candidate company/person name; '.' allowed only after Inc/Ltd/Corp/Co
_TOK = r"[A-Z][\w&-]*(?:(?<=Inc)\.|(?<=Ltd)\.|(?<=Corp)\.|(?<=Co)\.)?"
ENT = rf"{_TOK}(?:\s+{_TOK})*"
FILL = r"(?:\s+(?:has|have|had|will|agreed|announced|plans?|to|it|that|decided|said))*"
BUY = r"(?:acquire[sd]?|buy|buys|bought|purchase[sd]?)"
STAKE = r"(?:an?\s+[\d.]+%\s+stake\s+in\s+)?"
TITLES = {"CEO", "CFO", "CTO", "COO", "Chairman", "President"}

ACQ = [re.compile(rf"(?P<acquirer>{ENT}){FILL}\s+{BUY}\s+{STAKE}(?P<target>{ENT})"),
       re.compile(rf"(?P<acquirer>{ENT})'s\s+(?:acquisition|purchase)\s+of\s+(?P<target>{ENT})"),
       re.compile(rf"acquisition\s+of\s+(?P<target>{ENT})\s+by\s+(?P<acquirer>{ENT})")]
TAKEOVER = [re.compile(rf"(?P<target>{ENT})\s+(?:rejected|refused|declined|turned down)\s+.*?\b(?:offer|bid|proposal|approach)\s+(?:from|by)\s+(?P<acquirer>{ENT})"),
            re.compile(rf"(?P<acquirer>{ENT})\s+launched\s+.*?\b(?:offer|bid)\s+for\s+(?P<target>{ENT})"),
            re.compile(rf"(?P<acquirer>{ENT})\s+(?:withdrew|abandoned)\s+.*?\bfor\s+(?P<target>{ENT})"),
            re.compile(rf"(?P<acquirer>{ENT})\s+made\s+.*?(?:proposal|offer)\s+to\s+(?:buy|acquire)\s+(?P<target>{ENT})"),
            re.compile(rf"(?P<target>{ENT})\s+adopted\s+.*?\b(?:bid|offer|attempt)\s+(?:from|by)\s+(?P<acquirer>{ENT})")]
MERGER = re.compile(rf"(?P<a>{ENT})\s+and\s+(?P<b>{ENT})")
APPT = re.compile(rf"(?:appointed|named|hired|promoted|elected)\s+(?P<person>{ENT})\s+(?:as|to)\s+(?P<role>[A-Za-z ]+?)(?=\s+(?:effective|starting|from|on|succeeding|to)\b|[,.]|$)")
DEPART = re.compile(rf"(?P<role>chief\s+\w+\s+officer|CEO|CFO|CTO|COO|chairman|president)\s+(?P<person>{ENT})\s+(?:stepped down|resigned|retired|left|departed)")
REG = re.compile(r"\b(?P<reg>(?:The\s+)?(?:[A-Za-z]+\s+){0,2}?(?:regulator|Authority|authority|watchdog|Commission|commission|central bank|SEC|SEBI))"
                 r"\s+(?P<action>blocked|approved|cleared|fined|penali[sz]ed|imposed|opened|launched|ordered|banned)\b")
SELL = re.compile(r"\b(sell|selling|sale of|divest\w*|offload\w*|spin off)\b", re.I)
ASSET = re.compile(r"(?:sell|selling|divest|sale of|offload\w*|spin off)\s+(?:its\s+|an?\s+)?(.+?)(?=\s+to\s|\s+for\s|\s+on\s+\d|,|\.|$)", re.I)
RESTR = re.compile(r"restructur|reorgani[sz]|cost-cutting|layoff|lay off|bankruptcy|Chapter 11|cut\w*\s+[\d,]+\s+jobs", re.I)

NEG = re.compile(r"\b(reject\w*|refus\w*|declin\w*|turned down|withdr[ae]w\w*|abandon\w*|terminat\w*|blocked|no (?:deal|agreement)|would not)\b", re.I)
TALKS = re.compile(r"\btalks\b", re.I)
DONE = re.compile(r"\b(completed|closed|finali[sz]ed|has acquired|have acquired|acquired|has merged|sold)\b", re.I)
AGREED = re.compile(r"\b(agreed|signed|announced an?|announced the)\b", re.I)
PENDING = re.compile(r"\b(proposed|proposal|plans? to|will|expected to|subject to|unsolicited|intends?)\b", re.I)

CURRENCY = {"$": "USD", "US$": "USD", "USD": "USD", "dollars": "USD", "€": "EUR", "EUR": "EUR", "euros": "EUR",
            "£": "GBP", "GBP": "GBP", "₹": "INR", "INR": "INR", "Rs": "INR", "Rs.": "INR", "rupees": "INR"}
MONEY = [re.compile(r"(?P<cur>US\$|\$|€|£|₹|USD|EUR|GBP|INR|Rs\.?)\s?(?P<num>\d[\d,]*(?:\.\d+)?)(?!\d)(?:\s?(?P<mult>thousand|million|billion|trillion|crore|lakh))?(?!\s*per\s+share)"),
         re.compile(r"(?P<num>\d[\d,]*(?:\.\d+)?)\s*(?P<mult>thousand|million|billion|crore|lakh)\s+(?P<cur>rupees|dollars|euros)")]
MON = "Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
DATE = re.compile(rf"\b(\d{{1,2}}\s+(?:{MON})\s+\d{{4}}|(?:{MON})\s+\d{{1,2}},?\s+\d{{4}}|\d{{4}}-\d{{2}}-\d{{2}})\b")
PCT = re.compile(r"(\d+(?:\.\d+)?)\s?(?:%|per\s?cent)")
KEYWORD = {"Acquisition": r"acqui|buy|bought|purchase", "Merger": r"merg|combine", "Company Sale / Divestiture": r"sell|sale|divest|spin",
           "Hostile Takeover / Attempted Takeover": r"takeover|hostile|bid|offer|proposal", "Executive Change": r"appoint|named|resign|stepped|hired|retire",
           "Corporate Restructuring": RESTR.pattern, "Regulatory Action": r"regulator|authority|watchdog|fined|penalty|probe|investigation"}
DEAL_EVENTS = {"Acquisition", "Merger", "Company Sale / Divestiture", "Hostile Takeover / Attempted Takeover"}
SHOW = {"Acquisition": ["acquirer", "target", "deal_value", "currency", "stake_percent", "announcement_date", "status"],
        "Hostile Takeover / Attempted Takeover": ["acquirer", "target", "deal_value", "currency", "announcement_date", "status"],
        "Merger": ["companies", "deal_value", "currency", "announcement_date", "status"],
        "Company Sale / Divestiture": ["seller", "target", "acquirer", "deal_value", "currency", "stake_percent", "announcement_date", "status"],
        "Executive Change": ["companies", "persons", "role", "announcement_date", "status"],
        "Corporate Restructuring": ["companies", "deal_value", "currency", "announcement_date", "notes", "status"],
        "Regulatory Action": ["regulator", "regulatory_action", "companies", "deal_value", "currency", "announcement_date", "status"]}
LABEL = {"acquirer": "Acquirer (buyer)", "target": "Target (or asset sold)", "deal_value": "Deal value / amount", "stake_percent": "Stake %",
         "announcement_date": "Date mentioned", "regulatory_action": "Action", "companies": "Companies", "persons": "People"}


@lru_cache(maxsize=1)
def get_nlp():
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception:
        return None            # regex-only fallback; a warning is added to every result


def split_sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if s.strip()]


def _strip(name):
    name = re.sub(r"^(The|A|An)\s+", "", name.strip()).rstrip(".") if not re.search(r"(Inc|Ltd|Corp|Co)\.$", name) else name.strip()
    return name


def _lead_entity(s):
    """First name at sentence start, cut before job-title words (e.g. 'Helix Retail CEO Tom Walker' -> 'Helix Retail')."""
    m = re.match(ENT, s)
    if not m:
        return None
    words = []
    for w in _strip(m.group()).split():
        if w in TITLES:
            break
        words.append(w)
    return " ".join(words) or None


def find_money(s, deal_context=True):
    """Return (value_text, currency, n_found). Skips revenue/profit/savings/per-share figures for deal events."""
    found = []
    for pat in MONEY:
        for m in pat.finditer(s):
            if deal_context and re.search(r"(revenue|savings|profit|dividend)[^.]{0,15}$", s[:m.start()], re.I):
                continue
            found.append((m.start(), f"{m['num']} {m['mult'] or ''}".strip(), CURRENCY.get(m["cur"], m["cur"])))
    found.sort()
    return (found[0][1], found[0][2], len(found)) if found else (NF, NF, 0)


def status_of(s):
    if TALKS.search(s): return "Talks only; no agreement reached"
    if NEG.search(s): return "Not completed (rejected / withdrawn / blocked / no agreement)"
    if DONE.search(s): return "Reported completed (per article; not independently verified)"
    if AGREED.search(s): return "Agreement announced; completion not established"
    if PENDING.search(s): return "Proposed / pending; completion not established"
    return "Status unclear"


def _match(event, s):
    """Apply the event-specific rules to ONE sentence; return dict of fields or None."""
    if event in ("Acquisition", "Hostile Takeover / Attempted Takeover"):
        for p in (TAKEOVER if event.startswith("Hostile") else []) + ACQ:
            m = p.search(s)
            if m:
                return {"acquirer": _strip(m["acquirer"]), "target": _strip(m["target"]), "status": status_of(s)}
    elif event == "Merger" and re.search(r"merg|combin", s, re.I):
        m = MERGER.search(s)
        if m:
            return {"companies": [_strip(m["a"]), _strip(m["b"])], "status": status_of(s)}
    elif event == "Company Sale / Divestiture" and SELL.search(s):
        buyer, asset = re.search(rf"\bto\s+({ENT})", s), ASSET.search(s)
        return {"seller": _lead_entity(s) or NF, "acquirer": _strip(buyer.group(1)) if buyer else NF,
                "target": asset.group(1).strip() if asset else NF, "status": status_of(s)}
    elif event == "Executive Change":
        d, a = DEPART.search(s), APPT.search(s)
        if d or a:
            f = {"companies": [c for c in [_lead_entity(s)] if c], "persons": [], "status": ""}
            if d: f["persons"].append(d["person"]); f["role"] = d["role"]; f["status"] = "Departure reported"
            if a: f["persons"].append(a["person"]); f["role"] = f.get("role", a["role"]) if not d else f"{d['role']} (successor: {a['person']} as {a['role']})"; f["status"] = (f["status"] + "; " if d else "") + "Appointment reported"
            return f
    elif event == "Regulatory Action":
        m = REG.search(s)
        if m:
            act = m["action"].lower()
            f = {"regulator": _strip(m["reg"]), "regulatory_action": act, "companies": []}
            f["status"] = ("Blocked; transaction not completed" if act in ("blocked", "banned") else
                           "Approved (conditions may apply); completion not established" if act in ("approved", "cleared") else
                           "Penalty / order announced" if act in ("fined", "penalised", "penalized", "imposed", "ordered") else
                           "Investigation opened; no finding reported")
            pair = MERGER.search(s) if re.search(r"merg", s, re.I) else None
            deal = next((p.search(s) for p in ACQ if p.search(s)), None)
            tgt = re.search(rf"\b(?:fined|penali[sz]ed|on|into|against)\s+(?:the\s+)?({ENT})", s[m.start("action"):])
            f["companies"] = ([_strip(pair["a"]), _strip(pair["b"])] if pair else [_strip(deal["acquirer"]), _strip(deal["target"])] if deal
                              else [_strip(tgt.group(1))] if tgt else [])
            return f
    elif event == "Corporate Restructuring" and RESTR.search(s):
        notes = [f"Job cuts mentioned: {j.group(1)}" for j in [re.search(r"(\d[\d,]*)\s+(?:jobs|employees)", s)] if j]
        if re.search(r"Chapter 11|bankruptcy", s, re.I): notes.append("Bankruptcy filing mentioned")
        return {"companies": [c for c in [_lead_entity(s)] if c], "notes": "; ".join(notes) or NF,
                "status": "Bankruptcy filing reported" if notes and "Bankruptcy" in notes[-1] else "Action announced; execution not verified"}
    return None


def extract(text, event_type, url="", title=""):
    """Main entry. event_type normally comes from the classifier (a model output, not a fact)."""
    res = {"event_type": event_type, "companies": [], "persons": [], "acquirer": NF, "target": NF, "seller": NF, "regulator": NF,
           "regulatory_action": NF, "role": NF, "announcement_date": NF, "deal_value": NF, "currency": NF, "stake_percent": NF,
           "notes": NF, "status": NF, "evidence": NF, "source_url": url or NF, "warnings": []}
    text = clean_text(text)
    if not text:
        res["warnings"].append("Empty article text: nothing to extract."); return res
    sents, nlp = split_sentences(text), get_nlp()
    orgs, people = [], []
    if nlp:                                           # NER gives CANDIDATES only
        doc = nlp(text)
        orgs = [e.text for e in doc.ents if e.label_ == "ORG"]
        people = [e.text for e in doc.ents if e.label_ == "PERSON"]
    else:
        res["warnings"].append("spaCy model not installed: NER skipped (regex rules only).")
    ev_sent, fields = None, {}
    if event_type in SHOW:
        for s in sents:
            fields = _match(event_type, s) or {}
            if fields: ev_sent = s; break
        if not fields:
            res["warnings"].append("No supported sentence pattern matched; relationship fields left as 'Not found'.")
            ev_sent = next((s for s in sents if re.search(KEYWORD[event_type], s, re.I)), None)
    else:
        res["status"] = "No relevant corporate event identified"
    res.update({k: v for k, v in fields.items() if v})
    s = ev_sent or ""
    res["evidence"] = ev_sent or NF
    deal = event_type in DEAL_EVENTS
    if event_type in SHOW:                            # sentence first, whole article as fallback
        for scope in (s, text):
            val, cur, n = find_money(scope, deal)
            if val != NF: break
        res["deal_value"], res["currency"] = val, cur
        if n > 1: res["warnings"].append("Several amounts found; the first is shown.")
        res["announcement_date"] = next((m.group(1) for sc in (s, text) for m in [DATE.search(sc)] if m), NF)
        if deal:
            pm = PCT.search(s) or PCT.search(text)
            res["stake_percent"] = f"{pm.group(1)}%" if pm else NF
        if res["status"] == NF:
            res["status"] = status_of(s) if s else "Status unclear"
    rule_names = res["companies"] + [res[k] for k in ("acquirer", "seller") if res[k] != NF]
    if event_type != "Company Sale / Divestiture" and res["target"] != NF:
        rule_names.append(res["target"])                  # for divestitures 'target' is an asset, not a company
    people = [p for p in people if p not in rule_names]   # NER sometimes tags companies as PERSON
    seen, merged = set(), []
    for n in rule_names + orgs:                           # rule results first, NER candidates after
        n = re.sub(r"'s$", "", _strip(n))
        if (n and n.lower() not in seen and not re.search(r"\d", n) and n not in people
                and n not in res["persons"] and n != res["regulator"].replace("The ", "")):
            seen.add(n.lower()); merged.append(n)
    res["companies"] = merged
    res["persons"] = list(dict.fromkeys(res["persons"] + people))
    if event_type.startswith("Hostile") and res["status"] == "Status unclear":
        res["status"] = "Unsolicited / attempted bid; outcome not established"
    if event_type == "Acquisition" and res["status"].startswith("Not completed"):
        res["warnings"].append("Classified as Acquisition, but the text describes a rejected/blocked/withdrawn deal. Not an acquisition.")
    return res


def display_fields(res):
    """Ordered (label, value) pairs relevant to the event type, for the UI."""
    rows = []
    for k in SHOW.get(res["event_type"], ["companies"]):
        v = res[k]
        v = ", ".join(v) if isinstance(v, list) and v else (NF if isinstance(v, list) else v)
        rows.append((LABEL.get(k, k.replace("_", " ").capitalize()), v))
    rows += [("Evidence sentence", res["evidence"]), ("Source URL", res["source_url"])]
    return rows
