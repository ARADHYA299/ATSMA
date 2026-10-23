# Project Report: Financial News Classification, Corporate Event Extraction and Company Hierarchy Mapping

**Abstract.** The system classifies financial news into eight corporate-event categories (TF-IDF + Logistic Regression), extracts structured fields (spaCy NER + regex rules) with evidence sentences, and links companies in a NetworkX graph that separates confirmed, reported, proposed and completed relationships. A Gradio interface exposes analysis, dataset browsing, hierarchy lookup and evaluation. Data is synthetic; results are indicative only.

**Problem statement.** Corporate events are buried in unstructured news. Readers need the event type, parties, value and status, and must not mistake a rejected or proposed deal for a completed one.

**Objectives.** (1) Load/collect and clean news. (2) Classify events. (3) Extract companies, people, value, currency, stake, date, status, evidence. (4) Map parent/subsidiary links with evidence levels. (5) Provide a simple UI. (6) Evaluate honestly.

**Existing-system limitations.** Keyword alerts ignore context and negation; manual reading does not scale; price-feed terminals do not expose relationship status or evidence; many tools treat any "acquisition" headline as ownership.

**Proposed system.** Pipeline: clean -> classify -> event-specific extraction -> status/negation check -> graph edge (only when supported) -> UI. Every output shows evidence and a "model output, not verified fact" notice; missing fields read "Not found".

**Architecture.** `text_preprocessing` -> `classifier` and `information_extractor` -> `company_hierarchy` -> `app`. `news_collector` is optional input.

**Methodology.** *Preprocessing:* strip HTML, normalise whitespace, keep case/numbers/negations, title+body as input, drop exact duplicate texts/URLs. *Classifier:* TF-IDF (unigrams+bigrams, sublinear tf) -> Logistic Regression (C=10, balanced class weights); uncertainty flag when top score < 0.40 or margin to second class < 0.10 (heuristic thresholds); key n-grams shown via coefficient x TF-IDF. *Extraction:* spaCy ORG/PERSON as candidates; capitalised-name regex patterns assign roles (acquirer/target, seller/buyer, regulator, appointee); regexes for money (USD/EUR/GBP/INR, million/billion/crore/lakh), dates, percentages; status from negation/talks/completed/agreed/pending cue words. *Graph:* DiGraph parent->child with status, verification, evidence.

**Dataset.** Training: 192 template sentences (24/class, synthetic). Test: 40 hand-written synthetic articles (5/class), disjoint names, assertion of no text overlap. Includes rejected bids, talks-only mergers, non-event news. Tiny and synthetic: cannot establish real-world performance.

**Results (measured).** Held-out n=40: accuracy 0.95, macro-P 0.96, macro-R 0.95, macro-F1 0.95. Two errors (S003, S037). Tests: 13 pytest tests passed in this environment. Confusion matrix: Evaluation tab / `outputs/metrics.json`.

**Sample.** Input: "Alpha Corp agreed to acquire Beta Ltd for $500 million on 12 March 2024." Output: Acquisition (score 38%, flagged low-confidence); acquirer Alpha Corp; target Beta Ltd; value 500 million USD; date 12 March 2024; status "Agreement announced; completion not established"; graph edge `proposed_acquisition`, inferred_from_article. Input "Falcon Industries rejected an unsolicited $6 billion takeover offer from Raptor Capital" -> status "Not completed", no ownership edge.

**Demo checklist.** Train model; open app; paste both samples; open News Dataset and pick S016; search "Microsoft" and "Alpha Corp" in Hierarchy; show Evaluation tab and point out the synthetic-data warning.

**Limitations.** See README; additionally single-label design, rule coverage, NER noise on fictional names, possible optimistic scores.
**Future work.** Real labelled corpus (e.g. SEC 8-K/EDGAR), multi-label classification, dependency-parse rules or fine-tuned small NER, probability calibration, filing-based verification, entity resolution/aliases.
**Conclusion.** A small, transparent pipeline that works offline and avoids overstating certainty.
**References.** scikit-learn, spaCy (en_core_web_sm), NetworkX, Gradio, pandas, Beautiful Soup, feedparser documentation; company 10-K filings (Exhibit 21) for verifying relationships.

## Viva Q&A
- **TF-IDF?** Weights a term by frequency in the document times rarity across documents, so common words count less. Sublinear tf dampens repeats; bigrams capture phrases like "to acquire".
- **Logistic Regression?** Linear model; softmax over class scores gives class probabilities; trained by minimising cross-entropy with L2 regularisation. Coefficients are interpretable.
- **Supervised learning?** Learning a mapping from inputs to known labels, then testing on unseen labelled data.
- **Precision/recall/F1?** Precision = TP/(TP+FP); recall = TP/(TP+FN); F1 = harmonic mean. Macro averages classes equally.
- **NER?** Locating and typing names (ORG, PERSON) in text; here only candidates, then rules decide roles.
- **Regex?** Pattern language used for money, dates, percentages and sentence templates; precise but brittle.
- **Why NetworkX?** Lightweight in-memory directed graph with traversal and drawing; no database needed.
- **Classification vs extraction?** Classification labels the whole article; extraction pulls structured fields (who, what, how much) from specific sentences.
- **Dataset limitations?** Small, synthetic, one author, fictional names: scores are optimistic and not transferable.
- **False positive / negative?** FP: predicting "Acquisition" for a rejected bid. FN: missing a real acquisition. Which is worse depends on use.
- **Why distinguish proposed vs completed?** Agreements can fail, be blocked or rejected; treating them as ownership produces false facts. Only filings justify "confirmed".
