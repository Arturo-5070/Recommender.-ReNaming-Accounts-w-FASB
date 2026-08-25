# Account Label Recommender

A Streamlit proof of concept for standardizing ERP account descriptions into English labels using a compact public US GAAP / XBRL-style vocabulary, industry and chart-of-accounts context, historical auditor labels, and human review.

> This is a decision-support prototype. It is not an accounting conclusion engine, an audit opinion, or an official FASB interpretation. Any proposed label must be reviewed by the responsible accountant or auditor.

## Product behavior

The application accepts balances and optional journal-entry files as UTF-16 CSV files. The user can choose the separator or use automatic detection. Optional cleaning normalizes Unicode, removes control characters that are unsafe for tabular display, and collapses repeated spaces. The original business meaning is not translated or transliterated.

ERP exports may use any header names. Before validation, each file shows an editable mapping from the original headers to canonical fields. The first-pass suggestions use only cosine similarity between normalized header names; they do not inspect row values, balances, account descriptions, or other content. Each suggestion shows its score and ranked alternatives, and the user must confirm or change the mapping. After confirmation, the canonical balances fields are `account number`, `account type`, `balances`, `financial period`, and `account description`; the canonical journal-entry fields are `document number`, `posting date`, `account number`, `amount`, and `financial period`. An optional historical auditor-label file maps to `account number` and `auditor approved label`.

Unmapped ERP columns are preserved, so a user can select one as chart-of-accounts context after normalization. Duplicate source assignments or missing required mappings block recommendation generation.

The recommendation engine is intentionally deterministic and explainable. It computes lexical similarity against public synonym rules, then adds account-type, industry, chart-of-accounts, and historical-auditor-label signals. It also uses `data/synonym_library.json`, a reusable and versioned library organized by canonical label, account type, industry, context, source terms, synonyms, and priority. Connector words such as `for`, `the`, `a`, `as`, `on`, and `in` are excluded from token cosine calculations. Each loop can generate iterative synonym variants, preserve the change log, and compare the variants against the candidate canonical label.

It returns ranked alternatives for three views: ASC-oriented public terminology, context-aware terminology, and auditor-oriented terminology. The same engine is used for all three layers; only the weighting changes.

An optional LLM grammar stage can polish the top deterministic candidate. The LLM receives the source description, synonym candidate, canonical label, account type, industry, and chart context. It is constrained to return one label in strict JSON and is not allowed to invent accounting concepts. The output is revalidated against the source, synonym variant, and canonical label with deterministic cosine thresholds. A failed or unavailable LLM stage leaves the deterministic label in place and records the reason for review. The stage is disabled by default and uses the OpenAI-compatible environment variables `OPENAI_API_KEY` and `OPENAI_API_BASE` when enabled.

The reviewer can edit the final label, add a note, mark an account approved, and download the reviewed file as UTF-16 CSV. The application also produces heuristic review flags for high balance share, low confidence, unusual descriptions, and potential account-type conflicts. These flags prioritize review and must not be interpreted as formal audit materiality conclusions.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```

For Streamlit Community Cloud, connect the GitHub repository, set the main file to `app/app.py`, and use `requirements.txt` for dependencies. No API key is required by this prototype.

## Public reference basis

The public reference basis is the [FASB Accounting Standards Codification](https://asc.fasb.org/), the [FASB Standards page](https://www.fasb.org/standards), the [FASB GAAP Financial Reporting Taxonomy](https://www.fasb.org/page/detail?pageId=/projects/FASB-Taxonomies/2025-gaap-financial-reporting-taxonomy.html), and the [SEC Standard Taxonomies page](https://www.sec.gov/data-research/structured-data/taxonomies-schemas/standard-taxonomies). The application does not scrape or reproduce the text of the Codification.

## Next product increments

The most important next increment is the auditor-learning workflow. The confirmed column mappings and synonym-library additions should be saved per ERP and corporation so recurring files can be pre-populated safely while retaining a user confirmation step.

The auditor-learning workflow itself should save approved edits as labeled examples keyed by account number, corporation, ERP, industry, account type, and source description. A later version can use those examples for nearest-neighbor retrieval and confidence calibration, while preserving an audit trail of who changed each label and when. A second increment can add a user-maintained taxonomy file so that the accounting team can approve, retire, or version synonym rules without changing source code.
