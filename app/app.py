from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from column_mapping import TARGET_COLUMNS, default_mapping, mapping_status, rename_using_mapping, suggest_mappings
from io_utils import detect_separators, make_balance_view, read_csv_utf16
from recommender import materiality_and_quality_flags, recommend_dataframe
from synonym_engine import get_engine


st.set_page_config(page_title="Account Label Recommender", page_icon="A", layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 2rem; padding-bottom: 3rem; }
[data-testid="stMetricValue"] { font-size: 1.55rem; }
.mapping-card { border-left: 4px solid #2f6fed; padding: .4rem .8rem; margin: .2rem 0 .8rem 0; background: #f6f8fb; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def cached_read(file_bytes: bytes, separator: str, clean: bool, collapse_spaces: bool, keep_printable: bool) -> pd.DataFrame:
    return read_csv_utf16(io.BytesIO(file_bytes), separator, clean, collapse_spaces, keep_printable)


def read_uploaded(uploaded_file, separator: str, clean: bool, collapse_spaces: bool, keep_printable: bool) -> pd.DataFrame:
    return cached_read(uploaded_file.getvalue(), separator, clean, collapse_spaces, keep_printable)


def csv_utf16_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-16")


def separator_from_choice(choice: str, uploaded_file) -> str | None:
    if choice == "Auto-detect":
        if uploaded_file is None:
            return None
        counts = detect_separators(uploaded_file)
        return counts[0][0] if counts and counts[0][1] > 0 else ","
    return {"Comma (,)": ",", "Semicolon (;)": ";", "Tab": "\t", "Pipe (|)": "|"}.get(choice, ",")


def format_similarity(value: object) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "0.0%"


def mapping_editor(raw_df: pd.DataFrame, file_type: str, key_prefix: str) -> tuple[dict[str, str | None], dict[str, float], bool]:
    """Show cosine suggestions and editable selects; return mapping, scores, validity."""
    source_columns = list(raw_df.columns)
    suggestions = suggest_mappings(source_columns, file_type, alternatives=max(3, len(source_columns)))
    seed = default_mapping(source_columns, file_type)
    mapping: dict[str, str | None] = {}
    scores: dict[str, float] = {}
    choices = ["Do not map"] + source_columns

    st.markdown("**Column mapping**")
    st.caption("Suggestions use only the source and target column names. They do not inspect data values, balances, descriptions, or row content. Review every field before continuing.")
    header_a, header_b, header_c = st.columns([2.3, 2.8, 1.0])
    header_a.markdown("**Canonical field**")
    header_b.markdown("**Source column selected**")
    header_c.markdown("**Cosine score**")
    for target in TARGET_COLUMNS[file_type]:
        ranked = suggestions[target]
        by_source = {str(item["source_column"]): float(item["cosine_similarity"]) for item in ranked}
        selected = seed.get(target) or "Do not map"
        selected_index = choices.index(selected) if selected in choices else 0
        col_a, col_b, col_c = st.columns([2.3, 2.8, 1.0])
        col_a.write(f"`{target}`")
        selected_value = col_b.selectbox(target, choices, index=selected_index, key=f"{key_prefix}_{target}", label_visibility="collapsed")
        mapping[target] = None if selected_value == "Do not map" else selected_value
        score = by_source.get(selected_value, 0.0)
        scores[target] = score
        col_c.write(format_similarity(score))
        alternatives = ", ".join(f"{item['source_column']} ({format_similarity(item['cosine_similarity'])})" for item in ranked[:3])
        st.caption(f"Name-only alternatives: {alternatives or 'none'}")

    status = mapping_status(mapping, file_type)
    if status["valid"]:
        st.success(f"{file_type} mapping is complete and has no duplicate source assignments.")
    else:
        if status["missing_targets"]:
            st.error("Missing canonical fields: " + ", ".join(status["missing_targets"]))
        if status["duplicate_sources"]:
            st.error("The same source column is assigned more than once: " + ", ".join(status["duplicate_sources"]))
    return mapping, scores, bool(status["valid"])


def store_file_state(prefix: str, uploaded_file, raw_df: pd.DataFrame, mapping: dict[str, str | None], scores: dict[str, float], valid: bool) -> None:
    st.session_state[f"{prefix}_raw"] = raw_df
    st.session_state[f"{prefix}_mapping"] = mapping
    st.session_state[f"{prefix}_mapping_scores"] = scores
    st.session_state[f"{prefix}_mapping_valid"] = valid
    st.session_state[f"{prefix}_file_name"] = uploaded_file.name if uploaded_file else ""


if "results" not in st.session_state:
    st.session_state.results = None

st.title("Account Label Recommender")
st.caption("Prototype for standardizing ERP account descriptions using public US GAAP / XBRL-style terminology and human review.")
st.warning("This prototype is an analytical aid. It does not provide an accounting conclusion, audit opinion, or official FASB interpretation. FASB ASC content is not copied into the application; the vocabulary is a compact, public, taxonomy-style reference that should be validated by the auditor.")

with st.sidebar:
    st.header("Processing controls")
    st.write("Upload UTF-16 CSV files. Each ERP file can use different header names; the app suggests a mapping using header-name cosine similarity only.")
    separator_choice = st.selectbox("CSV separator", ["Auto-detect", "Comma (,)", "Semicolon (;)", "Tab", "Pipe (|)"])
    clean_values = st.checkbox("Enable cleaning procedures", value=True)
    keep_printable = st.checkbox("Keep only valid printable UTF-16 characters", value=True, disabled=not clean_values)
    collapse_spaces = st.checkbox("Collapse multiple spaces", value=True, disabled=not clean_values)
    st.divider()
    industry = st.text_input("Industry or business context", placeholder="e.g., SaaS software, retail, manufacturing")
    materiality_ratio = st.number_input("Materiality review threshold (% of absolute file balance)", min_value=0.1, max_value=100.0, value=1.0, step=0.1) / 100
    top_n = st.slider("Ranked alternatives per layer", min_value=1, max_value=5, value=3)
    st.caption(f"Synonym library version: {get_engine().version}")
    llm_refine = st.checkbox("Enable optional LLM grammar refinement", value=False, help="The LLM only polishes the top deterministic candidate. The result is revalidated by cosine similarity and rejected when semantic drift is detected.")
    llm_model = st.selectbox("LLM grammar model", ["gpt-5-mini", "gpt-5-nano"], disabled=not llm_refine)
    if llm_refine:
        st.caption("The grammar stage requires OPENAI_API_KEY and OPENAI_API_BASE. It is disabled safely when credentials are unavailable.")
    st.divider()
    st.markdown("**Public reference basis**")
    st.markdown("FASB ASC and the FASB GAAP Financial Reporting Taxonomy are conceptual references. The prototype uses compact public labels and synonym rules rather than reproducing restricted Codification text.")

st.subheader("1. Upload source files")
file_col_a, file_col_b = st.columns(2)
with file_col_a:
    balances_file = st.file_uploader("Balances CSV (required)", type=["csv", "txt"], key="balances_upload")
    journal_file = st.file_uploader("Journal entries CSV (optional)", type=["csv", "txt"], key="journal_upload")
with file_col_b:
    auditor_file = st.file_uploader("Historical auditor labels CSV (optional)", type=["csv", "txt"], key="auditor_upload")
    st.markdown("The initial canonical balances fields are `account number`, `account type`, `balances`, `financial period`, and `account description`.")
    st.markdown("The initial canonical journal fields are `document number`, `posting date`, `account number`, `amount`, and `financial period`.")

active_separator = separator_from_choice(separator_choice, balances_file or journal_file or auditor_file)
if active_separator is not None:
    st.caption(f"Active separator: `{('TAB' if active_separator == '\\t' else active_separator)}`")

for prefix, uploaded_file, file_type in [("balances", balances_file, "Balances"), ("journal", journal_file, "Journal entries"), ("auditor", auditor_file, "Auditor labels")]:
    if uploaded_file is None or active_separator is None:
        continue
    try:
        raw = read_uploaded(uploaded_file, active_separator, clean_values, collapse_spaces, keep_printable)
        st.session_state[f"{prefix}_raw"] = raw
        with st.expander(f"{file_type}: map ERP columns", expanded=(prefix == "balances")):
            mapping, scores, valid = mapping_editor(raw, file_type, f"{prefix}_map")
            store_file_state(prefix, uploaded_file, raw, mapping, scores, valid)
            st.caption(f"Loaded `{uploaded_file.name}` with {len(raw):,} rows and {len(raw.columns):,} original columns.")
            st.dataframe(raw.head(5), use_container_width=True, hide_index=True)
    except UnicodeDecodeError:
        st.error(f"{file_type} could not be decoded as UTF-16. Export the file explicitly as UTF-16 and upload it again.")
    except Exception as error:
        st.error(f"Could not read {file_type}: {error}")

st.subheader("2. Confirm normalized data and optional context")
normalized_balances = None
normalized_journal = None
normalized_auditor = None
if isinstance(st.session_state.get("balances_raw"), pd.DataFrame) and st.session_state.get("balances_mapping_valid"):
    normalized_balances = rename_using_mapping(st.session_state.balances_raw, st.session_state.balances_mapping)
if isinstance(st.session_state.get("journal_raw"), pd.DataFrame) and st.session_state.get("journal_mapping_valid"):
    normalized_journal = rename_using_mapping(st.session_state.journal_raw, st.session_state.journal_mapping)
if isinstance(st.session_state.get("auditor_raw"), pd.DataFrame) and st.session_state.get("auditor_mapping_valid"):
    normalized_auditor = rename_using_mapping(st.session_state.auditor_raw, st.session_state.auditor_mapping)

if normalized_balances is not None:
    available_columns = list(normalized_balances.columns)
    context_options = ["None"] + available_columns
    chart_context_column = st.selectbox("Optional chart-of-accounts context column", context_options, key="chart_context_column")
    chart_context_column = "" if chart_context_column == "None" else chart_context_column
    st.dataframe(normalized_balances.head(10), use_container_width=True, hide_index=True)
else:
    chart_context_column = ""
    st.info("Upload balances and complete its column mapping to continue.")

can_generate = normalized_balances is not None and bool(st.session_state.get("balances_mapping_valid"))
if st.button("Generate ranked recommendations", type="primary", disabled=not can_generate):
    source = normalized_balances.copy()
    auditor_column = ""
    if normalized_auditor is not None:
        auditor_label_candidates = ["auditor approved label", "approved label", "auditor label"]
        auditor_column = next((column for column in auditor_label_candidates if column in normalized_auditor.columns), "")
        if auditor_column and "account number" in normalized_auditor.columns and "account number" in source.columns:
            history = normalized_auditor[["account number", auditor_column]].drop_duplicates("account number")
            if auditor_column != "auditor approved label":
                history = history.rename(columns={auditor_column: "auditor approved label"})
                auditor_column = "auditor approved label"
            source = source.merge(history, how="left", on="account number")
    if not auditor_column and "auditor approved label" in source.columns:
        auditor_column = "auditor approved label"
    with st.spinner("Scoring account descriptions against the public synonym and context reference..."):
        generated = recommend_dataframe(source, industry=industry, chart_column=chart_context_column, auditor_column=auditor_column, top_n=top_n, llm_refine=llm_refine, llm_model=llm_model)
        generated = materiality_and_quality_flags(generated, materiality_ratio=materiality_ratio)
        if normalized_journal is not None:
            generated = make_balance_view(generated, normalized_journal)
        st.session_state.results = generated
    st.success(f"Recommendations generated for {len(generated):,} accounts.")

results = st.session_state.get("results")
if isinstance(results, pd.DataFrame) and not results.empty:
    st.subheader("3. Review recommendations")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Accounts", f"{len(results):,}")
    metric_cols[1].metric("Review flags", f"{int(results['review_required'].sum()):,}")
    metric_cols[2].metric("Materiality flags", f"{int(results['materiality_flag'].sum()):,}")
    historical_count = int(results["auditor approved label"].notna().sum()) if "auditor approved label" in results.columns else 0
    metric_cols[3].metric("Historical auditor labels", f"{historical_count:,}")

    tabs = st.tabs(["Summary", "ASC terminology", "Context-aware", "Auditor-oriented", "Review and export"])
    key_columns = [column for column in ["account number", "account description", "account type", "balances", "financial period"] if column in results.columns]
    with tabs[0]:
        summary_columns = key_columns + ["asc_top_label", "context_top_label", "auditor_top_label", "review_reason"]
        summary_columns = [column for column in summary_columns if column in results.columns]
        st.dataframe(results[summary_columns], use_container_width=True, hide_index=True)
        st.caption("Review flags are heuristic indicators for prioritization, not audit materiality determinations.")
        flag_counts = pd.DataFrame({"Flag": ["Materiality", "Review required", "Unusual account", "Classification inconsistency"], "Accounts": [int(results["materiality_flag"].sum()), int(results["review_required"].sum()), int(results["unusual_account_flag"].sum()), int(results["inconsistency_flag"].sum())]})
        st.bar_chart(flag_counts.set_index("Flag"))

    for tab, layer, title in [(tabs[1], "asc", "FASB ASC-oriented public terminology"), (tabs[2], "context", "Industry and chart-of-accounts context"), (tabs[3], "auditor", "Historical auditor-label-oriented")]:
        with tab:
            st.markdown(f"**{title}**")
            layer_columns = key_columns.copy()
            layer_columns += [f"{layer}_label_{rank}" for rank in range(1, top_n + 1)]
            layer_columns += [f"{layer}_confidence_{rank}" for rank in range(1, top_n + 1)]
            layer_columns += [f"{layer}_synonym_variant_{rank}" for rank in range(1, top_n + 1)]
            layer_columns += [f"{layer}_synonym_cosine_{rank}" for rank in range(1, top_n + 1)]
            layer_columns += [f"{layer}_synonym_change_log_{rank}" for rank in range(1, top_n + 1)]
            layer_columns += [f"{layer}_evidence_{rank}" for rank in range(1, top_n + 1)]
            layer_columns += [f"{layer}_llm_refined_label", f"{layer}_llm_refinement_status", f"{layer}_llm_revalidation"]
            layer_columns = [column for column in layer_columns if column in results.columns]
            st.dataframe(results[layer_columns], use_container_width=True, hide_index=True)
            st.caption("All three views use one deterministic engine; only the weighting of lexical, type, industry, chart, and historical-label signals changes.")

    with tabs[4]:
        st.markdown("Edit the final label, add a reviewer note, and mark the account as approved. The edited value is retained in the export.")
        review_columns = key_columns + ["asc_top_label", "context_top_label", "auditor_top_label", "review_reason"]
        review_columns = [column for column in review_columns if column in results.columns]
        review_df = results[review_columns].copy()
        review_df["final recommended label"] = results["auditor_top_label"].where(results["auditor_top_label"].astype(str).str.len() > 0, results["context_top_label"])
        review_df["reviewer note"] = ""
        review_df["approved"] = False
        edited = st.data_editor(review_df, use_container_width=True, hide_index=True, num_rows="fixed", column_config={"approved": st.column_config.CheckboxColumn("Approved", default=False), "final recommended label": st.column_config.TextColumn("Final recommended label", required=True), "reviewer note": st.column_config.TextColumn("Reviewer note")})
        export_df = results.copy()
        export_df["final recommended label"] = edited["final recommended label"].values
        export_df["reviewer note"] = edited["reviewer note"].values
        export_df["approved"] = edited["approved"].values
        st.download_button("Download reviewed recommendations (UTF-16 CSV)", data=csv_utf16_bytes(export_df), file_name="account_recommendations_reviewed_utf16.csv", mime="text/csv")
        st.download_button("Download cleaned balances (UTF-16 CSV)", data=csv_utf16_bytes(normalized_balances), file_name="cleaned_balances_utf16.csv", mime="text/csv")

st.divider()
st.markdown("**Reference sources:** [FASB Accounting Standards Codification](https://asc.fasb.org/), [FASB Standards](https://www.fasb.org/standards), [FASB GAAP Financial Reporting Taxonomy](https://www.fasb.org/page/detail?pageId=/projects/FASB-Taxonomies/2025-gaap-financial-reporting-taxonomy.html), and [SEC Standard Taxonomies](https://www.sec.gov/data-research/structured-data/taxonomies-schemas/standard-taxonomies).")
