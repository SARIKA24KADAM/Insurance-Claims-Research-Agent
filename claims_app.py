from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from claims_agent import CLAIM_COLUMNS, ClaimsResearchAgent, make_demo_claims, normalize_claims
from evaluate_claims_agent import evaluate_retrieval


st.set_page_config(page_title="Claims Research Agent", page_icon="ðŸ”Ž", layout="wide")

st.title("Claims Research Agent")
st.caption("Search synthetic policy guidance, inspect claim records, and compare similar claims with cited research leads.")
st.info("Research support only. The agent does not determine coverage, liability, fraud, or settlement. Verify every finding with an authorized claims professional.")


def guess_column(columns: list[str], target: str) -> str:
    aliases = {
        "claim_id": ("claim_id", "claim number", "claim_no", "id"),
        "policy_id": ("policy_id", "policy number", "policy_no"),
        "line_of_business": ("line_of_business", "lob", "line", "product"),
        "claim_type": ("claim_type", "loss_type", "peril", "cause"),
        "status": ("status", "claim_status"),
        "incurred_loss": ("incurred_loss", "incurred", "loss", "paid"),
        "reserve": ("reserve", "case_reserve", "outstanding"),
        "days_open": ("days_open", "open_days", "duration"),
        "territory": ("territory", "region", "location"),
        "description": ("description", "loss_description", "notes", "details"),
    }
    for col in columns:
        normalized = col.lower().replace(" ", "_")
        if any(alias.replace(" ", "_") in normalized for alias in aliases[target]):
            return col
    return columns[0]


@st.cache_data(show_spinner=False)
def get_demo_claims() -> pd.DataFrame:
    return make_demo_claims()


with st.sidebar:
    st.header("1 Â· Choose claim data")
    source = st.radio("Data source", ["Synthetic demo portfolio", "Upload a CSV"])
    upload = st.file_uploader("Claim records (CSV)", type=["csv"], disabled=source != "Upload a CSV")

    raw: pd.DataFrame | None = None
    if source == "Synthetic demo portfolio":
        claims = get_demo_claims()
    elif upload is None:
        claims = None
        st.caption("Upload a CSV to map the claim fields.")
    else:
        try:
            raw = pd.read_csv(upload)
            st.header("2 Â· Map claim fields")
            mapping: dict[str, str] = {}
            for target in CLAIM_COLUMNS:
                mapping[target] = st.selectbox(
                    target.replace("_", " ").title(), raw.columns,
                    index=list(raw.columns).index(guess_column(list(raw.columns), target)),
                    key=f"map_{target}",
                )
            claims = raw.rename(columns={source_col: target for target, source_col in mapping.items()})[CLAIM_COLUMNS]
        except Exception as exc:
            st.error(f"Could not read this CSV: {exc}")
            claims = None

    st.divider()
    st.header("3 Â· LLM mode")
    llm_available = bool(os.getenv("OPENAI_API_KEY", "").strip())
    use_llm = st.checkbox("Use configured LLM with read-only tools", value=False, disabled=not llm_available)
    if llm_available:
        model = os.getenv("OPENAI_MODEL", "gpt-6-luna").strip() or "gpt-6-luna"
        st.caption(f"API model: `{model}` (change `OPENAI_MODEL` in `.env`)")
        st.caption("OpenAI API usage may be billed separately from a ChatGPT subscription.")
    else:
        st.caption("Local TF-IDF mode is ready. Add an OpenAI API key to `.env` to enable optional tool calling.")
    if use_llm:
        st.warning("The question and selected claim-tool results will be sent to the configured API endpoint. Use fictional or approved data only.")


if claims is not None:
    try:
        claims = normalize_claims(claims)
        agent = ClaimsResearchAgent(claims)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    summary = agent.portfolio_summary()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Claims", f"{summary['claim_count']:,}")
    m2.metric("Open / pending", f"{summary['open_count'] + summary['pending_count']:,}")
    m3.metric("Total incurred", f"${summary['total_incurred']:,.0f}")
    m4.metric("Median incurred", f"${summary['median_incurred']:,.0f}")

    st.subheader("Research a claim or question")
    question = st.text_area(
        "Ask a question",
        value="Review claim CLM-10001 and show the most relevant policy research leads.",
        height=90,
        placeholder="For example: How many open claims are in the portfolio? Or research claim CLM-10001.",
    )
    run = st.button("Research", type="primary")
    if run:
        with st.spinner("Searching records and policy referencesâ€¦"):
            try:
                answer, mode = agent.answer(question, use_llm=use_llm)
                st.caption(f"Mode: {mode}")
                st.markdown(answer)
            except (RuntimeError, ValueError) as exc:
                st.error(str(exc))

    with st.expander("Claim records", expanded=False):
        st.dataframe(claims, hide_index=True, use_container_width=True)

    with st.expander("Retrieval evaluation", expanded=False):
        st.write("A small, curated check of whether expected policy references appear in the top three TF-IDF results. It measures retrieval only; it does not evaluate coverage decisions.")
        if st.button("Run retrieval evaluation"):
            result = evaluate_retrieval(agent.retriever)
            a, b, c = st.columns(3)
            a.metric("Precision@3", f"{result['precision_at_k']:.2f}")
            b.metric("Recall@3", f"{result['recall_at_k']:.2f}")
            c.metric("Mean reciprocal rank", f"{result['mrr']:.2f}")
            st.dataframe(pd.DataFrame(result["cases"]), hide_index=True, use_container_width=True)

st.divider()
st.caption("The built-in claim records and policy summaries are fictional demonstration data. Uploaded CSV files are analyzed in this app process and are not saved by this project.")


