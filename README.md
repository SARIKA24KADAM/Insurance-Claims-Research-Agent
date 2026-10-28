# Insurance Claims Research Agent

A local research assistant for exploring synthetic insurance claims. It searches policy guidance, looks up claim records, compares similar claims, and returns evidence with source IDs for human review.

> Demonstration project using fictional claims and synthetic policy summaries. It does not make coverage, liability, fraud, or settlement decisions.

## Features

- TF-IDF vector retrieval over a small policy and claim-handling reference library.
- Read-only tools for claim lookup, claim-type comparisons, and portfolio aggregates.
- Optional LLM tool calling through the OpenAI Responses API.
- CSV upload with field mapping for a user's claim data.
- Curated retrieval evaluation reporting Precision@3, Recall@3, and MRR.

## Run locally

Requires Python 3.10 or newer.

```powershell
python -m pip install -r requirements.txt
python -m streamlit run claims_app.py
```

The app opens at the local URL printed by Streamlit, usually http://localhost:8501. Local TF-IDF research mode works without an API key.

## Optional LLM mode

Copy `.env.example` to `.env`, then add your API key. The default model is `gpt-6-luna`; you can set `OPENAI_MODEL` to another model supported by your endpoint.

```env
OPENAI_API_KEY=your-key
OPENAI_MODEL=gpt-6-luna
OPENAI_BASE_URL=https://api.openai.com/v1
```

Restart Streamlit and select **Use configured LLM with read-only tools**. The question and tool results are sent to the configured endpoint. Do not use confidential or identifying claim data unless the endpoint and use are approved.

## CSV fields

Uploads are mapped to these fields in the sidebar: `claim_id`, `policy_id`, `line_of_business`, `claim_type`, `status`, `incurred_loss`, `reserve`, `days_open`, `territory`, and `description`. Invalid numeric values and negative amounts or durations are excluded. The app does not write uploaded files to disk.

## Evaluation and limitations

The retrieval panel checks eight hand-authored search queries against expected policy document IDs. Its scores measure retrieval ranking only; they do not evaluate answer correctness, citation faithfulness, or production reliability. See [the architecture guide](docs/claims-agent-architecture.md) for methodology and limitations.

Policy summaries are fictional and are not contract wording. Similar-claim metrics are descriptive rather than predictive. Confirm claim facts and actual policy terms with an authorized claims professional.
