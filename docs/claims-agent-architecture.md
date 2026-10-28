# Claims Research Agent

The Claims Research Agent is a local Streamlit demonstration of grounded document retrieval and bounded tool use for insurance claim research. It uses fictional policy guidance and synthetic claims. It is not a production coverage or settlement system.

## Run

Install the repository requirements, then run:

```powershell
python -m streamlit run claims_app.py
```

Local mode works without credentials. It uses scikit-learn TF-IDF vectors and cosine similarity to retrieve policy passages, then produces a deterministic summary from retrieved evidence and structured claim data.

Optional LLM mode uses an OpenAI-compatible Responses API endpoint. Configure `OPENAI_API_KEY`, `OPENAI_MODEL`, and optionally `OPENAI_BASE_URL` in the process environment. The app has an explicit toggle; no LLM request is made by default. Tool access is limited to read-only claim lookup, portfolio aggregates, claim-type comparisons, and policy document search. Tool calls are capped at five per answer. The request sets `store: false`; the question and any tool results still leave the local process and are sent to the configured endpoint.

## Data flow

```mermaid
flowchart LR
    U[Analyst question] --> A[ClaimsResearchAgent]
    C[CSV or synthetic claims] --> T[Read-only claim tools]
    D[Synthetic policy summaries] --> R[TF-IDF index]
    A --> R
    A --> T
    R --> E[Cited evidence passages]
    T --> F[Claim facts and aggregates]
    E --> A
    F --> A
    A --> O[Research summary for human review]
    Q[Curated retrieval queries] --> V[Precision@k, recall@k, MRR]
    R --> V
```

The local answering path identifies a claim ID or portfolio question, retrieves matching policy notes, and includes source IDs and a human-review reminder. When LLM mode is selected, the model can request only the four declared tools. It cannot run arbitrary code or mutate claim data.

## Claim CSV fields

Map the uploaded file to these required fields in the sidebar:

| Field | Meaning |
| --- | --- |
| `claim_id` | Claim identifier |
| `policy_id` | Policy identifier |
| `line_of_business` | For example, Auto or Home |
| `claim_type` | For example, Collision, Water, or Theft |
| `status` | Open, Pending, or Closed (other labels are retained) |
| `incurred_loss` | Numeric incurred amount |
| `reserve` | Numeric reserve |
| `days_open` | Non-negative duration in days |
| `territory` | Region or territory label |
| `description` | Short loss description |

Rows with invalid numeric amounts or negative loss, reserve, or duration are excluded. The uploaded file is held in the Streamlit process; this project does not write it to disk.

## Retrieval evaluation

The app's **Retrieval evaluation** panel runs eight fixed queries against the policy library. It reports:

- **Precision@3:** relevant expected documents in the first three results divided by three.
- **Recall@3:** expected documents found in the first three results divided by the expected relevant documents.
- **MRR:** average reciprocal rank of the first expected relevant result.

These queries are a small hand-authored smoke measure. They do not measure answer correctness, citation faithfulness, user utility, fairness, or production reliability. Extend the dataset with representative questions and independently reviewed relevance labels before treating retrieval scores as evidence of quality.

## Safety and limitations

- Policy passages are short synthetic summaries, not actual insurance contracts. A match is a research lead, not evidence of coverage.
- The agent does not decide coverage, liability, fraud, or settlement. An authorized professional must review the contract and claim facts.
- Similar-claim summaries are descriptive and may reflect small or biased samples. They are not predictions or recommended reserves.
- In local mode, data stays in the app process. In optional LLM mode, the question and tool outputs are sent to the configured endpoint. Do not use confidential or directly identifying claim data unless that endpoint and use are approved.
- LLM answers can be incomplete or incorrect even when they cite sources. Check every cited passage and claim record.
- No fine-tuning is included. Retrieval quality and bounded tool use are implemented first; fine-tuning would need a governed, labeled corpus and task-specific evaluation.

For the OpenAI request and function-tool loop, see [official OpenAI function-calling documentation](https://developers.openai.com/api/docs/guides/function-calling).


