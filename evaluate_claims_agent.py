"""Small curated retrieval evaluation for the synthetic policy document index."""
from __future__ import annotations

from typing import Any

from claims_agent import PolicyRetriever


EVALUATION_CASES = [
    {"query": "auto collision damage deductible covered vehicle", "line_of_business": "Auto", "expected_sources": ["POL-AUTO-01"]},
    {"query": "theft of personal belongings proof of ownership", "line_of_business": "Home", "expected_sources": ["POL-HOME-03"]},
    {"query": "water damage source duration mitigation", "line_of_business": "Home", "expected_sources": ["POL-HOME-01"]},
    {"query": "hail storm roof wear and deterioration", "line_of_business": "Home", "expected_sources": ["POL-HOME-02"]},
    {"query": "third party injury liability limits", "line_of_business": "Auto", "expected_sources": ["POL-AUTO-03"]},
    {"query": "claim reserve open closed pending status", "line_of_business": "All", "expected_sources": ["POL-CLAIM-02"]},
    {"query": "privacy confidential personal data external service", "line_of_business": "All", "expected_sources": ["POL-CLAIM-04"]},
    {"query": "coverage disputes litigation severe injury escalate to human", "line_of_business": "All", "expected_sources": ["POL-CLAIM-03"]},
]


def evaluate_retrieval(retriever: PolicyRetriever, k: int = 3) -> dict[str, Any]:
    """Compute precision@k, recall@k and MRR for the curated search cases."""
    case_results = []
    precisions = []
    recalls = []
    reciprocal_ranks = []
    for case in EVALUATION_CASES:
        hits = retriever.search(case["query"], case["line_of_business"], k)
        ranked_ids = [hit["source_id"] for hit in hits]
        expected = set(case["expected_sources"])
        found = expected.intersection(ranked_ids)
        precisions.append(len(found) / k)
        recalls.append(len(found) / len(expected))
        first_rank = next((rank for rank, source_id in enumerate(ranked_ids, start=1) if source_id in expected), None)
        reciprocal_ranks.append(1 / first_rank if first_rank else 0.0)
        case_results.append({
            "query": case["query"],
            "expected": ", ".join(sorted(expected)),
            "retrieved": ", ".join(ranked_ids),
            "hit": bool(found),
        })
    return {
        "k": k,
        "precision_at_k": sum(precisions) / len(precisions),
        "recall_at_k": sum(recalls) / len(recalls),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "cases": case_results,
    }


