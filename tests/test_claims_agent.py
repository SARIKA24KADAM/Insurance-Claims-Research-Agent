import json
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from claims_agent import (
    CLAIM_COLUMNS,
    ClaimsResearchAgent,
    PolicyRetriever,
    make_demo_claims,
    normalize_claims,
)
from evaluate_claims_agent import evaluate_retrieval


class ClaimsAgentTests(unittest.TestCase):
    def setUp(self):
        self.claims = make_demo_claims(rows=30)
        self.agent = ClaimsResearchAgent(self.claims)

    def test_demo_data_is_reproducible_and_has_expected_schema(self):
        pd.testing.assert_frame_equal(make_demo_claims(rows=30), self.claims)
        self.assertEqual(list(self.claims.columns), CLAIM_COLUMNS)
        self.assertTrue(self.claims["claim_id"].str.startswith("CLM-").all())

    def test_retriever_returns_relevant_auto_collision_document(self):
        results = PolicyRetriever().search("auto collision damage deductible", "Auto", limit=3)
        self.assertTrue(results)
        self.assertEqual(results[0]["source_id"], "POL-AUTO-01")

    def test_line_of_business_filter_is_case_insensitive(self):
        results = PolicyRetriever().search("collision damage", "auto", limit=5)
        self.assertTrue(results)
        self.assertTrue(all(hit["source_id"].startswith(("POL-AUTO", "POL-CLAIM")) for hit in results))

    def test_normalization_drops_non_numeric_negative_and_infinite_rows(self):
        frame = self.claims.iloc[:4].copy()
        frame.loc[frame.index[0], "incurred_loss"] = np.inf
        frame.loc[frame.index[1], "reserve"] = -1
        frame["days_open"] = frame["days_open"].astype(object)
        frame.loc[frame.index[2], "days_open"] = "not numeric"
        normalized = normalize_claims(frame)
        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized.iloc[0]["claim_id"], frame.iloc[3]["claim_id"])

    def test_normalization_rejects_empty_valid_input(self):
        frame = self.claims.iloc[:1].copy()
        frame["incurred_loss"] = np.inf
        with self.assertRaisesRegex(ValueError, "No rows have valid"):
            normalize_claims(frame)

    def test_lookup_summary_and_claim_comparison(self):
        row = self.claims.iloc[0]
        result = self.agent.lookup_claim(row["claim_id"])
        comparison = self.agent.compare_claims(row["claim_type"])
        summary = self.agent.portfolio_summary()
        self.assertEqual(result["claim_id"], row["claim_id"])
        self.assertGreaterEqual(comparison["sample_count"], 1)
        self.assertEqual(summary["claim_count"], 30)
        self.assertAlmostEqual(summary["total_incurred"], self.claims["incurred_loss"].sum())

    def test_missing_claim_returns_clear_error(self):
        result = self.agent.lookup_claim("CLM-99999")
        self.assertIn("error", result)

    def test_local_answer_includes_research_source_and_human_review_note(self):
        answer, mode = self.agent.answer("Review claim CLM-10001 and search collision coverage", use_llm=False)
        self.assertEqual(mode, "Local TF-IDF research mode")
        self.assertIn("POL-", answer)
        self.assertIn("Human review", answer)

    def test_retrieval_evaluation_returns_bounded_metrics_and_cases(self):
        result = evaluate_retrieval(self.agent.retriever)
        self.assertEqual(len(result["cases"]), 8)
        for metric in ("precision_at_k", "recall_at_k", "mrr"):
            self.assertGreaterEqual(result[metric], 0.0)
            self.assertLessEqual(result[metric], 1.0)

    def test_llm_mode_uses_only_declared_tools_and_checks_citations(self):
        def fake_response(payload):
            response = MagicMock()
            response.__enter__.return_value.read.return_value = json.dumps(payload).encode("utf-8")
            return response

        tool_response = fake_response({"output": [
            {"type": "function_call", "call_id": "call_lookup", "name": "lookup_claim", "arguments": '{"claim_id":"CLM-10001"}'},
            {"type": "function_call", "call_id": "call_search", "name": "search_policy_documents", "arguments": '{"query":"collision damage","line_of_business":"Auto"}'},
        ]})
        final_response = fake_response({"output_text": "The claim is open. See [POL-AUTO-01]."})
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "gpt-6-luna"}), patch(
            "claims_agent.urllib.request.urlopen", side_effect=[tool_response, final_response]
        ) as urlopen:
            answer, mode = self.agent.answer("Review CLM-10001", use_llm=True)

        self.assertEqual(mode, "LLM with read-only tools")
        self.assertIn("[POL-AUTO-01]", answer)
        self.assertEqual(urlopen.call_count, 2)
        first_request = urlopen.call_args_list[0].args[0]
        first_payload = json.loads(first_request.data.decode("utf-8"))
        self.assertTrue(first_request.full_url.endswith("/responses"))
        self.assertEqual({tool["name"] for tool in first_payload["tools"]}, {
            "lookup_claim", "compare_claims", "portfolio_summary", "search_policy_documents"
        })


if __name__ == "__main__":
    unittest.main()

