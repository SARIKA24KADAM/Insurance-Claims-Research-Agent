import unittest
from pathlib import Path

import pandas as pd

from claims_agent import ClaimsResearchAgent


class ClaimsWorkflowIntegrationTests(unittest.TestCase):
    def test_sample_csv_to_local_research_answer(self):
        sample_path = Path(__file__).resolve().parents[1] / "sample-claims-data.csv"
        claims = pd.read_csv(sample_path)
        agent = ClaimsResearchAgent(claims)

        answer, mode = agent.answer(
            "Review claim CLM-10001 and find collision coverage guidance.",
            use_llm=False,
        )

        self.assertEqual(mode, "Local TF-IDF research mode")
        self.assertIn("CLM-10001", answer)
        self.assertIn("POL-AUTO-01", answer)
        self.assertIn("Human review", answer)


if __name__ == "__main__":
    unittest.main()

