import unittest
from pathlib import Path

import pandas as pd

from claims_agent import CLAIM_COLUMNS, normalize_claims


class SampleClaimsDataTests(unittest.TestCase):
    def test_sample_data_is_synthetic_and_loads_through_agent_schema(self):
        path = Path(__file__).resolve().parents[1] / "sample-claims-data.csv"
        raw = pd.read_csv(path)
        clean = normalize_claims(raw)
        self.assertEqual(list(raw.columns), CLAIM_COLUMNS)
        self.assertEqual(len(clean), 8)
        self.assertTrue(clean["claim_id"].str.startswith("CLM-").all())
        self.assertTrue(clean["policy_id"].str.startswith("POL-").all())
        self.assertEqual(float(clean["incurred_loss"].sum()), 41080.0)


if __name__ == "__main__":
    unittest.main()

