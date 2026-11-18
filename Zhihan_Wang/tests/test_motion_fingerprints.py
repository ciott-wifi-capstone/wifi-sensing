"""Check identity coverage, aggregation and undefined fingerprint correlations."""

import unittest
import warnings
import numpy as np
from wical import motion as m


def fixture():
    features = []
    runs = []
    for run in ("run1", "run2"):
        runs.append(
            {
                "dataset": "example",
                "motion": "FRONT",
                "run": run,
                "events": [{}, {}, {}],
            }
        )
        for event, offset in enumerate((1.0, 3.0, 5.0), 1):
            for i, (state, pair) in enumerate(m.OBSERVATIONS):
                features.append(
                    {
                        "dataset": "example",
                        "motion": "FRONT",
                        "run": run,
                        "event_id": f"E{event}",
                        "state": state,
                        "pair": pair,
                        "amplitude_shift_db": i + offset,
                        "amplitude_shift_z": 2 * i + offset,
                        "amplitude_abs_response_z": i * i + offset,
                        "differential_energy_db": (i % 5) + offset,
                        "complex_displacement_norm": (i + offset) / 10,
                    }
                )
    return features, runs


class FingerprintTests(unittest.TestCase):
    def test_event_medians_and_row_order_independence(self):
        features, runs = fixture()
        result = m.build_run_fingerprints(features, runs)
        reordered = m.build_run_fingerprints(list(reversed(features)), runs)
        self.assertEqual(
            result["run_fingerprint_rows"], reordered["run_fingerprint_rows"]
        )
        self.assertEqual(len(result["run_fingerprint_rows"]), 36)
        self.assertEqual(
            result["run_fingerprint_rows"][0]["median_amplitude_shift_db"], 3.0
        )
        self.assertEqual(result["run_fingerprint_rows"][0]["candidate_event_count"], 3)
        self.assertAlmostEqual(result["similarity_rows"][0]["spearman_r"], 1.0)
        self.assertEqual(result["action_similarity_rows"], [])

    def test_missing_duplicate_and_nonfinite_features_rejected(self):
        features, runs = fixture()
        for bad in (
            features[1:],
            features + [features[0]],
            [{**features[0], "amplitude_shift_db": float("nan")}, *features[1:]],
        ):
            with self.assertRaises(ValueError):
                m.build_run_fingerprints(bad, runs)

    def test_constant_fingerprints_do_not_claim_correlation(self):
        features, runs = fixture()
        for row in features:
            row["amplitude_abs_response_z"] = 1.0
            row["differential_energy_db"] = 1.0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = m.build_run_fingerprints(features, runs)
        self.assertTrue(np.isnan(result["similarity_rows"][0]["spearman_r"]))
        self.assertIsNone(
            result["within_condition_repeatability"]["FRONT"]["median_spearman_r"]
        )
        self.assertTrue(any("UNDEFINED_CORRELATION" in str(w.message) for w in caught))

    def test_no_event_run_produces_no_fingerprint(self):
        with self.assertWarns(RuntimeWarning):
            result = m.build_run_fingerprints(
                [],
                [{"dataset": "empty", "motion": "DOWN", "run": "run1", "events": []}],
            )
        self.assertEqual(result["run_fingerprint_rows"], [])
        self.assertEqual(result["similarity_rows"], [])


if __name__ == "__main__":
    unittest.main()
