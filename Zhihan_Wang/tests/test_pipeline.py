"""Checks for data leakage, identity mapping, timing and input/output damage."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
import warnings
import numpy as np
from wical import farsense, sqi
from wical.csi import assign_states, state_ratio, validate_decoded
from wical.static import estimate_ratio
from wical.selection import select_without_reference, evaluate
from wical.io import ROOT, read_csv, fresh_output, check_time


class PipelineTests(unittest.TestCase):
    def test_decoder_checks_metadata_not_only_allocated_shape(self):
        fields = np.dtype([(n, "i4") for n in ("NumTones", "NumRx", "NumTx", "NumESS")])
        raw = np.zeros(
            2, dtype=[("CSI", [("Info", fields), ("CSI", "c16", (30, 3, 1))])]
        )
        info = raw["CSI"]["Info"]
        info["NumTones"] = 30
        info["NumRx"] = 3
        info["NumTx"] = 1
        self.assertEqual(validate_decoded(raw).shape, (2, 1, 3, 30))
        info["NumTones"][1] = 56
        with self.assertRaises(ValueError):
            validate_decoded(raw)

    def tables(self):
        return read_csv(ROOT / "results/static/observations.csv"), read_csv(
            ROOT / "results/static/quality.csv"
        )

    def test_saved_metrics_and_counts(self):
        obs, quality = self.tables()
        rows, summary = evaluate(obs, quality)
        self.assertEqual((len(obs), len(rows)), (144, 8))
        np.testing.assert_allclose(
            [r["mae_bpm"] for r in summary],
            [
                4.226726878963676,
                2.123619733877707,
                2.188143218942675,
                0.2834063667906077,
            ],
            rtol=1e-12,
            atol=1e-12,
        )

    def test_reference_cannot_affect_automatic_selection(self):
        obs, quality = self.tables()
        before = select_without_reference(obs, quality)
        for row in obs:
            row["gt_rr_bpm"] = "99999"
            row["abs_error_vs_gt"] = "NaN"
            row["condition"] = "-1 bpm"
        self.assertEqual(before, select_without_reference(obs, quality))

    def test_duplicate_and_missing_observations_rejected(self):
        obs, quality = self.tables()
        for bad in (obs[:-1], obs + [obs[0]]):
            with self.assertRaises(ValueError):
                select_without_reference(bad, quality)

    def test_nonfinite_and_inconsistent_errors_rejected(self):
        obs, quality = self.tables()
        obs[0]["abs_error_vs_gt"] = "999"
        with self.assertRaises(ValueError):
            evaluate(obs, quality)
        obs, quality = self.tables()
        quality[0]["sqi"] = "nan"
        with self.assertRaises(ValueError):
            select_without_reference(obs, quality)

    def test_acf_lag_units(self):
        fs = 10.0
        ac = np.cos(2 * np.pi * np.arange(200) / 40)
        self.assertAlmostEqual(farsense.extract_bpm_from_ac(ac, fs), 15.0)
        self.assertAlmostEqual(farsense.extract_bpm_from_ac(ac, 2 * fs), 30.0)

    def test_state_boundary_same_packet_pair_and_medians(self):
        h = np.ones((6, 1, 3, 30), complex)
        h[:, 0, 0, :] = np.asarray([2, 4, 6, 8, 10, 12])[:, None]
        h[:, 0, 1, :] = np.asarray([1, 2, 1, 2, 1, 2])[:, None]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = assign_states(
                h,
                [0, 2_000_000_000, 4_000_000_000],
                ["G1", "G2", "G1"],
                0,
                5_000_000_000,
            )
        self.assertEqual(
            data["packet_state"].tolist(), ["G1", "G1", "G2", "G2", "G1", "G1"]
        )
        ratio, time, fs = state_ratio(data, "G1", 0, 1)
        expected = np.median([10 / (1 + 1e-12), 12 / (2 + 1e-12)])
        self.assertAlmostEqual(ratio[1, 0].real, expected)
        self.assertEqual(fs, 0.25)
        self.assertEqual(time.tolist(), [0.0, 4.0])

    def test_empty_singleton_unsorted_timestamps_and_shapes(self):
        h = np.ones((2, 1, 3, 30), complex)
        for events, states in [([], []), ([2, 1], ["G1", "G2"])]:
            with self.assertRaises(ValueError):
                assign_states(h, events, states, 0, 10)
        with self.assertRaises(ValueError):
            assign_states(h[:1], [0], ["G1"], 0, 10)
        for time in ([0, 0], [0, np.nan], [1, 0]):
            with self.assertRaises(ValueError):
                check_time(time, 2)
        for ratio in (np.zeros((0, 30)), np.zeros((90, 29)), np.full((90, 30), np.nan)):
            with self.assertRaises(ValueError):
                estimate_ratio(ratio, 10.0)

    def test_output_does_not_overlap_inputs_or_frozen_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "input"
            data.mkdir()
            with self.assertRaises(ValueError):
                fresh_output(data / "output", [data])
            with self.assertRaises(ValueError):
                fresh_output(Path(tmp), [data])
            self.assertEqual(
                fresh_output(Path(tmp) / "output", [data]),
                (Path(tmp) / "output").resolve(),
            )

    def test_sqi_rejects_unequal_signal_lengths_and_bad_gains(self):
        with self.assertRaises(ValueError):
            sqi.compute_observation_sqi(np.ones(100), np.ones(99), 10.0)
        with self.assertRaises(ValueError):
            sqi.reconstruct_farsense_signals(
                np.ones((100, 30), complex),
                10.0,
                np.zeros(30),
                np.array([0]),
                np.array([np.nan]),
                np.array([1]),
            )
        duplicate = [{"state": "G1", "pair": "Rx0/Rx1", "sqi": 0.5}] * 2
        with self.assertRaises(ValueError):
            sqi.select_by_sqi(duplicate)
        with self.assertRaises(ValueError):
            fresh_output(ROOT / "results/new", [])

    def test_policy_matches_implementation(self):
        policy = json.loads((ROOT / "configs/frozen_policy.json").read_text())
        self.assertEqual(policy["bnr_nfft"], farsense.BNR_NFFT)
        self.assertEqual(policy["bnr_threshold"], farsense.BNR_THRESHOLD)
        self.assertEqual(policy["projection_angles"], farsense.NUM_PROJECTION_ANGLES)
        self.assertEqual(policy["sqi_agreement_bpm"], sqi.AGREEMENT_BRPM)
        self.assertEqual(policy["sg_window_sec"], sqi.SG_WINDOW_SEC)
        self.assertEqual(policy["states"], list(sqi.STATE_ORDER))


if __name__ == "__main__":
    unittest.main()
