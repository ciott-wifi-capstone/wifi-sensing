"""Mathematical unit tests for WiCAL-adapted VitalCSI SQI v1.0.1."""

from __future__ import annotations

import unittest

import numpy as np

from wical.sqi import (
    _agreement_score,
    adapted_band_power_ratio,
    adapted_peak_count,
    compute_observation_sqi,
    fft_respiration_rate,
    select_by_sqi,
    spectral_purity_index,
)


FS = 10.0
DURATION = 90.0
TIME = np.arange(int(FS * DURATION)) / FS


def sine(bpm: float, scale: float = 1.0) -> np.ndarray:
    return scale * np.sin(2.0 * np.pi * (bpm / 60.0) * TIME)


class TestSQIMathematics(unittest.TestCase):
    def test_band_power_prefers_respiratory_band(self) -> None:
        inside = adapted_band_power_ratio(sine(15.0), FS)
        outside = adapted_band_power_ratio(sine(60.0), FS)
        self.assertGreater(inside, 0.95)
        self.assertLess(outside, 0.05)

    def test_fft_uses_native_bin(self) -> None:
        self.assertAlmostEqual(fft_respiration_rate(sine(20.0), FS), 20.0, places=10)

    def test_peak_count_clean_sine(self) -> None:
        rr, count = adapted_peak_count(sine(15.0), FS)
        self.assertGreaterEqual(count, 20)
        self.assertAlmostEqual(rr, 15.0, delta=0.15)

    def test_spi_is_amplitude_scale_invariant(self) -> None:
        reference = spectral_purity_index(sine(15.0), FS)
        tiny = spectral_purity_index(sine(15.0, 1e-20), FS)
        huge = spectral_purity_index(sine(15.0, 1e20), FS)
        self.assertAlmostEqual(reference, tiny, places=12)
        self.assertAlmostEqual(reference, huge, places=12)

    def test_spi_is_higher_for_clean_tone_than_noise(self) -> None:
        rng = np.random.default_rng(20261013)
        clean = spectral_purity_index(sine(15.0), FS)
        noise = spectral_purity_index(rng.normal(size=len(TIME)), FS)
        self.assertGreater(clean, noise)

    def test_spi_rejects_flat_signal(self) -> None:
        with self.assertRaisesRegex(ValueError, "DEGENERATE_SPECTRUM"):
            spectral_purity_index(np.zeros_like(TIME), FS)

    def test_agreement_boundary_is_inclusive(self) -> None:
        self.assertEqual(_agreement_score(10.0, 14.0), 1.0)
        self.assertEqual(_agreement_score(10.0, 14.000001), 0.0)
        self.assertEqual(_agreement_score(float("nan"), 14.0), 0.0)
        # The public observation function is also checked for finite [0, 1] output.
        result = compute_observation_sqi(sine(15.0), sine(15.0), FS)
        self.assertEqual(result["gamma_agree"], 1.0)
        self.assertGreaterEqual(result["sqi"], 0.0)
        self.assertLessEqual(result["sqi"], 1.0)

    def test_selection_tie_uses_fixed_order(self) -> None:
        rows = [
            {"state": "G2", "pair": "Rx1/Rx2", "sqi": 0.5},
            {"state": "G1", "pair": "Rx0/Rx2", "sqi": 0.5},
        ]
        selection = select_by_sqi(rows)
        self.assertEqual(selection["status"], "SELECTED")
        self.assertEqual(selection["selected"]["state"], "G1")
        self.assertEqual(selection["selected"]["pair"], "Rx0/Rx2")
        self.assertEqual(selection["tie_count"], 2)

    def test_all_zero_abstains(self) -> None:
        rows = [
            {"state": "G1", "pair": "Rx0/Rx1", "sqi": 0.0},
            {"state": "G4", "pair": "Rx1/Rx2", "sqi": 0.0},
        ]
        selection = select_by_sqi(rows)
        self.assertEqual(selection["status"], "NO_QUALITY_DIFFERENTIATION")
        self.assertIsNone(selection["selected"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
