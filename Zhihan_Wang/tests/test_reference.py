import tempfile
from pathlib import Path
import unittest
import numpy as np
from wical.reference import load_neulog_gt, detect_gt_rr


class ReferenceTests(unittest.TestCase):
    def fixture(self, text):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        p = Path(temp.name) / "reference.csv"
        p.write_text("header\n" * 7 + text)
        return p

    def test_bad_measurement_cannot_leave_unpaired_time(self):
        with self.assertRaisesRegex(ValueError, "line 9"):
            load_neulog_gt(self.fixture("00:00:00,1\n00:00:01,bad\n00:00:02,2\n"))

    def test_empty_nonfinite_and_duplicate_time_rejected(self):
        for text in ("", "00:00:00,nan\n00:00:01,1\n", "00:00:00,1\n00:00:00,2\n"):
            with self.assertRaises(ValueError):
                load_neulog_gt(self.fixture(text))

    def test_known_footer_is_logged_and_excluded(self):
        with self.assertWarns(RuntimeWarning):
            time, value, fs = load_neulog_gt(
                self.fixture(
                    "00:00:00,1\n00:00:01,2\n\nsys info:[]\nAttention! Any change in the file may prevent reading its data.\n"
                )
            )
        self.assertEqual(len(time), 2)
        self.assertEqual(len(value), 2)

    def test_units_and_existing_landmark_formula(self):
        t = np.arange(900) / 10
        self.assertAlmostEqual(
            detect_gt_rr(t, np.sin(2 * np.pi * 0.25 * t), 10), 15, places=10
        )
        time, value, fs = load_neulog_gt(
            self.fixture("00:01:00,1\n00:01:00.1,2\n00:01:00.2,3\n")
        )
        np.testing.assert_allclose(time, [0, 0.1, 0.2], atol=1e-12)
        self.assertAlmostEqual(fs, 10)
