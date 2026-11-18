"""Exercise real command boundaries and ensure bad inputs cannot damage files."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from wical.io import ROOT, sha256


class CommandTests(unittest.TestCase):
    def run_script(self, name, *args):
        return subprocess.run(
            [sys.executable, "-B", str(ROOT / "scripts" / name), *map(str, args)],
            cwd=tempfile.gettempdir(),
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_help_imports_from_another_directory(self):
        for name in ("analyze_static.py", "analyze_motion.py", "evaluate_selection.py"):
            with self.subTest(script=name):
                result = self.run_script(name, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_input_reports_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            result = self.run_script(
                "analyze_static.py", "--run-dir", Path(tmp) / "missing", "--out", output
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Missing input", result.stderr)
            self.assertFalse(output.exists())

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            sentinel = Path(tmp) / "keep.txt"
            sentinel.write_text("keep")
            result = self.run_script(
                "evaluate_selection.py",
                "--tables",
                ROOT / "results/static",
                "--out",
                tmp,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("already exists", result.stderr)
            self.assertEqual(sentinel.read_text(), "keep")

    def test_empty_waveform_directory_keeps_failure_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "input"
            folder.mkdir()
            output = Path(tmp) / "out"
            result = self.run_script(
                "analyze_static.py", "--waveforms", folder, "--out", output
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("No *_waveforms.npz", result.stderr)
            self.assertIn("FAILED", (output / "run.json").read_text())


if __name__ == "__main__":
    unittest.main()
