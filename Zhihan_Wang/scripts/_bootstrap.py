"""Set the local import path and a noninteractive plotting backend."""

import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "wical-matplotlib")
)
os.environ.setdefault("MPLBACKEND", "Agg")
