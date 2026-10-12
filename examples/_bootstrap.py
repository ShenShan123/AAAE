"""Make the local package available to scripts and their child processes."""

import os
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))
_existing = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = str(SOURCE_ROOT) + (os.pathsep + _existing if _existing else "")
