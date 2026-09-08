"""Shared configuration for the occurrence regression tests."""

import os
import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = REPO_DIR.parent

# The source tree is currently used as a namespace package rather than an
# installed distribution. Add its parent just as the project driver does.
sys.path.insert(0, str(PACKAGE_PARENT))

# Keep Matplotlib's test cache local to the test directory and writable.
os.environ.setdefault("MPLCONFIGDIR", str(REPO_DIR / "tests" / ".mplconfig"))
