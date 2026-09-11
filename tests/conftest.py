"""Shared configuration for the occurrence regression tests."""

import os
import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = REPO_DIR.parent

# The source tree is used as a namespace package rather than an installed
# distribution, so expose its parent during local tests.
sys.path.insert(0, str(PACKAGE_PARENT))

# Keep Matplotlib's test cache local to the test directory and writable.
os.environ.setdefault("MPLCONFIGDIR", str(REPO_DIR / "tests" / ".mplconfig"))
