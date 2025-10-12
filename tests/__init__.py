"""Test package bootstrap helpers."""

from __future__ import annotations

import sys
from pathlib import Path


# Ensure the library source directory is importable without installing the package.
_ROOT = Path(__file__).resolve().parent.parent
_PKG_DIR = _ROOT / "libs" / "envector"
if _PKG_DIR.is_dir():
    pkg_path = str(_PKG_DIR)
    if pkg_path not in sys.path:
        sys.path.insert(0, pkg_path)
