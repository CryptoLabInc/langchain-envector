#!/usr/bin/env python3
"""Run unit tests without requiring a running Envector (ES2) server.

This script runs pytest while excluding tests marked as `integration`.
It is safe to use in environments without the es2 SDK or server.
"""

import sys
import subprocess


def main() -> int:
    cmd = [sys.executable, "-m", "pytest", "-q", "-m", "not integration"]
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        print("pytest not found. Install with: python -m pip install pytest", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

