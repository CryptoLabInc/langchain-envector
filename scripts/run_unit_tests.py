"""Run the unit tests (fakes only, no server) — a thin wrapper around pytest.

An earlier version imported a fixed list of test modules and called each
``test_*`` function directly. That skipped every module not on the list,
ignored ``xfail`` markers, and counted ``async def`` tests as passed without
running them. pytest already handles all of that, so delegate to it.
"""

from __future__ import annotations

import sys

import pytest

if __name__ == "__main__":
    raise SystemExit(pytest.main(["tests", "-m", "not integration", "-q", *sys.argv[1:]]))
