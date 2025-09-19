# Contributing to LangChain Envector Integration

Thanks for your interest in improving the project! This guide covers local setup, testing, and expectations for contributions.

## Project Setup
1. Use Python 3.11 (other 3.9–3.13 versions are supported but 3.11 is our baseline).
2. Create a virtual environment:
   - `python3.11 -m venv .venv && source .venv/bin/activate`
3. Install tooling and dependencies:
   - `pip install -U pip setuptools wheel`
   - `pip install -e .`
   - `pip install -r tests/requirements.txt`

## Testing
- **Unit tests** (fakes only): `python run_unit_tests.py`
- **Integration tests** (requires ES2 server + keys):
  - Export `ES2_ADDRESS`, `ES2_KEY_PATH`, `ES2_KEY_ID`
  - Optional: `ES2_USE_EMBEDDINGS=1`, `ES2_EMB_MODEL`, `ES2_USE_HF_DATASET=1`
  - Run `pytest -m integration -s`

Please run relevant tests before submitting a PR and mention coverage in the description.

## Development Guidelines
- Keep code, comments, and docs in English.
- Prefer the high-level `es2` SDK APIs; avoid direct gRPC/indexer calls unless required.
- Keep changes focused and documented; update README or notebooks when behavior changes.
- Follow existing formatting and type-hint conventions.

## Pull Request Checklist
- [ ] Rebased on the latest `main` (or resolved merge conflicts).
- [ ] Added tests or updated existing ones when behavior changes.
- [ ] Updated documentation if user-facing behavior changed.
- [ ] Verified `python run_unit_tests.py` (and integration tests if applicable).

We appreciate small, well-scoped PRs. Feel free to open a draft PR early to discuss larger changes.
