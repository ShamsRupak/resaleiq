# Contributing to ResaleIQ

Thanks for the interest. This project is primarily a personal portfolio artifact, but contributions that improve reproducibility, test coverage, or documentation are welcome.

## Development setup

```bash
git clone https://github.com/ShamsRupak/resaleiq.git
cd resaleiq
uv sync --all-extras
```

Everything else flows from `make help`:

```bash
make help
```

## Branching

- `main` is the published branch. It should always be green on CI.
- Work on topic branches named `feat/<short-slug>`, `fix/<short-slug>`, or `docs/<short-slug>`.
- Open a PR against `main` when your branch is ready.

## Code quality gates

Every change must pass these checks before merge. CI runs them automatically on every push and PR.

```bash
make lint    # ruff check + ruff format --check
make type    # mypy strict mode on src/
make test    # pytest suite (104 tests)
make check   # all three gates at once
```

### Style

- **Python:** formatted by `ruff format`; line length 100; target `py311`. Ruff lints for `E`, `W`, `F`, `I`, `N`, `UP`, `B`, `SIM`, and `RUF` rule families. See `pyproject.toml` for the full config.
- **Type hints:** required on all new public functions. Mypy runs in strict mode. Third-party libraries without stubs (e.g., `xgboost`, `pyarrow`) are listed as exceptions in `pyproject.toml`.
- **Docstrings:** one-line summary minimum on public functions; module-level docstring on every new file. Longer functions should document side effects and return shapes for dataframes.
- **Imports:** sorted by `ruff` (isort-compatible). Absolute imports within the package.
- **SQL:** leading-comma style, CTEs preferred over nested subqueries, explicit `ORDER BY` on any query feeding a result table.

### Tests

- New modules require at least one test file.
- Fixtures live in `tests/conftest.py`. Session-scoped fixtures cache expensive setup (data generation, model training).
- SQL tests auto-skip when Postgres is unreachable. Everything else runs in under 5 seconds.
- Use `pytest -k <substring>` to run a subset. Use `make test` for the full run locally.

## Changing the dataset

The dataset is fully reproducible from `MASTER_SEED = 20260420` in `src/resaleiq/config.py`. Changing the seed or any generator parameter will shift every downstream number. If you intentionally change a generator parameter:

1. Regenerate: `make generate`.
2. Retrain every model: `make phase3-all && make lot-train`.
3. Update any docs that cite specific numbers (`README.md`, `docs/MODEL_CARD.md`, `docs/PHASE3_TECHNICAL_NOTES.md`, the dashboard captions).
4. Verify tests still pass: `make test`.

Drift between cited numbers and reproduced numbers is the most common review comment. Keep them in sync.

## Commit messages

Short imperative subject line under 72 characters. Expanded body optional. Examples:

```
feat(ml): add empirical-quantile conformal calibration to lot model
fix(dashboard): use modern groupby.agg to silence pandas 2.2 warning
docs(readme): add Phase 6 Streamlit Community Cloud deploy steps
test(lot_model): cover holdout-too-small edge case
```

## Reporting bugs

Open an issue with:

1. What you ran (`make ...` or the exact command).
2. What you expected.
3. What you got. Copy-paste terminal output is ideal.
4. Your environment: `python --version`, `uv --version`, OS, `xgboost.__version__` if a numeric result drifted.

Numeric drift across platforms of ±0.05 percentage points on MAPE is expected and not a bug. Larger drift is worth reporting.

## What not to send

- Changes that add proprietary data, API keys, or real company names to the tracked files.
- Changes that bypass the quality gates (e.g., adding `# type: ignore` without justification, suppressing ruff rules in new code).
- Changes that regress test coverage without a clear reason.

## License

By contributing, you agree that your contributions will be licensed under the project's MIT license (see `LICENSE`).
