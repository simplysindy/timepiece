# Repository Guidelines

## Project Structure & Module Organization
- `src/scraping/`: Discovery, scraping, validation. Entry: `python -m src.scraping.scraping`.
- `src/data_prep/`: Cleans and aggregates scraped data. Entry: `python -m src.data_prep.data_prep`.
- `src/training/`: ML training pipeline (models, features, metrics). Entry: `python -m src.training.training`.
- `conf/*.yaml`: Hydra configs for each pipeline (`scraping.yaml`, `data_prep.yaml`, `training.yaml`).
- `docs/`: Design notes and refactor docs. `outputs/` and data folders are generated at runtime.

## Build, Test, and Development Commands
- Setup (Python 3.10+ recommended)
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
- Run scraping: `python -m src.scraping.scraping [overrides...]`
  - Example: `python -m src.scraping.scraping scraping.headless=false discovery.target_count_per_brand=5`
- Run data prep: `python -m src.data_prep.data_prep data.max_files=10`
- Run training: `python -m src.training.training training.models=["lightgbm","xgboost"]`
- Optional formatting/lint (if installed): `black src` and `ruff check src`
Note: Hydra changes the working directory; prefer config-driven paths or absolute paths when reading/writing files.

## Coding Style & Naming Conventions
- Python style: PEP8, 4-space indentation, type hints where practical.
- Naming: `snake_case` for functions/variables/modules, `PascalCase` for classes.
- Config keys in YAML use `snake_case`. Place new configs in `conf/` and name by domain (e.g., `scraping_experiment.yaml`).
- Keep modules focused by domain: `scraping/`, `data_prep/`, `training/`, shared helpers in `utils/`.

## Testing Guidelines
- Current repo has minimal tests. Use `pytest` and place tests under `tests/` mirroring `src/`.
- File names: `tests/scraping/test_discovery.py`, `tests/training/test_metrics.py`.
- Conventions: `test_*` functions, deterministic seeds, mock browser/IO where needed.
- Run: `pytest -q` (add `pytest` to your environment if not installed).

## Commit & Pull Request Guidelines
- Commit style: imperative mood, concise, no trailing period.
  - Examples from history: `Add ML training pipeline`, `Fix discovery headless mode bug`.
- PRs should include: clear description, motivation, key changes, run instructions (commands + config overrides), linked issues, and logs/screenshots for scraping/training when relevant.

## Security & Configuration Tips
- Never commit secrets, cookies, or API keys; use environment variables or local Hydra overrides.
- Respect site policies; keep `delay_range` and `headless` settings reasonable to avoid rate limits.
- Large data and generated artifacts (e.g., `data/processed`, `logs/`, `outputs/`) should remain untracked or listed in `.gitignore`.
