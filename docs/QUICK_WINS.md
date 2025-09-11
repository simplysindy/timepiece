# Quick Wins & High‑Leverage Refactors

Scored by Impact (H/M/L) and Effort (S/M/L). Highest ROI first.

1) Add optional ML deps extras (pip extras) and env docs — Impact: H, Effort: S
- Problem: `requirements.txt` pins scraping only; training needs optional deps (sklearn/scipy/xgboost/lightgbm/statsmodels/tensorflow).
- Fix: Document `pip install -r requirements.txt` plus `pip install scikit-learn scipy xgboost lightgbm statsmodels tensorflow` or provide extras.
- Benefit: Fewer runtime failures; smoother onboarding.

2) Persist unified dataset in Data Prep — Impact: H, Effort: S
- Problem: Training builds `data/output/featured_data.csv` only when missing; make it explicit from Data Prep.
- Fix: Add `output.write_unified=true` to `data_prep.yaml` and write to `data/output/featured_data.csv` after processing.
- Benefit: Clear contract between prep and training; faster iteration.

3) Stabilize Cloudflare handling switches — Impact: H, Effort: M
- Problem: Headless toggles differ for discovery/scraping; UC availability varies.
- Fix: Single `browser.mode={regular,undetected}` + per‑phase `headless`; automatic fallback logs.
- Benefit: Fewer scraping stalls and clearer operability.

4) Add basic e2e smoke tests — Impact: M, Effort: M
- Problem: No tests; regressions slip.
- Fix: `tests/` with mocked Selenium/IO and deterministic sample CSVs; test feature generation and temporal split logic.
- Benefit: Confidence for refactors; CI‑friendly.

5) Parquet support + schema snapshot — Impact: M, Effort: S
- Problem: CSV is large and lossy for types.
- Fix: Toggle to write Parquet for processed/unified datasets; dump column inventory to JSON alongside.
- Benefit: Faster I/O; better type safety.

6) Training reproducibility guardrails — Impact: M, Effort: S
- Problem: Seeds set but not enforced across all libs.
- Fix: Central `set_seed()` and doc GPU nondeterminism; pin versions for tree libs.
- Benefit: Comparable results across runs/machines.

7) Feature catalog auto‑doc — Impact: M, Effort: M
- Problem: Hard to know which features are present per watch/config.
- Fix: After prep, write `data/output/feature_catalog.json` listing columns, origins (lag/rolling/etc.), and config that enabled them.
- Benefit: Transparency; eases model debugging.

8) CLI ergonomics & presets — Impact: M, Effort: S
- Problem: Long commands for overrides.
- Fix: Add `conf/test_*.yaml` presets (e.g., 1 brand, 3 watches), and `make`/shell scripts for common runs.
- Benefit: Faster iteration; safer demos.

9) Error budget & backoff tuning — Impact: M, Effort: M
- Problem: Retries/delays are static.
- Fix: Adaptive backoff based on recent failures per brand; centralized delay controller.
- Benefit: Lower ban risk; better throughput.

10) Model registry shim (local) — Impact: M, Effort: M
- Problem: Artifacts saved as loose files.
- Fix: Versioned directory per run with config + metrics snapshot; optional MLflow/W&B later.
- Benefit: Experiment tracking foundation.

11) Consolidate filename conventions — Impact: S, Effort: S
- Problem: Multiple name sources (`model_name` vs filename parts).
- Fix: Single utility to derive `{brand}-{id}-{name}` across discovery/scraping/prep.
- Benefit: Fewer duplicates and mis‑joins.

12) Pre‑flight checks command — Impact: S, Effort: S
- Problem: Failures late (missing dirs, drivers, deps).
- Fix: `python -m src.utils.doctor` to verify Chrome/driver, permissions, folders.
- Benefit: Faster troubleshooting.

