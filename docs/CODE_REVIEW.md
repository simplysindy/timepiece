# Engineering Review (Architecture + Code Quality)

This is a senior-level review of the repository’s architecture and code with concrete, source-grounded recommendations to improve readability, maintainability, correctness, and performance. Testability is noted but not the focus per request.

## Summary
- Strong domain separation and operability via Hydra-driven CLIs.
- Robust scraping practices for a difficult target (Cloudflare, dynamic charts).
- End-to-end pipeline implemented: discovery → scraping → validation → data prep → training.
- Key risks: temporal correctness in training, implicit dataset contracts, brittle DOM/Chart.js assumptions, oversized modules with mixed concerns.

---

## Recent Changes (Applied)
- Training temporal splits now strictly sort by a real timestamp per asset and slice by position; a `timestamp` column is derived from `date` when absent, and the code fails fast if temporal ordering cannot be established (`src/training/training.py`).
- Raw time fields are excluded from model features to prevent leakage or garbage signals: `date` and `timestamp` are now in the excluded set (`src/training/features.py`).
- CSV validation checks full-date monotonicity across the entire file, reporting inversion counts and unparseable dates (`src/scraping/validation.py`).

These address the most critical correctness issues with minimal blast radius.

---

## What’s Done Well
- Clear domain boundaries
  - `src/scraping`, `src/data_prep`, `src/training` reflect pipeline phases well.
  - Single-command entry points with Hydra configs for each phase.
- Practical scraping resiliency
  - Retries, random delays, brand batching, Cloudflare detection/wait, headless toggles.
  - Safe CSV writes with backup and incremental updates (`utils.io.safe_write_csv_with_backup`, `load_existing_csv_data`).
- Comprehensive data prep
  - Cleaning, resampling, lags/rolling/momentum/volatility, EMA/RSI/Bollinger, watch/brand/seasonality features, target creation (`src/data_prep/process.py`).
- Training orchestration
  - Multiple algorithms, temporal-per-asset intent, artifacts and metrics saved, clean CLI (`src/training/training.py`).
- Logging and configuration
  - Informative logs throughout; Hydra makes experimentation ergonomic.

---

## High-Risk Issues (Architectural/Correctness)
1) Temporal correctness in training is not guaranteed
- `create_temporal_splits` sorts by DataFrame index, not by a known timestamp column, and the raw `date` string may leak through features.
- `prepare_features` keeps `date` in X and converts to numeric, producing weak/garbage signals and risking leakage.

2) Dataset contract is implicit and not enforced
- No single place defines required columns/types/invariants for “training-ready” data (e.g., `asset_id`, strictly monotonic `timestamp`, non-null `target`).
- Watch IDs/filenames can diverge; `unknown` IDs can collide.

3) Scraping DOM/Chart.js assumptions are brittle
- Extraction requires `point.x instanceof Date`; Chart.js often encodes dates as numbers/timestamps too. Silent failures are possible.
- Page-readiness conditions differ between discovery/scraping; duplicate browser logic.

4) Validation is shallow
- Date ordering check compares only first vs last; partial disorder won’t be caught.

5) Overlarge modules and mixed concerns
- `src/data_prep/process.py` mixes IO, cleaning, features, metadata, and output in one class.
- Browser orchestration and navigation logic duplicated between discovery and scraping.

---

## Architectural Improvements
1) Make time a first-class concept and enforce a dataset contract
- Define and enforce a “training-ready” schema at boundaries:
  - Required columns: `asset_id: str`, `timestamp: datetime64[ns, UTC]`, `target: float`, feature columns (all numeric), optional `brand`, `model`.
  - Invariants per `asset_id`: `timestamp` strictly monotonic increasing; `target` non-null; no duplicate `(asset_id, timestamp)`.
- Validate on load and before split; fail fast with clear errors.

2) Normalize temporal handling across the pipeline
- Always parse/attach a `timestamp` column during data prep.
- In training, group by `asset_id`, sort by `timestamp`, then split by position to avoid leakage.
- Do not include raw `date/timestamp` as a numeric feature; instead derive cyclical features (day-of-week/month) explicitly and drop raw columns.

3) Separate concerns with a thin application layer
- Introduce lightweight domain models (dataclasses): `WatchTarget`, `PricePoint`, `ProcessedRow`.
- Split `process.py` into focused modules: `cleaning.py`, `features.py`, `targets.py`, `metadata.py`, `io.py` (already exists) and keep `data_prep.py` orchestration thin.
- Create a small `ArtifactsRepo` (paths and helpers for processed/unified/models/metrics) to avoid scattering path logic.

4) Unify browser/session orchestration
- One `BrowserFactory` with `mode={regular,undetected}` and `headless=true|false`, used by both discovery and scraping.
- Centralize “navigate+ready” logic with consistent `EC.any_of(...)` and Cloudflare wait strategy.

5) Strengthen validation layer
- Add monotonicity/inversion checks, duplicate timestamps, and missing essential columns.
- Promote validation to gate artifacts (e.g., block training if schema not met).

---

## Concrete Correctness & Robustness Changes
1) Fix temporal splits (src/training/training.py) — IMPLEMENTED
- Use explicit `timestamp` column per asset and sort by it before slicing.

```python
# inside create_temporal_splits()
assert 'asset_id' in df.columns, "asset_id required"
# prefer a strict timestamp column; derive if only 'date' exists
if 'timestamp' not in df.columns:
    if 'date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
    else:
        raise ValueError("No timestamp/date column found")

X, y = prepare_features(df, cfg.data.target_column)

splits = { 'train': [], 'val': [], 'test': [] }
for asset_id, grp in df[['asset_id','timestamp']].dropna().groupby('asset_id'):
    order = grp.sort_values('timestamp').index
    asset_X = X.loc[order]
    asset_y = y.loc[order]
    n = len(asset_X)
    if n < 20:  # configurable min
        logger.warning(f"Skipping {asset_id}: only {n} samples")
        continue
    t_start = int(n * (1 - cfg.training.test_size))
    v_start = int(n * (1 - cfg.training.test_size - cfg.training.val_size))
    splits['train'].append((asset_X.iloc[:v_start], asset_y.iloc[:v_start]))
    splits['val'].append((asset_X.iloc[v_start:t_start], asset_y.iloc[v_start:t_start]))
    splits['test'].append((asset_X.iloc[t_start:], asset_y.iloc[t_start:]))
```

2) Exclude raw date from features (src/training/features.py) — IMPLEMENTED
- Avoid accidental numeric coercion of string dates.

```python
excluded_columns = {
    target_column, 'asset_id', 'brand', 'model', 'date', 'timestamp'
}
```

3) Strengthen Chart.js extraction (src/scraping/core/base_scraper.py)
- Accept both Date objects and numeric timestamps.

```javascript
function normalizePoint(p) {
  const toISO = (d) => new Date(d).toISOString().split('T')[0];
  if (!p) return null;
  if (p.x instanceof Date) return { date: toISO(p.x), price: p.y };
  if (typeof p.x === 'number') return { date: toISO(p.x), price: p.y };
  if (typeof p.x === 'string') return { date: toISO(p.x), price: p.y };
  return null;
}
function extractFromCharts() {
  if (!window.Chart || !window.Chart.instances) return null;
  for (const chart of Object.values(window.Chart.instances)) {
    const ds = (chart.data && chart.data.datasets) || [];
    for (const dataset of ds) {
      const points = (dataset.data || []).map(normalizePoint).filter(Boolean);
      if (points.length) return points;
    }
  }
  return null;
}
return JSON.stringify(extractFromCharts());
```

4) Monotonic date validation (src/scraping/validation.py) — IMPLEMENTED
- Parse all dates; assert strict monotonicity and report inversions.

```python
dates = pd.to_datetime([r[0] for r in rows[1:]], errors='coerce')
if dates.isnull().any():
    return False, row_count, "Unparseable dates present", True
is_monotonic = dates.is_monotonic_increasing
if not is_monotonic:
    inversions = int(((dates.diff() < pd.Timedelta(0))).sum())
    return False, row_count, f"Non-monotonic dates ({inversions} inversions)", False
```

5) Unify filename/ID derivation (central helper)
- One function used by discovery/scraping/data_prep to derive `{brand}-{id}-{name}` safely; hash surrogate when missing.

```python
def build_watch_key(brand: str, watch_id: str | None, model_name: str) -> str:
    bid = watch_id if (watch_id and watch_id != 'unknown') else str(abs(hash(model_name)) % 10**9)
    return f"{make_filename_safe(brand)}-{make_filename_safe(bid)}-{make_filename_safe(model_name)}"
```

---

## Maintainability & Readability
- Split large modules and isolate side effects
  - Refactor `process.py` into cohesive modules; keep IO confined; make most functions pure (input df → output df).
- Type hints and docstrings everywhere
  - Add explicit return types and document invariants (e.g., sorted, no NaNs on essential columns).
- Standardize error handling
  - Prefer raising typed exceptions (e.g., `DatasetContractError`, `NavigationError`) over returning booleans; catch at orchestration boundaries.
- Logging consistency
  - Structured fields: always log `brand`, `watch_id`, `phase`; optionally provide a JSON logging formatter toggle.
- Configuration immutability
  - Convert Hydra configs to pydantic/dataclasses for runtime immutability within modules to avoid accidental mutation.

---

## Performance & Scalability
- Data prep feature generation
  - Avoid recomputing rolling min/max when used across features; compute once per window and reuse.
  - Default configs should enable only necessary features; expand selectively.
- IO formats
  - Allow Parquet for processed/unified datasets (column types preserved, faster IO) with a simple toggle.
- Scraping throughput and stability
  - Maintain a per-brand error budget and adaptive backoff; allow limited parallelism across brands (respecting robot policies).
  - Cache discovery results; keep a “seen targets” registry with timestamps to avoid redundant work.

---

## Guidelines for “Good Code” Here
- Readable
  - Small functions; explicit names; clear variable lifetimes. Keep logic linear and guard-clauses early.
- Maintainable
  - Single responsibility per module/function; centralized helpers for cross-cutting concerns (IDs, filenames, paths, browser options).
- Correct
  - Treat time as sacred; validate schemas/invariants at boundaries; avoid silent coercions (e.g., dates to numerics).
- Performant
  - Vectorize operations; reuse intermediate results; avoid redundant parsing; choose Parquet for heavy data.
- Idempotent & safe
  - Idempotent writes with backups (already present); deterministic outputs for the same inputs and config.

---

## Suggested Roadmap (Prioritized)
1) Time & Schema (High impact, low risk) — PARTIALLY DONE
- Done: explicit `timestamp` derivation, date exclusion, strict temporal splits, monotonic validation.
- Next: add duplicate `(asset_id, timestamp)` detection and a lightweight dataset contract validator.

2) Module Boundaries & Browser Unification (High impact, medium effort)
- Split `process.py` and unify browser/navigation logic; centralize filename/ID building.

3) Robust Extraction & Data Contracts (Medium impact, medium effort)
- Harden Chart.js extraction to numeric timestamps; add dataset validators pre-training and pre-save.

4) Performance & Formats (Medium impact, low-medium effort)
- Memoize rolling windows; Parquet toggle; adaptive backoff for scraping.

If you want, I can apply a surgical patch for (1) right away (temporal split + date exclusion + monotonic validation) to lock in correctness with minimal blast radius.
