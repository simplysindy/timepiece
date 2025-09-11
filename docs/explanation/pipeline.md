Purpose

  - Machine-readable map of the entire pipeline so tools and scripts can reason about it without scraping code or prose docs.
  - Acts as a single source of truth for entrypoints, configs, components, datasets, DAG edges, training setup, and artifact locations.

  What It Contains

  - entrypoints: How to run each stage (command + config file).
  - configs: Key config sections and important knobs to expect/validate.
  - components: Logical units (discovery, scraper, validator, processor, trainer) with inputs/outputs.
  - datasets: Produced/consumed files with paths, formats, and schemas.
  - flow: Edges describing the DAG from discovery through training.
  - training: Algorithms, metrics, split strategy, hyperparams, artifacts.

  Typical Uses

  - Generate docs/diagrams programmatically (e.g., build a DAG view or component inventory).
  - Pre-flight/CI checks (e.g., validate dataset paths exist, config keys are present).
  - Orchestration glue (construct commands with the right configs and toggles).
  - Tooling to list artifacts (models, metrics) and wire simple registries.
  - Onboarding aids: scripts can print “what to run” and “what gets produced”.

  How To Keep It Useful

  - Update when adding/removing components, datasets, or config keys.
  - Treat it as a contract between code and ops/docs automation.