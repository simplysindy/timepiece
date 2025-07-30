# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Primary Development Commands
```bash
# Run complete scraping pipeline (main entry point)
python -m src.scraping.scraping

# Run data preparation pipeline (simplified watch-only version)
python -m src.data_prep.data_prep

# Install required dependencies
pip install hydra-core selenium pandas beautifulsoup4 webdriver-manager scipy scikit-learn

# Run specific pipeline phases
python -m src.scraping.scraping pipeline.run_discovery=false pipeline.run_validation=false  # scraping only
python -m src.scraping.scraping pipeline.run_scraping=false pipeline.run_validation=false   # discovery only
python -m src.scraping.scraping pipeline.run_discovery=false pipeline.run_scraping=false    # validation only
```

### Configuration Override Examples
```bash
# Customize scraping behavior
python -m src.scraping.scraping scraping.delay_range=[5,15] validation.min_rows=50

# Run in visible browser mode (for debugging)
python -m src.scraping.scraping scraping.headless=false discovery.headless=false

# Customize data preparation pipeline
python -m src.data_prep.data_prep data.max_files=10
python -m src.data_prep.data_prep processing.interpolation_method=linear processing.outlier_method=zscore
python -m src.data_prep.data_prep features.include_technical=false features.lag_periods=[1,3,7]
```

### Linting and Testing
- No formal test suite or linting tools are configured in this project
- When making changes, test functionality by running the scraping pipeline with limited scope
- Always ask user for specific linting/testing commands if needed

## Project Architecture

### High-Level Design
This is a **luxury watch price data scraping system** that collects historical pricing data from WatchCharts.com for 10+ premium watch brands. The architecture follows a three-phase pipeline design:

1. **Discovery Phase**: Crawls brand pages to find individual watch URLs
2. **Scraping Phase**: Extracts price chart data from each watch page
3. **Validation Phase**: Ensures data quality and completeness

### Core Components

**Main Entry Point** (`src/scraping/scraping.py`)
- Hydra-based configuration management with CLI overrides
- Orchestrates all three pipeline phases
- Single command execution: `python -m src.scraping.scraping`

**WatchScraper** (`src/scraping/core/base_scraper.py`)
- Consolidated scraping logic with Cloudflare bypass capabilities
- Extracts Chart.js price data using Selenium
- Handles incremental updates and error recovery
- Anti-detection browser configuration

**BrowserManager** (`src/scraping/core/browser.py`)
- Chrome WebDriver management with stealth capabilities
- Cloudflare challenge detection and waiting
- Separate driver instances for discovery vs scraping

**WatchDiscovery** (`src/scraping/discovery.py`)
- Parses brand listing pages to discover individual watch URLs
- Extracts watch metadata and creates target lists
- Configurable discovery limits per brand

**WatchDataValidator** (`src/scraping/validation.py`)
- CSV file quality assessment and reporting
- Validates minimum data requirements (default: 100+ rows)
- Optional invalid file quarantine system

### Configuration System
- **Central config**: `config/scraping.yaml` (Hydra-managed)
- **CLI overrides**: `section.key=value` syntax
- **Pipeline control**: Enable/disable individual phases
- **Rate limiting**: Configurable delays between requests
- **Brand targeting**: 10 luxury watch brands pre-configured

### Key Design Principles
- **Single entry point**: All functionality accessible via one command
- **No abstract classes**: Simplified inheritance hierarchy
- **Consolidated functionality**: WatchScraper handles all scraping logic
- **Hydra integration**: Full CLI configuration override support
- **Anti-detection**: Stealth browsing to bypass site protections

### Data Flow
```
Brand URLs → Discovery → data/targets/watch_targets.json → Scraping → data/watches/*.csv → Data Prep → data/processed/*.csv
```

### Output Formats
- **CSV files**: `{Brand}-{Model}-{WatchID}.csv` (individual watch price data)
- **JSON targets**: `data/targets/watch_targets.json` (discovered watch URLs)
- **Validation logs**: `logs/csv_validation_YYYYMMDD_HHMMSS/`
- **Processed data**: `data/processed/watch_data_processed.csv` (ML-ready dataset)

## Data Preparation Pipeline

### Simplified Architecture (New)
The data preparation pipeline has been refactored into a simplified, watch-only focused system:

```
src/
├── data_prep/
│   ├── __init__.py
│   ├── data_prep.py    # Main Hydra entry point
│   ├── process.py      # Consolidated processing logic
│   └── config.py       # Hydra configuration schemas
├── utils/
│   ├── __init__.py
│   └── io.py           # Simple I/O utilities
conf/
└── data_prep.yaml      # Hydra configuration
```

**Key Improvements:**
- **Simplified Structure**: Consolidated from 2,500+ lines across multiple files to ~600 lines in 2 main files
- **Hydra Integration**: Full configuration management with CLI overrides
- **Watch-Only Focus**: Removed multi-asset complexity, optimized for luxury watch data
- **All Features Preserved**: Maintains all 80+ features including watch-specific luxury market indicators

**Data Preparation Components:**

**Main Entry Point** (`src/data_prep/data_prep.py`)
- Hydra-based configuration management
- Single command execution: `python -m src.data_prep.data_prep`
- Comprehensive logging and progress reporting

**WatchDataProcessor** (`src/data_prep/process.py`)
- All-in-one data processing: loading, cleaning, feature engineering
- Watch-specific validation and luxury tier classification
- Comprehensive feature engineering (80+ features)
- Output generation: combined dataset + metadata summary

**Configuration Schema** (`src/data_prep/config.py`)
- Type-safe configuration using dataclasses
- Hydra integration for CLI overrides
- Watch-specific luxury tier and brand tier definitions

## Important Implementation Notes

### Cloudflare Handling
The scraper includes sophisticated Cloudflare bypass logic. When working with browser automation:
- Always use the BrowserManager for driver creation
- Check for Cloudflare challenges before proceeding with scraping
- Implement appropriate delays and retry logic

### Rate Limiting Strategy
- Configurable delays between requests (default: 10-20 seconds)
- Brand-level delays (default: 60 seconds between brands)
- Randomized timing to avoid detection patterns

### Error Recovery
- Screenshot capture on scraping failures
- Configurable retry attempts (default: 3 retries)
- Graceful degradation when individual watches fail
- Comprehensive logging at all pipeline stages

## Git Commit Preferences

### Commit Structure
- **One commit per logical change**: Each commit should focus on a single, cohesive modification
- **Multiple commits for large changes**: Break down complex features into logical, atomic commits
- **No co-author details**: Do not add "Co-Authored-By" lines in commit messages

### Commit Message Format
```
Brief descriptive title (50 chars or less)

- Detailed bullet points explaining what was added/changed
- Each bullet point should be specific and actionable
- Use present tense ("Add" not "Added")
- Focus on what and why, not how
```

## Development Workflow

When making changes to this codebase:
1. Test changes using the main pipeline command with limited scope
2. Use configuration overrides to test specific functionality
3. Verify CSV output format and validation logic
4. Check browser automation still bypasses anti-bot measures
5. Update this CLAUDE.md if adding new commands or changing architecture