# AC-Watches

Luxury watch data scraping and analysis platform for collecting price data from WatchCharts.com across 10+ premium brands.

## 🚀 Watch Scraping Pipeline

The core scraping pipeline is located in `src/scraping/` and provides a unified, configurable system for discovering, scraping, and validating watch price data.

### Quick Start

```bash
# Install dependencies
pip install hydra-core selenium pandas beautifulsoup4 webdriver-manager

# Run the complete pipeline
python -m src.scraping.scraping

# Run specific pipeline steps
python -m src.scraping.scraping pipeline.run_scraping=false pipeline.run_validation=false
```

## 📁 Project Structure

```
src/
└── scraping/
    ├── core/
    │   ├── base_scraper.py      # WatchScraper class with all scraping logic
    │   └── browser.py           # Browser management and driver creation
    ├── discovery.py             # Watch URL discovery from brand pages
    ├── validation.py            # CSV data validation and quality checks
    └── scraping.py             # Main entry point with Hydra integration

config/
└── scraping.yaml               # Pipeline configuration

data/
├── watches/                    # Scraped watch price data (CSV files)
└── watch_targets_100.json     # Discovered watch targets
```

## ⚙️ Configuration

The pipeline is configured via `config/scraping.yaml` with full Hydra support for CLI overrides:

### Pipeline Control
```yaml
pipeline:
  run_discovery: true    # Discover watch URLs from brand pages
  run_scraping: true     # Scrape price data from discovered watches  
  run_validation: true   # Validate scraped CSV files
```

### Discovery Settings
```yaml
discovery:
  target_count_per_brand: 10     # Watches to discover per brand
  delay_range: [3, 8]           # Random delay between requests (seconds)
  headless: true                # Run browser in headless mode
  output_file: "watch_targets_100.json"
```

### Scraping Settings
```yaml
scraping:
  delay_range: [10, 20]         # Random delay between scraping requests
  max_retries: 3                # Max retry attempts per URL
  brand_delay: 60               # Delay between different brands (seconds)
  output_dir: "data/watches"    # Output directory for CSV files
  headless: true                # Run browser in headless mode
```

### Validation Settings
```yaml
validation:
  data_dir: "data/watches"      # Directory containing CSV files to validate
  min_rows: 100                 # Minimum rows required for valid CSV
  move_invalid: false           # Move invalid files to kiv/ directory
  log_dir: "logs"              # Directory for validation logs
```

## 🎯 Supported Brands

The pipeline scrapes data from 10 luxury watch brands:

- **Top Tier**: Patek Philippe, Rolex, Audemars Piguet, Vacheron Constantin
- **Mid Tier**: Omega, Tudor, Hublot  
- **Entry Level**: Tissot, Longines, Seiko

## 🔧 Command Line Usage

### Basic Operations
```bash
# Run complete pipeline (discovery → scraping → validation)
python -m src.scraping.scraping

# Run only discovery phase
python -m src.scraping.scraping pipeline.run_scraping=false pipeline.run_validation=false

# Run only scraping phase  
python -m src.scraping.scraping pipeline.run_discovery=false pipeline.run_validation=false

# Run only validation phase
python -m src.scraping.scraping pipeline.run_discovery=false pipeline.run_scraping=false
```

### Configuration Overrides
```bash
# Customize scraping delays
python -m src.scraping.scraping scraping.delay_range=[5,15] scraping.brand_delay=30

# Run in visible browser mode
python -m src.scraping.scraping scraping.headless=false discovery.headless=false

# Change validation requirements
python -m src.scraping.scraping validation.min_rows=50 validation.move_invalid=true

# Discover fewer watches per brand
python -m src.scraping.scraping discovery.target_count_per_brand=5
```

## 📊 Output Files

### CSV Data Files
Individual watch price data saved as: `{Brand}-{Model}-{WatchID}.csv`
```
data/watches/
├── Rolex-Submariner_Date-126610LN.csv
├── Patek_Philippe-Nautilus-5711_1A_010.csv
└── ...
```

### Discovery Output
```json
{
  "brand": "Rolex",
  "model_name": "126610LN - Submariner Date",
  "url": "https://watchcharts.com/watch_model/126610LN-rolex-submariner-date/overview",
  "source": "generated"
}
```

## 🔍 Validation & Quality Control

The validation pipeline checks:
- **Minimum Data Points**: Ensures ≥100 rows of price data per watch
- **Date Ordering**: Validates chronological data sequence
- **File Integrity**: Checks for corrupted or empty CSV files
- **Data Quality**: Identifies watches needing re-scraping

### Validation Reports
- Console output with color-coded status
- Session logs saved to `logs/csv_validation_YYYYMMDD_HHMMSS/`
- Invalid file summaries and recommendations

## 🛠️ Development

### Core Components

**WatchScraper** (`src/scraping/core/base_scraper.py`)
- Consolidated scraping logic with Cloudflare bypass
- Chart.js price data extraction
- Incremental data updates
- Error handling and retry logic

**BrowserManager** (`src/scraping/core/browser.py`)  
- Chrome driver configuration with stealth capabilities
- Anti-detection settings for bypassing protection
- Separate drivers for discovery vs scraping tasks

**WatchDiscovery** (`src/scraping/discovery.py`)
- Brand page parsing and URL extraction
- Watch metadata cleaning and standardization
- Configurable discovery targets per brand

**WatchDataValidator** (`src/scraping/validation.py`)
- CSV quality assessment and reporting
- File management with optional invalid file quarantine
- Detailed validation metrics and summaries

## 📈 Performance

- **Stealth Browsing**: Anti-detection Chrome configuration
- **Rate Limiting**: Configurable delays to respect site limits  
- **Incremental Updates**: Only fetch new data points
- **Brand Batching**: Organized processing with inter-brand delays
- **Error Recovery**: Screenshot capture and retry mechanisms

## 🚨 Error Handling

- Cloudflare challenge detection and waiting
- Network timeout and retry logic
- Screenshot capture on scraping failures
- Comprehensive logging at all pipeline stages
- Graceful degradation when individual watches fail