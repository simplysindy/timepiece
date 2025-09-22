# Luxury Watch Price Prediction System

A complete end-to-end machine learning pipeline for predicting luxury watch prices. This system scrapes real-time data from WatchCharts.com, engineers 80+ features, and trains multiple forecasting models to predict price movements across premium watch brands like Rolex, Patek Philippe, and Audemars Piguet.

## Environment Setup
1. Create a virtual environment and activate it:
   ```
   python -m venv .venv && source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install hydra-core selenium pandas beautifulsoup4 webdriver-manager scipy scikit-learn streamlit plotly
   ```

## 🔧 Quick Start

Run the complete pipeline in sequence, or execute individual stages as needed.

### 1. Scrape Watch Data
```bash
python -m src.scraper.scraper
```
Discovers and scrapes price data from WatchCharts.com for 10+ luxury brands. Saves individual watch CSV files under `data/watches/`.

### 2. Prepare ML Dataset
```bash
python -m src.data_prep.data_prep
```
Processes raw data into ML-ready format with 80+ engineered features. Outputs to `data/processed/`.

### 3. Train Prediction Models
```bash
python -m src.training.training
```
Trains multiple algorithms on temporal splits. Models and predictions saved to `data/output/models/`.

### 4. Generate Predictions
```bash
python -m src.inference.inference
```
Uses trained models to generate forward-looking price predictions.

### 5. Interactive Dashboard
```bash
streamlit run src/inference/streamlit_app.py
```
Launch web interface to explore predictions, visualize trends, and analyze model performance.

## ⚙️ Configuration

All pipelines use Hydra for configuration management. Customize behavior via CLI overrides:

```bash
# Run scraping with custom settings
python -m src.scraper.scraper scraping.headless=false discovery.target_count_per_brand=5

# Train only specific models and horizons
python -m src.training.training training.models=["lightgbm","xgboost"] training.horizons=[1,7]

# Customize feature engineering
python -m src.data_prep.data_prep features.include_technical=false processing.outlier_method=zscore
```

Default configurations are in `conf/*.yaml` files.

## 📊 What You Get

- **Raw Data**: Individual watch price histories (`data/watches/*.csv`)
- **Processed Dataset**: ML-ready features (`data/processed/*.csv`)
- **Trained Models**: Serialized models for each algorithm (`data/output/models/`)
- **Predictions**: Forward-looking price forecasts (`data/output/predictions.csv`)
- **Interactive Dashboard**: Web interface for data exploration

## 🎯 Supported Brands

**Premium Tier**: Patek Philippe, Rolex, Audemars Piguet, Vacheron Constantin
**Mid Tier**: Omega, Tudor, Hublot
**Entry Tier**: Tissot, Longines, Seiko

## 🧠 Technical Highlights

- **Anti-Detection Web Scraping**: Bypasses Cloudflare protection with stealth browser automation
- **Temporal Data Splits**: Proper train/validation/test splits respecting time series nature
- **Rich Feature Engineering**: Price momentum, technical indicators, luxury tiers, seasonality
- **Multi-Algorithm Support**: Tree-based, linear, and neural network models
- **Production Architecture**: Modular design, comprehensive logging, error recovery

## 🚀 Use Cases

- **Investment Research**: Analyze luxury watch market trends and price patterns
- **Portfolio Management**: Predict price movements for watch collecting strategies
- **Market Analysis**: Understand brand performance and seasonal effects
- **Academic Research**: Study luxury goods pricing dynamics
- **ML Learning**: Example of end-to-end time series prediction pipeline

## 📋 Project Structure

```
src/
├── scraper/        # Web scraping and data collection
├── data_prep/      # Feature engineering and data preparation
├── training/       # Model training and evaluation
├── inference/      # Prediction generation and web dashboard
└── utils/          # Shared utilities

conf/               # Hydra configuration files
data/               # Data storage (watches/, processed/, output/)
```
