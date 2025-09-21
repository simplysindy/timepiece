"""Minimal Streamlit front end for luxury watch price forecasts."""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List

import re

import altair as alt
import pandas as pd
import streamlit as st
from omegaconf import OmegaConf

# Ensure project root is available on the Python path when launched via Streamlit.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.inference.inference import generate_predictions  # noqa: E402


ALLOWED_MODELS = {"ridge", "lightgbm", "xgboost", "linear", "random_forest"}
MODEL_DISPLAY_NAMES = {
    "ridge": "ridge",
    "lightgbm": "lightgbm",
    "xgboost": "xgb",
    "linear": "linear",
    "random_forest": "rf",
}
MODEL_ORDER = ["ridge", "lightgbm", "xgboost", "linear", "random_forest"]


st.set_page_config(page_title="Luxury Watch Forecast", layout="wide")

CONFIG_PATH = PROJECT_ROOT / "conf" / "inference.yaml"


def resolve_path(path_str: str) -> Path:
    """Return an absolute path for the given repository-relative string."""
    path = Path(path_str)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


@lru_cache(maxsize=1)
def load_default_config() -> dict:
    """Load defaults from conf/inference.yaml if available."""
    if CONFIG_PATH.exists():
        cfg = OmegaConf.load(CONFIG_PATH)
        data_cfg = cfg.get("data", {})
        inference_cfg = cfg.get("inference", {})
        return {
            "data_path": data_cfg.get("path", "data/output/featured_data.csv"),
            "asset_column": data_cfg.get("asset_column", "watch_id"),
            "timestamp_column": data_cfg.get("timestamp_column", "date"),
            "target_column": data_cfg.get("target_column", "target"),
            "model_dir": inference_cfg.get("model_dir", "data/output/models"),
            "prediction_path": inference_cfg.get(
                "output_path", "data/output/models/forward_predictions.csv"
            ),
            "models": tuple(inference_cfg.get("models", ["ridge"])),
            "horizons": tuple(inference_cfg.get("prediction_horizon_days", [7])),
            "include_actuals": bool(inference_cfg.get("include_actuals", True)),
        }
    # Fallback defaults when config is missing.
    return {
        "data_path": "data/output/featured_data.csv",
        "asset_column": "watch_id",
        "timestamp_column": "date",
        "target_column": "target",
        "model_dir": "data/output/models",
        "prediction_path": "data/output/models/forward_predictions.csv",
        "models": ("ridge",),
        "horizons": (7,),
        "include_actuals": True,
    }


@st.cache_data(show_spinner=False)
def load_feature_data(data_path: str) -> pd.DataFrame:
    """Read the feature dataset used for training/inference."""
    resolved_path = resolve_path(data_path)
    frame = pd.read_csv(resolved_path)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame


@st.cache_data(show_spinner=False)
def load_predictions_file(predictions_path: str) -> pd.DataFrame:
    """Load previously generated predictions from disk."""
    resolved_path = resolve_path(predictions_path)
    frame = pd.read_csv(resolved_path)
    missing = {"model_name", "asset_id", "prediction"} - set(frame.columns)
    if missing:
        raise ValueError(
            "Predictions file missing required columns: " + ", ".join(sorted(missing))
        )
    for column in ("feature_date", "prediction_date"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column])
    return frame


@st.cache_data(show_spinner=True)
def compute_predictions(
    data_path: str,
    model_dir: str,
    models: Iterable[str],
    horizons: Iterable[int],
    asset_column: str,
    timestamp_column: str,
    target_column: str,
    include_actuals: bool,
) -> pd.DataFrame:
    """Run model inference and cache the resulting tidy frame."""
    return generate_predictions(
        data_path=str(resolve_path(data_path)),
        model_dir=str(resolve_path(model_dir)),
        models=list(models),
        horizons=list(horizons),
        asset_column=asset_column,
        timestamp_column=timestamp_column,
        target_column=target_column,
        include_actuals=include_actuals,
    )


@lru_cache(maxsize=1)
def available_models(model_dir: str) -> List[str]:
    """List pickled model files located in the configured directory."""
    model_path = resolve_path(model_dir)
    if not model_path.exists():
        return []

    candidates = set()
    for path in model_path.glob("*.pkl"):
        stem = path.stem
        base = re.sub(r"(__h|_h)\d+$", "", stem)
        if base in ALLOWED_MODELS:
            candidates.add(base)

    return sorted(candidates, key=MODEL_ORDER.index)


def format_currency(value: float | None) -> str:
    if value is None or pd.isna(value):  # type: ignore[arg-type]
        return "N/A"
    return f"S${value:,.2f}"


def format_delta(value: float | None) -> str:
    if value is None or pd.isna(value):  # type: ignore[arg-type]
        return "N/A"
    sign = "+" if value >= 0 else "-"
    return f"{sign}S${abs(value):,.2f}"


def prettify_asset_id(asset_id: str, brand_lookup: dict[str, str]) -> str:
    """Return a human-friendly asset label for display tables."""
    if not isinstance(asset_id, str):
        return str(asset_id)

    raw_parts = [part for part in asset_id.split("_") if part]
    if not raw_parts:
        return asset_id

    reference = raw_parts[-1]
    core_parts = raw_parts[:-1]

    brand = brand_lookup.get(asset_id, "").strip()
    if brand:
        brand_tokens = {token.lower() for token in re.findall(r"[A-Za-z0-9]+", brand)}
        trimmed_parts = [part for part in core_parts if part.lower() not in brand_tokens]
        if trimmed_parts:
            core_parts = trimmed_parts

    core_text = " ".join(core_parts).strip()
    reference_text = reference.replace("_", " ").strip()

    label_chunks = [chunk for chunk in (brand, core_text) if chunk]
    label = " ".join(label_chunks).strip()

    if reference_text and reference_text.lower() not in label.lower():
        label = f"{label} ({reference_text})" if label else reference_text

    return label or asset_id.replace("_", " ")


st.title("Luxury Watch Forecasting")
st.caption("Minimal dashboard for exploring next-week price predictions across premium watches.")

defaults = load_default_config()

model_dir = defaults["model_dir"]
predictions_path = defaults["prediction_path"]

detected_models = available_models(model_dir)
model_options = detected_models or MODEL_ORDER
default_models = [m for m in defaults["models"] if m in model_options]
if not default_models and model_options:
    default_models = [model_options[0]]

models = default_models


asset_brand_lookup: dict[str, str] = {}

try:
    feature_data = load_feature_data(defaults["data_path"])
except Exception as exc:  # pragma: no cover - defensive UI handling
    st.error(f"Unable to load feature dataset: {exc}")
    feature_data = pd.DataFrame()

if not feature_data.empty:
    asset_column = defaults.get("asset_column", "watch_id")
    if asset_column in feature_data.columns and "brand" in feature_data.columns:
        brand_pairs = (
            feature_data[[asset_column, "brand"]]
            .dropna()
            .drop_duplicates(subset=[asset_column])
        )
        asset_brand_lookup = dict(zip(brand_pairs[asset_column], brand_pairs["brand"]))

predictions = pd.DataFrame()
predictions_source: str | None = None

predictions_path = predictions_path.strip()
if predictions_path:
    try:
        predictions = load_predictions_file(predictions_path)
        predictions_source = f"Loaded predictions from {predictions_path}"
    except FileNotFoundError as exc:
        st.warning(f"Predictions file not found: {exc}")
    except Exception as exc:  # pragma: no cover - defensive UI handling
        st.error(f"Unable to load predictions file: {exc}")

if predictions.empty:
    if not models:
        st.warning("Select at least one model to generate predictions.")
    else:
        try:
            predictions = compute_predictions(
                data_path=defaults["data_path"],
                model_dir=model_dir,
                models=tuple(models),
                horizons=defaults["horizons"],
                asset_column=defaults["asset_column"],
                timestamp_column=defaults["timestamp_column"],
                target_column=defaults["target_column"],
                include_actuals=defaults["include_actuals"],
            )
            predictions_source = "Generated predictions from models"
        except FileNotFoundError as exc:  # pragma: no cover - defensive UI handling
            st.error(f"Prediction inputs missing: {exc}")
        except Exception as exc:  # pragma: no cover - defensive UI handling
            st.error(f"Prediction generation failed: {exc}")

if predictions.empty:
    st.info("No predictions ready yet. Adjust the configuration or ensure models are available.")
    st.stop()

for column in ("feature_date", "prediction_date"):
    if column in predictions.columns:
        predictions[column] = pd.to_datetime(predictions[column])

if models:
    predictions = predictions[predictions["model_name"].isin(models)]


if predictions_source:
    st.caption(predictions_source)

if predictions.empty:
    st.info("No predictions for the selected models and horizons.")
    st.stop()

predictions = predictions.sort_values(["asset_id", "model_name", "horizon_days"]).reset_index(drop=True)

asset_options = sorted(predictions["asset_id"].unique())
selected_asset = st.selectbox("Select a watch", asset_options)
asset_rows = predictions[predictions["asset_id"] == selected_asset]

horizon_choices = sorted(asset_rows["horizon_days"].unique())
selected_horizon = horizon_choices[-1] if horizon_choices else 7

model_choices = sorted(
    asset_rows["model_name"].unique(),
    key=lambda name: (MODEL_ORDER.index(name) if name in MODEL_ORDER else len(MODEL_ORDER), name),
)
selected_model = model_choices[0] if model_choices else "ridge"

selected_row = asset_rows[
    (asset_rows["model_name"] == selected_model)
    & (asset_rows["horizon_days"] == selected_horizon)
].iloc[0]

left, middle, right = st.columns(3)
prediction_value = selected_row.get("prediction")
current_price = selected_row.get("current_price")
actual_value = selected_row.get("actual_target")

delta_value = None
delta_percent = None
if prediction_value is not None and pd.notna(prediction_value) and current_price is not None and pd.notna(current_price):
    try:
        delta_value = float(prediction_value) - float(current_price)
        if float(current_price) != 0:
            delta_percent = delta_value / float(current_price) * 100
    except (TypeError, ValueError):
        delta_value = None
        delta_percent = None

with left:
    left.metric("Current price", format_currency(current_price))

with middle:
    delta_label = None
    if delta_value is not None:
        delta_label = format_delta(delta_value)
        if delta_percent is not None:
            delta_label = f"{delta_label} ({delta_percent:+.2f}%)"
    middle.metric(
        "Predicted price",
        format_currency(prediction_value),
        delta=delta_label,
        delta_color="inverse" if delta_value is not None and delta_value < 0 else "normal",
    )

with right:
    # For forward predictions, we don't have actual values
    # Show prediction horizon instead
    right.metric(
        f"Horizon",
        f"{selected_horizon} days"
    )

st.markdown("---")

st.subheader("Price history & forecast")
if feature_data.empty:
    st.write("Feature dataset unavailable; cannot plot historical prices.")
else:
    history = feature_data[feature_data[defaults["asset_column"]] == selected_asset][
        [defaults["timestamp_column"], "price(SGD)"]
    ].rename(
        columns={
            defaults["timestamp_column"]: "date",
            "price(SGD)": "price",
        }
    )
    history = history.sort_values("date").tail(180)

    forecast_rows = (
        asset_rows[asset_rows["model_name"] == selected_model][
            ["prediction_date", "prediction"]
        ]
        .dropna(subset=["prediction"])
        .rename(columns={"prediction_date": "date", "prediction": "price"})
    )
    forecast_rows = forecast_rows.sort_values("date")

    history_chart = (
        alt.Chart(history)
        .mark_line(color="#1f77b4", strokeWidth=2)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("price:Q", title="Price (SGD)", scale=alt.Scale(zero=False)),
            tooltip=["date:T", alt.Tooltip("price:Q", format=",.2f")],
        )
    )

    combined_chart = history_chart
    if not forecast_rows.empty:
        forecast_chart = (
            alt.Chart(forecast_rows)
            .mark_line(color="#ff7f0e", strokeDash=[2, 2], strokeWidth=2)
            .encode(
                x="date:T",
                y=alt.Y("price:Q", scale=alt.Scale(zero=False)),
                tooltip=["date:T", alt.Tooltip("price:Q", format=",.2f")],
            )
        )
        combined_chart = combined_chart + forecast_chart

    st.altair_chart(combined_chart, use_container_width=True)

st.markdown("---")
st.subheader("All predictions")

display_predictions = predictions.copy()
if "model_name" in display_predictions.columns:
    display_predictions = display_predictions.drop(columns=["model_name"])

if "asset_id" in display_predictions.columns:
    display_predictions["asset_id"] = display_predictions["asset_id"].map(
        lambda value: prettify_asset_id(value, asset_brand_lookup)
    )

for column in ("feature_date", "prediction_date"):
    if column in display_predictions.columns:
        dates = pd.to_datetime(display_predictions[column], errors="coerce")
        display_predictions[column] = dates.dt.date.astype(str)
        display_predictions.loc[dates.isna(), column] = ""

if "prediction" in display_predictions.columns:
    display_predictions = display_predictions.rename(
        columns={"prediction": "prediction_price"}
    )

price_columns = ["current_price", "prediction_price"]
if any(col in display_predictions.columns for col in price_columns):
    tail_columns = [col for col in display_predictions.columns if col not in price_columns]
    if "current_price" in display_predictions.columns:
        tail_columns.append("current_price")
    if "prediction_price" in display_predictions.columns:
        tail_columns.append("prediction_price")
    display_predictions = display_predictions[tail_columns]

st.dataframe(display_predictions, hide_index=True)
