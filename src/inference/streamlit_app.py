"""Minimal Streamlit front end for luxury watch price forecasts."""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import os
import re

import altair as alt
import pandas as pd
import requests
import streamlit as st
from omegaconf import DictConfig, OmegaConf

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


def get_cloud_api_base_url() -> Optional[str]:
    """Return the configured cloud API URL from environment variables or secrets."""
    env_keys = ("GCP_API_URL", "WATCH_API_URL", "CLOUD_FUNCTION_URL", "CLOUD_API_URL")
    for key in env_keys:
        value = os.getenv(key)
        if value:
            return value.rstrip("/")

    try:  # Optional Streamlit secrets support during deployment
        secret_url = st.secrets.get("cloud_api_url")  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - Streamlit secrets not available locally
        secret_url = None

    if secret_url:
        return str(secret_url).rstrip("/")

    # Default to production API URL
    return "https://timepiece-api-zmlm4rlafq-uc.a.run.app"


def _request_cloud_json(api_base_url: str, method: str, path: str, **kwargs: Any) -> Dict[str, Any]:
    """Perform an HTTP request to the cloud API and return JSON content."""
    url = f"{api_base_url.rstrip('/')}{path}"
    response = requests.request(method, url, timeout=30, **kwargs)
    response.raise_for_status()
    try:
        return response.json()
    except ValueError as exc:
        raise ValueError(f"Cloud API returned non-JSON payload for {url}") from exc


@st.cache_data(show_spinner=False)
def fetch_cloud_watches(api_base_url: str) -> List[str]:
    """Fetch the list of watch identifiers from the cloud API."""
    payload = _request_cloud_json(api_base_url, "GET", "/watches")
    watches = payload.get("watches", [])
    if not isinstance(watches, list):
        raise ValueError("Unexpected response format from /watches")
    return [str(item) for item in watches]


@st.cache_data(show_spinner=False)
def fetch_cloud_models(api_base_url: str) -> List[str]:
    """Fetch the list of supported models from the cloud API."""
    payload = _request_cloud_json(api_base_url, "GET", "/models")
    models = payload.get("models", [])
    if not isinstance(models, list):
        raise ValueError("Unexpected response format from /models")
    return [str(item) for item in models]


@st.cache_data(show_spinner=True)
def fetch_cloud_prediction(
    api_base_url: str,
    watch_id: str,
    model_name: str,
    horizon: int,
) -> Dict[str, Any]:
    """Request a single prediction from the cloud API."""
    payload = _request_cloud_json(
        api_base_url,
        "POST",
        "/predict",
        json={
            "watch_id": watch_id,
            "model_name": model_name,
            "horizon": horizon,
        },
    )
    return payload


@st.cache_data(show_spinner=True)
def fetch_cloud_predictions_batch(
    api_base_url: str,
    watch_id: str,
    model_name: str,
    horizons: List[int],
) -> List[Dict[str, Any]]:
    """Request multiple predictions for different horizons from the cloud API."""
    predictions = []
    for horizon in horizons:
        try:
            prediction = fetch_cloud_prediction(api_base_url, watch_id, model_name, horizon)
            prediction["horizon"] = horizon  # Ensure horizon is in the response
            predictions.append(prediction)
        except Exception as exc:
            st.warning(f"Failed to fetch prediction for horizon {horizon}: {exc}")
            continue
    return predictions


def load_feature_dataset_with_lookup(
    data_path: str,
    asset_column: str,
) -> tuple[pd.DataFrame, Dict[str, str], Optional[str]]:
    """Load feature data along with a brand lookup map and optional error message."""
    try:
        frame = load_feature_data(data_path)
    except FileNotFoundError as exc:
        return pd.DataFrame(), {}, f"Feature dataset not found: {exc}"
    except Exception as exc:  # pragma: no cover - defensive UI handling
        return pd.DataFrame(), {}, f"Unable to load feature dataset: {exc}"

    lookup: Dict[str, str] = {}
    if asset_column in frame.columns and "brand" in frame.columns:
        brand_pairs = (
            frame[[asset_column, "brand"]]
            .dropna()
            .drop_duplicates(subset=[asset_column])
        )
        lookup = dict(zip(brand_pairs[asset_column], brand_pairs["brand"]))

    return frame, lookup, None


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
        if not isinstance(cfg, DictConfig):
            raise TypeError("Expected inference config to be a DictConfig")
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


def run_local_ui(defaults: dict) -> None:
    """Render the local inference experience using on-disk models."""
    model_dir = defaults.get("model_dir", "data/output/models")
    predictions_path = defaults.get("prediction_path", "")

    detected_models = available_models(model_dir)
    model_options = detected_models or MODEL_ORDER
    default_models = [m for m in defaults.get("models", ()) if m in model_options]
    if not default_models and model_options:
        default_models = [model_options[0]]

    models = default_models

    feature_data, asset_brand_lookup, feature_error = load_feature_dataset_with_lookup(
        defaults.get("data_path", "data/output/featured_data.csv"),
        defaults.get("asset_column", "watch_id"),
    )
    if feature_error:
        st.error(feature_error)
        feature_data = pd.DataFrame()
        asset_brand_lookup = {}

    predictions = pd.DataFrame()
    predictions_source: Optional[str] = None

    predictions_path_clean = str(predictions_path).strip()
    if predictions_path_clean:
        try:
            predictions = load_predictions_file(predictions_path_clean)
            predictions_source = f"Loaded predictions from {predictions_path_clean}"
        except FileNotFoundError as exc:
            st.warning(f"Predictions file not found: {exc}")
        except Exception as exc:  # pragma: no cover - defensive UI handling
            st.error(f"Unable to load predictions file: {exc}")

    if predictions.empty:
        if not models:
            st.warning("Select at least one model to generate predictions.")
            return
        try:
            predictions = compute_predictions(
                data_path=defaults.get("data_path", "data/output/featured_data.csv"),
                model_dir=model_dir,
                models=tuple(models),
                horizons=defaults.get("horizons", (7,)),
                asset_column=defaults.get("asset_column", "watch_id"),
                timestamp_column=defaults.get("timestamp_column", "date"),
                target_column=defaults.get("target_column", "target"),
                include_actuals=bool(defaults.get("include_actuals", True)),
            )
            predictions_source = "Generated predictions from models"
        except FileNotFoundError as exc:  # pragma: no cover - defensive UI handling
            st.error(f"Prediction inputs missing: {exc}")
            return
        except Exception as exc:  # pragma: no cover - defensive UI handling
            st.error(f"Prediction generation failed: {exc}")
            return

    if predictions.empty:
        st.info("No predictions ready yet. Adjust the configuration or ensure models are available.")
        return

    for column in ("feature_date", "prediction_date"):
        if column in predictions.columns:
            predictions[column] = pd.to_datetime(predictions[column])

    if models:
        predictions = predictions[predictions["model_name"].isin(models)]

    if predictions_source:
        st.caption(predictions_source)

    if predictions.empty:
        st.info("No predictions for the selected models and horizons.")
        return

    predictions = predictions.sort_values(["asset_id", "model_name", "horizon_days"]).reset_index(drop=True)

    asset_options = sorted(predictions["asset_id"].unique())
    if not asset_options:
        st.info("No predictions available for display.")
        return

    selected_asset = st.selectbox("Select a watch", asset_options)
    asset_rows = predictions[predictions["asset_id"] == selected_asset]
    if asset_rows.empty:
        st.warning("No prediction rows for the selected watch.")
        return

    horizon_choices = sorted(asset_rows["horizon_days"].unique())
    selected_horizon = horizon_choices[-1] if horizon_choices else 7

    model_choices = sorted(
        asset_rows["model_name"].unique(),
        key=lambda name: (
            MODEL_ORDER.index(name) if name in MODEL_ORDER else len(MODEL_ORDER),
            name,
        ),
    )
    selected_model = model_choices[0] if model_choices else "ridge"

    selection_mask = (
        (asset_rows["model_name"] == selected_model)
        & (asset_rows["horizon_days"] == selected_horizon)
    )
    if not selection_mask.any():
        st.warning("Unable to locate prediction for the selected configuration.")
        return

    selected_row = asset_rows.loc[selection_mask].iloc[0]

    left, middle, right = st.columns(3)
    prediction_value = selected_row.get("prediction")
    current_price = selected_row.get("current_price")

    delta_value = None
    delta_percent = None
    if (
        prediction_value is not None
        and pd.notna(prediction_value)
        and current_price is not None
        and pd.notna(current_price)
    ):
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
        right.metric("Horizon", f"{selected_horizon} days")

    st.markdown("---")
    st.subheader("Price history & forecast")

    timestamp_column = defaults.get("timestamp_column", "date")
    asset_column = defaults.get("asset_column", "watch_id")
    if feature_data.empty:
        st.write("Feature dataset unavailable; cannot plot historical prices.")
    else:
        history = (
            feature_data[feature_data[asset_column] == selected_asset][
                [timestamp_column, "price(SGD)"]
            ]
            .rename(columns={timestamp_column: "date", "price(SGD)": "price"})
        )
        history = history.sort_values("date").tail(180)

        forecast_rows = (
            asset_rows[asset_rows["model_name"] == selected_model][
                ["prediction_date", "prediction"]
            ]
            .dropna(subset=["prediction"])
            .rename(columns={"prediction_date": "date", "prediction": "price"})
            .sort_values("date")
        )

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


def run_cloud_ui(defaults: dict, api_base_url: str) -> None:
    """Render the cloud-backed inference experience backed by the deployed API."""
    asset_column = defaults.get("asset_column", "watch_id")
    timestamp_column = defaults.get("timestamp_column", "date")

    feature_data, asset_brand_lookup, feature_error = load_feature_dataset_with_lookup(
        defaults.get("data_path", "data/output/featured_data.csv"),
        asset_column,
    )
    if feature_error:
        st.warning(feature_error)
        feature_data = pd.DataFrame()
        asset_brand_lookup = {}

    try:
        watch_options = sorted(fetch_cloud_watches(api_base_url))
    except Exception as exc:  # pragma: no cover - defensive UI handling
        st.error(f"Unable to load watch catalog from cloud API: {exc}")
        return

    if not watch_options:
        st.info("Cloud API returned no watches. Confirm that the deployment is initialized.")
        return

    try:
        model_options = fetch_cloud_models(api_base_url)
    except Exception as exc:  # pragma: no cover - defensive UI handling
        st.error(f"Unable to load model catalog from cloud API: {exc}")
        model_options = []

    if not model_options:
        model_options = MODEL_ORDER

    model_labels = [MODEL_DISPLAY_NAMES.get(name, name) for name in model_options]
    label_to_model = {MODEL_DISPLAY_NAMES.get(name, name): name for name in model_options}

    selected_watch = st.selectbox("Select a watch", watch_options)

    default_model_label = MODEL_DISPLAY_NAMES.get(model_options[0], model_options[0])
    default_index = model_labels.index(default_model_label) if default_model_label in model_labels else 0
    selected_model_label = st.selectbox("Model", model_labels, index=default_index)
    selected_model = label_to_model[selected_model_label]

    if not selected_watch:
        st.info("Choose a watch to fetch predictions.")
        return

    # Fetch predictions for all horizons D1-D7
    horizons = list(range(1, 8))  # [1, 2, 3, 4, 5, 6, 7]

    try:
        predictions = fetch_cloud_predictions_batch(
            api_base_url=api_base_url,
            watch_id=selected_watch,
            model_name=selected_model,
            horizons=horizons,
        )
    except Exception as exc:  # pragma: no cover - defensive UI handling
        st.error(f"Failed to fetch predictions: {exc}")
        return

    if not predictions:
        st.error("No predictions could be retrieved.")
        return

    # Get current price from the first prediction or feature data
    current_price = predictions[0].get("current_price")
    feature_date = None
    feature_date_raw = predictions[0].get("feature_date")
    if feature_date_raw:
        feature_date = pd.to_datetime(feature_date_raw)

    if current_price is None and not feature_data.empty and "price(SGD)" in feature_data.columns:
        latest_row = (
            feature_data[feature_data[asset_column] == selected_watch]
            .sort_values(timestamp_column)
            .tail(1)
        )
        if not latest_row.empty:
            current_price = float(latest_row["price(SGD)"].iloc[0])
            if feature_date is None:
                feature_date = latest_row[timestamp_column].iloc[0]

    # Display current price
    st.metric("Current Price", format_currency(current_price))
    st.caption(f"Live inference via {api_base_url}")

    st.markdown("---")
    st.subheader("Price history & D1-D7 forecast")

    if feature_data.empty:
        st.write("Feature dataset unavailable; cannot plot historical prices.")
    else:
        history = (
            feature_data[feature_data[asset_column] == selected_watch][
                [timestamp_column, "price(SGD)"]
            ]
            .rename(columns={timestamp_column: "date", "price(SGD)": "price"})
        )
        history = history.sort_values("date").tail(180)

        # Create forecast data for all horizons
        forecast_rows_data = []

        # Add current price as starting point
        if feature_date is not None and current_price is not None:
            forecast_rows_data.append({
                "date": feature_date,
                "price": current_price,
                "horizon": "Current"
            })

        # Add all D1-D7 predictions
        for pred in predictions:
            prediction_date_raw = pred.get("prediction_date")
            prediction_value = pred.get("prediction")
            horizon = pred.get("horizon") or pred.get("horizon_days", 0)

            if prediction_date_raw and prediction_value is not None:
                prediction_date = pd.to_datetime(prediction_date_raw)
                forecast_rows_data.append({
                    "date": prediction_date,
                    "price": prediction_value,
                    "horizon": f"D{horizon}"
                })

        forecast_rows = pd.DataFrame(forecast_rows_data)

        # Create base history chart
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
            # Forecast line with dotted style
            forecast_chart = (
                alt.Chart(forecast_rows)
                .mark_line(color="#ff7f0e", strokeDash=[5, 5], strokeWidth=2)
                .encode(
                    x=alt.X("date:T"),
                    y=alt.Y("price:Q", scale=alt.Scale(zero=False)),
                    tooltip=[
                        "horizon:N",
                        "date:T",
                        alt.Tooltip("price:Q", format=",.2f")
                    ],
                )
            )

            combined_chart = combined_chart + forecast_chart

        st.altair_chart(combined_chart, use_container_width=True)

    st.markdown("---")
    with st.expander("Raw API responses"):
        for i, pred in enumerate(predictions):
            horizon = pred.get("horizon") or pred.get("horizon_days", 0)
            st.write(f"**D{horizon} Prediction:**")
            st.json(pred)
            if i < len(predictions) - 1:
                st.divider()


def render_developer_docs(api_base_url: Optional[str]) -> None:
    """Display developer-focused usage instructions inside an expander."""
    with st.expander("API for developers"):
        base_url = (api_base_url or "https://<your-cloud-function-url>").rstrip("/")
        st.markdown(
            "Use the deployed REST API for programmatic access or integrations."
        )

        # Show available resources
        st.markdown("**Available Resources:**")
        st.code(f"""# Get available watches
curl {base_url}/watches

# Get available models
curl {base_url}/models

# Check API health
curl {base_url}/health""", language="bash")

        st.markdown("**Single Prediction:**")
        st.code(
            f"""# REST example with correct watch ID
curl -X POST {base_url}/predict \\
  -H "Content-Type: application/json" \\
  -d '{{"watch_id": "Philippe_Nautilus_5711_Stainless_Steel_5711_1A", "model_name": "lightgbm", "horizon": 7}}'
""",
            language="bash",
        )

        st.markdown("**Batch Predictions:**")
        st.code(
            f"""# Batch prediction example
curl -X POST {base_url}/predict/batch \\
  -H "Content-Type: application/json" \\
  -d '{{"watch_ids": ["Philippe_Nautilus_5711_Stainless_Steel_5711_1A", "Philippe_Aquanaut_5167_Stainless_Steel_5167A"], "model_name": "lightgbm", "horizon": 7}}'
""",
            language="bash",
        )

        st.markdown("**Python Example:**")
        st.code(
            """import requests

# Single prediction
response = requests.post(
    \"{base_url}/predict\",
    json={{
        \"watch_id\": \"Philippe_Nautilus_5711_Stainless_Steel_5711_1A\",
        \"model_name\": \"lightgbm\",
        \"horizon\": 7,
    }},
    timeout=30,
)
response.raise_for_status()
print(response.json())

# Available models: lightgbm, linear, random_forest, ridge, xgboost
# Available watches: Use /watches endpoint to get current list
""".format(base_url=base_url),
            language="python",
        )
        st.markdown(
            "💡 **Tip:** Set the `GCP_API_URL` environment variable before launching Streamlit to "
            "default to the cloud backend."
        )


st.title("Luxury Watch Forecasting")
st.caption("Minimal dashboard for exploring next-week price predictions")

defaults = load_default_config()
env_api_url = get_cloud_api_base_url() or ""

with st.sidebar:
    st.header("Backend configuration")
    raw_cloud_api_url = st.text_input(
        "Cloud API URL",
        value=env_api_url,
        key="cloud_api_url",
        placeholder="https://predict-<function-id>-<region>.a.run.app",
        help="Enter or export GCP_API_URL to call the deployed Cloud Function/Run API.",
    )
    cloud_api_input = raw_cloud_api_url.strip()
    has_cloud_api = bool(cloud_api_input)

    use_cloud_backend = has_cloud_api

    if not has_cloud_api:
        st.warning("Cloud API URL required. Enter the URL above to enable inference.")
    else:
        st.success("Using Cloud API for inference")

cloud_api_url = cloud_api_input.rstrip("/") if cloud_api_input else None

if use_cloud_backend and cloud_api_url:
    run_cloud_ui(defaults, cloud_api_url)
else:
    st.error("Please provide a Cloud API URL to use the application.")

render_developer_docs(cloud_api_url or env_api_url)
