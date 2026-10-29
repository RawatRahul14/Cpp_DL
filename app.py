"""
Streamlit UI for the C++20 LibTorch Wine Quality Regression Engine.
All model training, standardization, and inference execute natively in C++.
"""

from pathlib import Path
import pandas as pd
import streamlit as st
import os
import sys

# ==============================================================================
# Paths & Constants
# ==============================================================================
ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "winequality-red.csv"

ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "wine_model.pt"
SCALER_PATH = ARTIFACTS_DIR / "scaler.csv"

FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

# Default chemical measurements (first row of UCI Red Wine dataset)
DEFAULT_MEASUREMENTS = [
    7.4,     # fixed acidity
    0.70,    # volatile acidity
    0.00,    # citric acid
    1.9,     # residual sugar
    0.076,   # chlorides
    11.0,    # free sulfur dioxide
    34.0,    # total sulfur dioxide
    0.9978,  # density
    3.51,    # pH
    0.56,    # sulphates
    9.4,     # alcohol
]

# ==============================================================================
# Page Configuration & Header
# ==============================================================================
st.set_page_config(page_title="C++ Wine Quality Lab", layout="wide")

st.title("Wine Quality Prediction Lab")
st.caption("C++20 + LibTorch · pybind11 · Streamlit")
st.write(
    "Educational regression demo. It predicts the dataset's recorded human "
    "quality score; it is not a commercial wine grader."
)

# ==============================================================================
# Extension & Dataset Verification
# ==============================================================================

# Ensure root directory is in sys.path and register LibTorch DLL directory on Windows
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if sys.platform == "win32" and Path("C:/libtorch/lib").exists():
    os.add_dll_directory("C:/libtorch/lib")

try:
    import wine_engine
except ImportError:
    st.error(
        "C++ extension (`wine_engine`) not found. Build the project with CMake "
        "and copy `wine_engine*.pyd` into the project root. See `README.md`."
    )
    st.stop()

if not DATA_PATH.exists():
    st.warning("Dataset missing. Run `python scripts/download_data.py` first.")
    st.stop()

# ==============================================================================
# Dataset Preview Section
# ==============================================================================
df = pd.read_csv(DATA_PATH, sep=";")

with st.expander("Dataset preview"):
    st.write(f"{len(df):,} rows, 11 numerical features, target `quality`.")
    st.dataframe(df.head(10), use_container_width=True)

# ==============================================================================
# C++ Model Training & Evaluation Section
# ==============================================================================
st.subheader("Train and evaluate")

epochs = st.slider(
    label="Epochs",
    min_value=50,
    max_value=1000,
    value=300,
    step=50,
)

if st.button("Train C++ model", type="primary"):
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    with st.spinner("Training C++ LibTorch MLP on CPU..."):
        try:
            metrics = wine_engine.train_csv(
                csv_path=str(DATA_PATH),
                model_path=str(MODEL_PATH),
                scaler_path=str(SCALER_PATH),
                epochs=epochs,
                seed=42,
            )
            st.session_state.metrics = dict(metrics)
        except Exception as exc:
            st.exception(exc)

if "metrics" in st.session_state:
    m = st.session_state.metrics
    col_mae, col_rmse, col_r2 = st.columns(3)

    col_mae.metric("Test MAE", f"{m['test_mae']:.3f}")
    col_rmse.metric("Test RMSE", f"{m['test_rmse']:.3f}")
    col_r2.metric("Test R²", f"{m['test_r2']:.3f}")

    st.caption(
        f"Train rows: {m['train_rows']} · "
        f"Test rows: {m['test_rows']} · "
        f"Final training MSE: {m['final_train_loss']:.4f}"
    )

# ==============================================================================
# Single-Sample Inference Section
# ==============================================================================
st.subheader("Predict from measurements")

if not MODEL_PATH.exists() or not SCALER_PATH.exists():
    st.info("Train the C++ model first to generate model and scaler artifacts.")
else:
    with st.form("predict_form"):
        cols = st.columns(3)
        input_values = []

        for idx, (feature_name, default_val) in enumerate(zip(FEATURES, DEFAULT_MEASUREMENTS)):
            with cols[idx % 3]:
                val = st.number_input(
                    label=feature_name,
                    value=float(default_val),
                    format="%.5f",
                )
                input_values.append(val)

        submit_prediction = st.form_submit_button("Predict quality score")

    if submit_prediction:
        try:
            prediction = wine_engine.predict_one(
                features=input_values,
                model_path=str(MODEL_PATH),
                scaler_path=str(SCALER_PATH),
            )
            st.metric("Predicted score", f"{prediction[0]:.2f}")
        except Exception as exc:
            st.exception(exc)

# ==============================================================================
# Exploratory Feature Correlations
# ==============================================================================
st.subheader("Feature correlations with quality")
correlations = (
    df.corr(numeric_only=True)["quality"]
    .drop("quality")
    .sort_values()
)
st.bar_chart(correlations)