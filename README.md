# NeuralNet-Cpp: Wine Quality Prediction Engine

A compact, end-to-end deep learning regression system where **C++20 and LibTorch** handle all dataset parsing, feature standardization, neural network training, evaluation, and serialization. **Python (Streamlit)** serves strictly as a presentation frontend communicating with the native C++ engine via **pybind11**.

---

## Key Features

* **Pure C++20 ML Pipeline:** Zero reliance on `scikit-learn` or Python `torch` for data preprocessing, training, or inference.
* **MLP Regressor Architecture:** Feedforward network (`11 -> 32 -> 16 -> 1`) with `ReLU` activations trained via the `Adam` optimizer.
* **Leakage-Free Preprocessing:** Deterministic 80/20 train/test split using `std::mt19937`, with Z-score standardization statistics ($\mu, \sigma$) computed strictly on the training partition.
* **Native Artifact Serialization:** Saves trained model weights (`artifacts/wine_model.pt`) and custom scaler parameters (`artifacts/scaler.csv`) directly from C++.
* **Interactive Dashboard:** Streamlit UI (`app.py`) for triggering C++ training epochs, inspecting out-of-sample test metrics (`MAE`, `RMSE`, $R^2$), and running real-time single-sample predictions.

---

## System Architecture & Mathematical Pipeline

```text
+-----------------------------------------------------------------------+
|                       Streamlit Frontend (app.py)                     |
|   - Epoch Configuration Slider        - 11-Feature Input Form         |
|   - Out-of-Sample Metric Cards        - Correlation Visualization     |
+-----------------------------------+-----------------------------------+
                                    |
                         pybind11 C-API Bridge
                          (wine_engine.*.pyd)
                                    |
+-----------------------------------v-----------------------------------+
|                     C++20 / LibTorch Core Engine                      |
|                                                                       |
|  1. CSV Ingestion      -> Parses `;` delimited UCI dataset            |
|  2. Shuffling & Split  -> Seeded std::mt19937 (80% Train / 20% Test)  |
|  3. Z-Score Scaling    -> z = (x - mu_train) / sd_train               |
|  4. Forward/Backward   -> 11 -> Linear(32) -> ReLU -> Linear(16)      |
|                           -> ReLU -> Linear(1) (MSE Loss + Adam)      |
|  5. Persistence        -> Writes wine_model.pt & scaler.csv           |
+-----------------------------------------------------------------------+
```

### 1. Feature Standardization (Preventing Data Leakage)
For each of the $F = 11$ physicochemical features, the mean $\mu_j$ and standard deviation $\sigma_j$ are computed exclusively over the training indices $I_{\text{train}}$:

$$\mu_j = \frac{1}{\vert{}I_{\text{train}}\vert{}} \sum_{i \in I_{\text{train}}} x_{i,j}, \quad \sigma_j = \sqrt{\frac{1}{\vert{}I_{\text{train}}\vert{}} \sum_{i \in I_{\text{train}}} (x_{i,j} - \mu_j)^2}$$

Both training and testing tensors are transformed using training statistics:

$$z_{i,j} = \frac{x_{i,j} - \mu_j}{\sigma_j}$$

### 2. Multi-Layer Perceptron (MLP) Topology
Given a standardized input vector $\mathbf{z} \in \mathbb{R}^{11}$:

$$\mathbf{h}_1 = \text{ReLU}(\mathbf{W}_1 \mathbf{z} + \mathbf{b}_1), \quad \mathbf{W}_1 \in \mathbb{R}^{32 \times 11}$$

$$\mathbf{h}_2 = \text{ReLU}(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2), \quad \mathbf{W}_2 \in \mathbb{R}^{16 \times 32}$$

$$\hat{y} = \mathbf{W}_3 \mathbf{h}_2 + b_3, \quad \mathbf{W}_3 \in \mathbb{R}^{1 \times 16}$$

### 3. Training Objective & Evaluation Metrics
The network minimizes full-batch **Mean Squared Error (MSE)** via `torch::optim::Adam` ($\text{lr} = 0.003$):

$$\mathcal{L}_{\text{MSE}} = \frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} (\hat{y}_i - y_i)^2$$

During evaluation (`net->eval()` and `torch::NoGradGuard`), the engine computes out-of-sample metrics on the held-out 20% test split:
* **MAE:** $\frac{1}{N_{\text{test}}} \sum \vert{}\hat{y}_i - y_i\vert{}$
* **RMSE:** $\sqrt{\frac{1}{N_{\text{test}}} \sum (\hat{y}_i - y_i)^2}$
* **$R^2$ (Coefficient of Determination):** $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$

---

## Project Structure

```text
NeuralNet-Cpp/
├── CMakeLists.txt           # CMake build specification for MSVC + LibTorch + pybind11
├── README.md                # Project documentation and build instructions
├── LICENSE                  # MIT License
├── requirements.txt         # Python dependencies (Streamlit, Pandas, pybind11)
├── app.py                   # Streamlit dashboard & user interface
├── scripts/
│   └── download_data.py     # Fetches and extracts the official UCI Wine Quality CSV
├── src/
│   ├── wine_model.h         # LibTorch module definitions and API declarations
│   ├── wine_model.cpp       # CSV parser, standardization, training loop, and inference
│   └── bindings.cpp         # pybind11 module wrapper (`wine_engine`)
├── data/                    # Generated by download_data.py (winequality-red.csv)
├── artifacts/               # Generated at runtime (wine_model.pt, scaler.csv)
└── build/                   # Generated by CMake during compilation
```

---

## Prerequisites

* **OS:** Windows 10/11 (x64)
* **Compiler:** Visual Studio C++ Build Tools (MSVC v143 / Visual Studio 18 2026)
* **Build System:** CMake 3.20+
* **Python:** Python 3.11 (64-bit, with development headers installed)
* **LibTorch:** CPU LibTorch (Release, x64) extracted at `C:\libtorch`

---

## Step-by-Step Build & Execution Guide

### Step 1: Initialize the x64 MSVC Developer PowerShell
LibTorch and Python 64-bit extensions require the native `amd64` MSVC toolchain (`cl.exe` and `link.exe`). Launch the developer shell to populate your environment variables:

```powershell
$vsDevShell = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\Tools\Launch-VsDevShell.ps1"
& $vsDevShell -Arch amd64 -HostArch amd64
```

### Step 2: Set Up the Python Virtual Environment & Dataset
Create an isolated Python 3.11 environment, install `pybind11` and `streamlit`, and download the UCI Wine Quality (Red Wine) dataset (1,599 observations):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts\download_data.py
```

### Step 3: Configure and Compile the C++ Extension via CMake
Query `pybind11` for its CMake configuration directory, pass both `C:/libtorch` and `$pybindDir` to `CMAKE_PREFIX_PATH`, and compile the `Release` configuration:

```powershell
$pybindDir = python -m pybind11 --cmakedir

cmake -S . -B build -G "Visual Studio 18 2026" -A x64 `
  -DCMAKE_PREFIX_PATH="C:/libtorch;$pybindDir" `
  -DPython_EXECUTABLE="$((Get-Command python).Source)"

cmake --build build --config Release
```

> **Troubleshooting CMake:**
> * If your Visual Studio version differs (e.g., VS 2022), run `cmake --help` to check available generators (such as `"Visual Studio 17 2022"`).
> * Always ensure your LibTorch build flavor matches `--config Release` (Release LibTorch binaries cannot link against a `Debug` MSVC build due to CRT mismatch).

### Step 4: Stage the Compiled `.pyd` Module and LibTorch DLLs
Python imports `.pyd` dynamic libraries from the project root, and Windows requires `C:\libtorch\lib` (which houses `torch_cpu.dll` and `c10.dll`) in the system `PATH` at runtime:

```powershell
$pyd = Get-ChildItem .\build -Recurse -Filter "wine_engine*.pyd" | Select-Object -First 1
if (-not $pyd) { throw "Extension not found; inspect build output." }
Copy-Item $pyd.FullName .\

$env:Path = "C:\libtorch\lib;$env:Path"
```

### Step 5: Launch the Streamlit Application
With the virtual environment active and `C:\libtorch\lib` on your `PATH`, start the web interface:

```powershell
python -m streamlit run app.py
```

1. Click **Train C++ model** to run full-batch optimization in C++ and serialize `artifacts/wine_model.pt` and `artifacts/scaler.csv`.
2. Inspect the held-out test metrics (**MAE**, **RMSE**, and **$R^2$**).
3. Adjust any of the 11 chemical parameters in the prediction form and click **Predict quality score** to execute C++ single-sample inference.

---

## Dataset & Limitations

* **Dataset:** [UCI Wine Quality Dataset (Red Wine)](https://archive.ics.uci.edu/dataset/186/wine+quality) — 1,599 instances, 11 physicochemical input features, and a sensory quality score target (0 to 10).
* **Scope:** This project serves as a clean architectural blueprint for integrating C++20/LibTorch with Python via `pybind11`. It trains a full-batch MLP on a single seeded 80/20 split rather than performing k-fold cross-validation or hyperparameter tuning.
* **Kineto Warning:** Some CPU LibTorch distributions emit a non-fatal `Kineto` warning during CMake configuration; this can be safely ignored.