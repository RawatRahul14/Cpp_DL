NeuralNet-Cpp: Wine Quality Prediction
A compact end-to-end regression project: C++20 + LibTorch owns model training and inference. Python is the UI and bridge only; no Python ML model is trained.

Features
MLP regressor: 11 numerical inputs → 32 → 16 → 1
CSV loading, deterministic 80/20 split, train-only standardization
MAE, RMSE and R² evaluation
Saved C++ model and scaler parameters
pybind11 bridge and Streamlit dashboard
Dataset: UCI Wine Quality (red wine), 1,599 observations. Official source: https://archive.ics.uci.edu/dataset/186/wine+quality. The target is a human-assigned score; this is an educational benchmark, not an objective or commercial wine grader.

Requirements
Windows x64, Visual Studio C++ Build Tools, CMake, Python 3.11, CPU LibTorch at C:\\libtorch.

1. Use x64 Developer PowerShell
$vsDevShell = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\Tools\Launch-VsDevShell.ps1"
& $vsDevShell -Arch amd64 -HostArch amd64

2. Install Python dependencies and dataset
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts\download_data.py

3. Configure and build
$pybindDir = python -m pybind11 --cmakedir
cmake -S . -B build -G "Visual Studio 18 2026" -A x64 `
  -DCMAKE_PREFIX_PATH="C:/libtorch;$pybindDir" `
  -DPython_EXECUTABLE="$((Get-Command python).Source)"
cmake --build build --config Release

If the Visual Studio generator name differs, check cmake --help. Keep the same Python environment active for CMake and Streamlit. If CMake cannot find Python development components, verify your Python installation has development headers.

4. Copy the built extension and set DLL path
$pyd = Get-ChildItem .\build -Recurse -Filter "wine_engine*.pyd" | Select-Object -First 1
if (-not $pyd) { throw "Extension not found; inspect build output." }
Copy-Item $pyd.FullName .\
$env:Path = "C:\libtorch\lib;$env:Path"

5. Run
python -m streamlit run app.py

Click Train C++ model, then use the prediction form.

Layout
src/wine_model.cpp: C++ data handling, preprocessing, training, evaluation and inference
src/bindings.cpp: pybind11 API
app.py: Streamlit interface
scripts/download_data.py: downloads the official dataset
artifacts/: generated model/scaler files
build/: generated CMake files
Limitations
This is a full-batch baseline MLP, not a tuned state-of-the-art model. Test metrics come from one seeded split. A serious report should compare with a mean-prediction baseline and use cross-validation. The Kineto warning from some CPU LibTorch distributions is not itself a build failure.