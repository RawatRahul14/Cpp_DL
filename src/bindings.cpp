#include "wine_model.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(wine_engine, module) {
    // Describe the Python extension module.
    module.doc() = "C++ LibTorch wine-quality regression engine";

    // Expose the C++ training function to Python.
    module.def(
        "train_csv",
        [](
            const std::string& csv,
            const std::string& model,
            const std::string& scaler,
            int epochs,
            int seed
        ) {
            // Train the model and collect its evaluation metrics.
            auto result = train_csv(
                csv,
                model,
                scaler,
                epochs,
                seed
            );

            // Convert the training results into a Python dictionary.
            py::dict metrics;

            metrics["test_mae"] = result.test_mae;
            metrics["test_rmse"] = result.test_rmse;
            metrics["test_r2"] = result.test_r2;

            metrics["final_train_loss"] = result.final_train_loss;

            metrics["train_rows"] = result.train_rows;
            metrics["test_rows"] = result.test_rows;

            return metrics;
        },

        // Define the Python argument names and default values.
        py::arg("csv_path"),
        py::arg("model_path"),
        py::arg("scaler_path"),
        py::arg("epochs") = 300,
        py::arg("seed") = 42
    );

    // Expose the single-sample prediction function to Python.
    module.def(
        "predict_one",
        &predict_one,
        py::arg("features"),
        py::arg("model_path"),
        py::arg("scaler_path")
    );
}