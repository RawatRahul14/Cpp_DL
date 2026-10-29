#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>

/**
 * @brief Multi-Layer Perceptron (MLP) architecture for Wine Quality Regression.
 * 
 * Topology: 11 (Input) -> 32 (Linear + ReLU) -> 16 (Linear + ReLU) -> 1 (Linear Output)
 */
struct WineRegressorImpl : torch::nn::Module {
    WineRegressorImpl();

    /**
     * @brief Executes the forward pass through the sequential network.
     * @param x Input tensor of shape [batch_size, 11]
     * @return Output prediction tensor of shape [batch_size, 1]
     */
    torch::Tensor forward(torch::Tensor x);

    torch::nn::Sequential network{nullptr};
};

// Generates the `WineRegressor` smart-pointer wrapper class around `WineRegressorImpl`
TORCH_MODULE(WineRegressor);

/**
 * @brief Container holding out-of-sample evaluation metrics and dataset split counts.
 */
struct TrainResult {
    double test_mae; // Mean Absolute Error on the 20% test split
    double test_rmse; // Root Mean Squared Error on the 20% test split
    double test_r2; // Coefficient of Determination (R^2) on the 20% test split
    double final_train_loss; // Final epoch Mean Squared Error (MSE) on the 80% train split
    int train_rows; // Number of observations used for training
    int test_rows; // Number of observations used for testing
};

/**
 * @brief Loads the UCI CSV dataset, standardizes features using train-split statistics,
 *        trains the MLP via Adam, evaluates on the test split, and serializes artifacts.
 * 
 * @param csv_path    Path to the semicolon-delimited `winequality-red.csv` file
 * @param model_path  Destination file path to serialize trained LibTorch weights (`.pt`)
 * @param scaler_path Destination file path to save feature mean and std parameters (`.csv`)
 * @param epochs      Number of full-batch gradient descent iterations (1 to 10,000)
 * @param seed        PRNG seed for deterministic shuffling and weight initialization
 * @return TrainResult struct populated with evaluation metrics
 */
TrainResult train_csv(
    const std::string& csv_path,
    const std::string& model_path,
    const std::string& scaler_path,
    int epochs,
    int seed
);

/**
 * @brief Runs single-sample inference using saved scaler parameters and model weights.
 * 
 * @param features    Vector of 11 raw chemical measurements
 * @param model_path  Path to the serialized LibTorch `.pt` model file
 * @param scaler_path Path to the saved `.csv` standardization parameters
 * @return Vector containing a single double: the predicted wine quality score
 */
std::vector<double> predict_one(
    const std::vector<double>& features,
    const std::string& model_path,
    const std::string& scaler_path
);