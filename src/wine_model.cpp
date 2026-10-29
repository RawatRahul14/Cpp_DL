#include "wine_model.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace {

// Number of input physicochemical features in the UCI Wine Quality dataset
constexpr int kNumFeatures = 11;

/**
 * @brief Internal container for parsed raw CSV observations.
 */
struct Data {
    std::vector<std::vector<double>> x; // 2D feature matrix [N][11]
    std::vector<double> y; // 1D target vector [N]
};

/**
 * @brief Parses the semicolon-delimited UCI Wine Quality CSV file.
 */
Data read_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open CSV: " + path);
    }

    // Skip the header row
    std::string line;
    std::getline(in, line);

    Data data;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }

        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row_values;

        // Parse semicolon-separated floats
        while (std::getline(ss, cell, ';')) {
            row_values.push_back(std::stod(cell));
        }

        // Expect 11 input features + 1 target column = 12 columns total
        if (row_values.size() != 12) {
            throw std::runtime_error(
                "Expected 11 features + quality target; use UCI red-wine CSV."
            );
        }

        data.x.emplace_back(row_values.begin(), row_values.begin() + kNumFeatures);
        data.y.push_back(row_values.back());
    }

    if (data.x.size() < 20) {
        throw std::runtime_error("Too few CSV rows to perform a valid split.");
    }

    return data;
}

/**
 * @brief Converts indexed subset of features into a standardized 2D Float32 Tensor.
 * 
 * Standardization formula: z = (x - mu) / sd
 * 
 * IMPORTANT: `torch::from_blob` does not take ownership of the underlying `std::vector`
 * buffer. Calling `.clone()` allocates a new LibTorch-managed memory buffer before
 * `flat_buffer` goes out of scope.
 */
torch::Tensor make_x(
    const Data& data,
    const std::vector<size_t>& indices,
    const std::vector<double>& mu,
    const std::vector<double>& sd
) {
    std::vector<float> flat_buffer;
    flat_buffer.reserve(indices.size() * kNumFeatures);

    for (size_t row_idx : indices) {
        for (int col_idx = 0; col_idx < kNumFeatures; ++col_idx) {
            double standardized = (data.x[row_idx][col_idx] - mu[col_idx]) / sd[col_idx];
            flat_buffer.push_back(static_cast<float>(standardized));
        }
    }

    auto tensor_view = torch::from_blob(
        flat_buffer.data(),
        {static_cast<long>(indices.size()), kNumFeatures},
        torch::kFloat32
    );

    return tensor_view.clone();
}

/**
 * @brief Converts indexed subset of target values into a 2D Float32 column Tensor [N, 1].
 */
torch::Tensor make_y(const Data& data, const std::vector<size_t>& indices) {
    std::vector<float> flat_buffer;
    flat_buffer.reserve(indices.size());

    for (size_t row_idx : indices) {
        flat_buffer.push_back(static_cast<float>(data.y[row_idx]));
    }

    auto tensor_view = torch::from_blob(
        flat_buffer.data(),
        {static_cast<long>(indices.size()), 1},
        torch::kFloat32
    );

    return tensor_view.clone();
}

/**
 * @brief Serializes the training set feature means and standard deviations to disk.
 * 
 * Format:
 * Line 1: wine_scaler_v1 (version header)
 * Line 2: comma-separated means (11 values)
 * Line 3: comma-separated standard deviations (11 values)
 */
void save_scaler(
    const std::string& path,
    const std::vector<double>& mean,
    const std::vector<double>& std_dev
) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Cannot write scaler file: " + path);
    }

    out << "wine_scaler_v1\n";

    for (int j = 0; j < kNumFeatures; ++j) {
        out << mean[j] << (j + 1 == kNumFeatures ? '\n' : ',');
    }

    for (int j = 0; j < kNumFeatures; ++j) {
        out << std_dev[j] << (j + 1 == kNumFeatures ? '\n' : ',');
    }
}

/**
 * @brief Deserializes feature means and standard deviations from disk for inference.
 */
void load_scaler(
    const std::string& path,
    std::vector<double>& mean,
    std::vector<double>& std_dev
) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open scaler file: " + path);
    }

    std::string line;
    std::getline(in, line);
    if (line != "wine_scaler_v1") {
        throw std::runtime_error("Invalid scaler version header.");
    }

    // Pass 0 reads means; Pass 1 reads standard deviations
    for (int pass = 0; pass < 2; ++pass) {
        if (!std::getline(in, line)) {
            throw std::runtime_error("Scaler file incomplete.");
        }

        std::stringstream ss(line);
        std::string cell;
        std::vector<double> values;

        while (std::getline(ss, cell, ',')) {
            values.push_back(std::stod(cell));
        }

        if (values.size() != kNumFeatures) {
            throw std::runtime_error("Scaler must contain exactly 11 values per row.");
        }

        if (pass == 0) {
            mean = std::move(values);
        } else {
            std::move(values.begin(), values.end(), std::back_inserter(std_dev));
            std_dev = values;
        }
    }
}

} // anonymous namespace

// ==============================================================================
// Model Implementation
// ==============================================================================

WineRegressorImpl::WineRegressorImpl() {
    network = register_module(
        "network",
        torch::nn::Sequential(
            torch::nn::Linear(kNumFeatures, 32),
            torch::nn::ReLU(),
            torch::nn::Linear(32, 16),
            torch::nn::ReLU(),
            torch::nn::Linear(16, 1)
        )
    );
}

torch::Tensor WineRegressorImpl::forward(torch::Tensor x) {
    return network->forward(x);
}

// ==============================================================================
// Training & Evaluation Pipeline
// ==============================================================================

TrainResult train_csv(
    const std::string& csv_path,
    const std::string& model_path,
    const std::string& scaler_path,
    int epochs,
    int seed
) {
    if (epochs < 1 || epochs > 10000) {
        throw std::runtime_error("epochs must be between 1 and 10000.");
    }

    // 1. Set LibTorch PRNG seed for reproducible weight initialization
    torch::manual_seed(seed);

    // 2. Load raw CSV data
    Data data = read_csv(csv_path);

    // 3. Generate deterministic 80/20 Train/Test split indices
    std::vector<size_t> indices(data.x.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 rng(static_cast<unsigned int>(seed));
    std::shuffle(indices.begin(), indices.end(), rng);

    size_t train_count = static_cast<size_t>(data.x.size() * 0.8);
    std::vector<size_t> train_idx(indices.begin(), indices.begin() + train_count);
    std::vector<size_t> test_idx(indices.begin() + train_count, indices.end());

    // 4. Compute Z-score scaler parameters STRICTLY on the training split (prevents test-set data leakage into training normalization)
    std::vector<double> mu(kNumFeatures, 0.0);
    std::vector<double> sd(kNumFeatures, 0.0);

    for (size_t row_idx : train_idx) {
        for (int j = 0; j < kNumFeatures; ++j) {
            mu[j] += data.x[row_idx][j];
        }
    }
    for (double& val : mu) {
        val /= static_cast<double>(train_idx.size());
    }

    for (size_t row_idx : train_idx) {
        for (int j = 0; j < kNumFeatures; ++j) {
            double diff = data.x[row_idx][j] - mu[j];
            sd[j] += diff * diff;
        }
    }
    for (double& val : sd) {
        val = std::sqrt(val / static_cast<double>(train_idx.size()));
        // Guard against division by zero for constant features
        if (val < 1e-12) {
            val = 1.0;
        }
    }

    // Persist scaler parameters for future single-sample inference
    save_scaler(scaler_path, mu, sd);

    // 5. Build standardized LibTorch tensors for training and validation
    torch::Tensor x_train = make_x(data, train_idx, mu, sd);
    torch::Tensor y_train = make_y(data, train_idx);
    torch::Tensor x_test  = make_x(data, test_idx, mu, sd);
    torch::Tensor y_test  = make_y(data, test_idx);

    // 6. Initialize model and Adam optimizer (learning rate = 3e-3)
    WineRegressor net;
    torch::optim::Adam optimizer(
        net->parameters(),
        torch::optim::AdamOptions(0.003)
    );

    // 7. Full-batch training loop
    double final_train_loss = 0.0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        net->train();
        optimizer.zero_grad();

        torch::Tensor predictions = net->forward(x_train);
        torch::Tensor loss = torch::mse_loss(predictions, y_train);

        loss.backward();
        optimizer.step();

        final_train_loss = loss.item<double>();
    }

    // 8. Out-of-sample evaluation on the held-out 20% test split
    net->eval();
    torch::NoGradGuard no_grad; // Disable autograd graph construction

    torch::Tensor preds_1d = net->forward(x_test).reshape({-1});
    torch::Tensor targets_1d = y_test.reshape({-1});

    // Compute MAE and RMSE
    double mae = torch::l1_loss(preds_1d, targets_1d).item<double>();
    double mse = torch::mse_loss(preds_1d, targets_1d).item<double>();
    double rmse = std::sqrt(mse);

    // Compute R^2 = 1 - (SS_res / SS_tot)
    double y_mean = targets_1d.mean().item<double>();
    double ss_tot = ((targets_1d - y_mean) * (targets_1d - y_mean)).sum().item<double>();
    double ss_res = ((targets_1d - preds_1d) * (targets_1d - preds_1d)).sum().item<double>();
    double r2 = (ss_tot > 1e-12) ? (1.0 - (ss_res / ss_tot)) : 0.0;

    // 9. Serialize trained model parameters to disk
    torch::save(net, model_path);

    return TrainResult{
        mae,
        rmse,
        r2,
        final_train_loss,
        static_cast<int>(train_idx.size()),
        static_cast<int>(test_idx.size())
    };
}

// ==============================================================================
// Single-Sample Inference
// ==============================================================================

std::vector<double> predict_one(
    const std::vector<double>& features,
    const std::string& model_path,
    const std::string& scaler_path
) {
    if (features.size() != kNumFeatures) {
        throw std::runtime_error("Exactly 11 features are required for inference.");
    }

    // 1. Load saved standardization parameters
    std::vector<double> mu;
    std::vector<double> sd;
    load_scaler(scaler_path, mu, sd);

    // 2. Standardize raw input vector
    std::vector<float> standardized_features;
    standardized_features.reserve(kNumFeatures);
    for (int j = 0; j < kNumFeatures; ++j) {
        double z = (features[j] - mu[j]) / sd[j];
        standardized_features.push_back(static_cast<float>(z));
    }

    // 3. Construct [1, 11] input tensor
    torch::Tensor x_input = torch::from_blob(
        standardized_features.data(),
        {1, kNumFeatures},
        torch::kFloat32
    ).clone();

    // 4. Load serialized model weights and run forward pass
    WineRegressor net;
    torch::load(net, model_path);
    net->eval();

    torch::NoGradGuard no_grad;
    double prediction = net->forward(x_input).item<double>();

    return {prediction};
}