#pragma once
#include <torch/torch.h>
#include <string>
#include <vector>
struct WineRegressorImpl : torch::nn::Module {
  WineRegressorImpl();
  torch::Tensor forward(torch::Tensor x);
  torch::nn::Sequential network{nullptr};
};
TORCH_MODULE(WineRegressor);
struct TrainResult {
  double test_mae, test_rmse, test_r2, final_train_loss;
  int train_rows, test_rows;
};
TrainResult train_csv(const std::string&, const std::string&, const std::string&, int, int);
std::vector<double> predict_one(const std::vector<double>&, const std::string&, const std::string&);
