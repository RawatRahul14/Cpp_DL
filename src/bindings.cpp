#include "wine_model.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py=pybind11;
PYBIND11_MODULE(wine_engine,m) {
  m.doc()="C++ LibTorch wine-quality regression engine";
  m.def("train_csv",[](const std::string& csv,const std::string& model,const std::string& scaler,int epochs,int seed){
    auto r=train_csv(csv,model,scaler,epochs,seed);py::dict d;
    d["test_mae"]=r.test_mae;d["test_rmse"]=r.test_rmse;d["test_r2"]=r.test_r2;
    d["final_train_loss"]=r.final_train_loss;d["train_rows"]=r.train_rows;d["test_rows"]=r.test_rows;return d;
  },py::arg("csv_path"),py::arg("model_path"),py::arg("scaler_path"),py::arg("epochs")=300,py::arg("seed")=42);
  m.def("predict_one",&predict_one,py::arg("features"),py::arg("model_path"),py::arg("scaler_path"));
}
