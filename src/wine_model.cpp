#include "wine_model.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace {
constexpr int F = 11;
struct Data { std::vector<std::vector<double>> x; std::vector<double> y; };

Data read_csv(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("Cannot open CSV: " + path);
  std::string line; std::getline(in, line);
  Data d;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::stringstream ss(line); std::string cell; std::vector<double> v;
    while (std::getline(ss, cell, ';')) v.push_back(std::stod(cell));
    if (v.size() != 12) throw std::runtime_error("Expected 11 features + quality target; use UCI red-wine CSV.");
    d.x.emplace_back(v.begin(), v.begin()+F); d.y.push_back(v.back());
  }
  if (d.x.size() < 20) throw std::runtime_error("Too few CSV rows.");
  return d;
}
torch::Tensor make_x(const Data& d, const std::vector<size_t>& ids,
                     const std::vector<double>& mu, const std::vector<double>& sd) {
  std::vector<float> a; a.reserve(ids.size()*F);
  for (auto i: ids) for(int j=0;j<F;++j) a.push_back(float((d.x[i][j]-mu[j])/sd[j]));
  return torch::from_blob(a.data(), {long(ids.size()),F}, torch::kFloat32).clone();
}
torch::Tensor make_y(const Data& d, const std::vector<size_t>& ids) {
  std::vector<float> a; a.reserve(ids.size());
  for(auto i:ids) a.push_back(float(d.y[i]));
  return torch::from_blob(a.data(), {long(ids.size()),1}, torch::kFloat32).clone();
}
void save_scaler(const std::string& p,const std::vector<double>& m,const std::vector<double>& s) {
  std::ofstream o(p); if(!o) throw std::runtime_error("Cannot write scaler file.");
  o<<"wine_scaler_v1\n";
  for(int j=0;j<F;++j)o<<m[j]<<(j+1==F?'\n':',');
  for(int j=0;j<F;++j)o<<s[j]<<(j+1==F?'\n':',');
}
void load_scaler(const std::string& p,std::vector<double>& m,std::vector<double>& s) {
  std::ifstream in(p); if(!in)throw std::runtime_error("Cannot open scaler file.");
  std::string line; std::getline(in,line); if(line!="wine_scaler_v1")throw std::runtime_error("Invalid scaler version.");
  for(int pass=0;pass<2;++pass) {
    if(!std::getline(in,line))throw std::runtime_error("Scaler file incomplete.");
    std::stringstream ss(line);std::string cell;std::vector<double> v;
    while(std::getline(ss,cell,','))v.push_back(std::stod(cell));
    if(v.size()!=F)throw std::runtime_error("Scaler must contain 11 values.");
    (pass==0?m:s)=v;
  }
}
}
WineRegressorImpl::WineRegressorImpl() {
  network=register_module("network",torch::nn::Sequential(
    torch::nn::Linear(F,32),torch::nn::ReLU(),
    torch::nn::Linear(32,16),torch::nn::ReLU(),torch::nn::Linear(16,1)));
}
torch::Tensor WineRegressorImpl::forward(torch::Tensor x){return network->forward(x);}

TrainResult train_csv(const std::string& csv,const std::string& model_path,
                      const std::string& scaler_path,int epochs,int seed) {
  if(epochs<1||epochs>10000)throw std::runtime_error("epochs must be 1..10000");
  torch::manual_seed(seed); auto d=read_csv(csv);
  std::vector<size_t> ids(d.x.size());std::iota(ids.begin(),ids.end(),0);
  std::mt19937 rng(unsigned(seed));std::shuffle(ids.begin(),ids.end(),rng);
  size_t n=size_t(d.x.size()*0.8);
  std::vector<size_t> tr(ids.begin(),ids.begin()+n),te(ids.begin()+n,ids.end());
  std::vector<double> mu(F,0),sd(F,0);
  for(auto i:tr)for(int j=0;j<F;++j)mu[j]+=d.x[i][j];
  for(auto& v:mu)v/=double(tr.size());
  for(auto i:tr)for(int j=0;j<F;++j){double z=d.x[i][j]-mu[j];sd[j]+=z*z;}
  for(auto& v:sd){v=std::sqrt(v/double(tr.size()));if(v<1e-12)v=1;}
  save_scaler(scaler_path,mu,sd);
  auto xt=make_x(d,tr,mu,sd),yt=make_y(d,tr),xv=make_x(d,te,mu,sd),yv=make_y(d,te);
  WineRegressor net; torch::optim::Adam opt(net->parameters(),torch::optim::AdamOptions(0.003));
  double loss_value=0;
  for(int e=0;e<epochs;++e){net->train();opt.zero_grad();auto p=net->forward(xt);
    auto loss=torch::mse_loss(p,yt);loss.backward();opt.step();loss_value=loss.item<double>();}
  net->eval();torch::NoGradGuard ng;auto p=net->forward(xv).reshape({-1}), y=yv.reshape({-1});
  double mae=torch::l1_loss(p,y).item<double>(),rmse=std::sqrt(torch::mse_loss(p,y).item<double>());
  double mean=y.mean().item<double>(),tot=((y-mean)*(y-mean)).sum().item<double>();
  double res=((y-p)*(y-p)).sum().item<double>(),r2=tot>1e-12?1-res/tot:0;
  torch::save(net,model_path);
  return {mae,rmse,r2,loss_value,int(tr.size()),int(te.size())};
}
std::vector<double> predict_one(const std::vector<double>& f,const std::string& model_path,const std::string& scaler_path) {
  if(f.size()!=F)throw std::runtime_error("Exactly 11 features are required.");
  std::vector<double> m,s;load_scaler(scaler_path,m,s);std::vector<float> z;
  for(int j=0;j<F;++j)z.push_back(float((f[j]-m[j])/s[j]));
  auto x=torch::from_blob(z.data(),{1,F},torch::kFloat32).clone();
  WineRegressor net;torch::load(net,model_path);net->eval();torch::NoGradGuard ng;
  return {net->forward(x).item<double>()};
}
