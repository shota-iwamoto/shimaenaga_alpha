#include "objective.h"
#include "../util/log.h"
#include <cmath>
#include <algorithm>
#include <vector>

namespace shimaenaga {

// Weighted-agnostic quantile of a copied buffer (used for robust init scores
// and the adaptive Huber delta).
static double Quantile(std::vector<double> v, double q) {
  if (v.empty()) return 0.0;
  size_t idx = (size_t)std::min((double)(v.size() - 1),
                                std::max(0.0, q * (double)(v.size() - 1)));
  std::nth_element(v.begin(), v.begin() + idx, v.end());
  return v[idx];
}

class RegressionL2 : public Objective {
 public:
  void Init(const Dataset&) override {}

  std::vector<score_t> InitScore(const Dataset& ds) const override {
    // F_0 = weighted mean (E.50)
    double sw = 0.0, swy = 0.0;
    for (data_size_t i = 0; i < ds.NumData(); ++i) {
      double w = ds.HasWeights() ? ds.Weights()[i] : 1.0;
      sw  += w;
      swy += w * ds.Labels()[i];
    }
    double f0 = (sw > 0) ? swy / sw : 0.0;
    return std::vector<score_t>(ds.NumData(), f0);
  }

  void GetGradients(const score_t* F, data_size_t n,
                    const float* labels, const float* weights,
                    score_t* g, score_t* h) const override {
    #pragma omp parallel for schedule(static) if (n >= 16384)
    for (data_size_t i = 0; i < n; ++i) {
      double w = weights ? weights[i] : 1.0;
      g[i] = w * (F[i] - labels[i]);   // (E.50)
      h[i] = w;
    }
  }

  double EvalMetric(const score_t* F, data_size_t n, const float* labels,
                    const std::vector<int32_t>* /*group_bounds*/ = nullptr) const override {
    double mse = ChunkedSum(n, 1, [&](data_size_t i) {
      double d = F[i] - labels[i];
      return d * d;
    });
    return std::sqrt(mse / n);  // RMSE
  }

  std::string MetricName() const override { return "rmse"; }
};

// Huber loss with adaptive delta: delta_m = huber_alpha-quantile of |residual|,
// recomputed every iteration. Outliers past delta contribute a clipped
// (constant-magnitude) gradient, so a few corrupted labels cannot dominate
// tree growth the way they do under pure L2.
class RegressionHuber : public Objective {
 public:
  explicit RegressionHuber(double alpha) : alpha_(alpha) {}

  void Init(const Dataset&) override {}

  std::vector<score_t> InitScore(const Dataset& ds) const override {
    // F_0 = median(y): robust to label outliers, unlike the L2 mean init.
    std::vector<double> y(ds.Labels(), ds.Labels() + ds.NumData());
    double f0 = Quantile(std::move(y), 0.5);
    return std::vector<score_t>(ds.NumData(), f0);
  }

  void GetGradients(const score_t* F, data_size_t n,
                    const float* labels, const float* weights,
                    score_t* g, score_t* h) const override {
    std::vector<double> abs_r(n);
    #pragma omp parallel for schedule(static) if (n >= 16384)
    for (data_size_t i = 0; i < n; ++i)
      abs_r[i] = std::abs(F[i] - labels[i]);
    double delta = std::max(Quantile(std::move(abs_r), alpha_), 1e-10);

    #pragma omp parallel for schedule(static) if (n >= 16384)
    for (data_size_t i = 0; i < n; ++i) {
      double w = weights ? weights[i] : 1.0;
      double r = F[i] - labels[i];
      g[i] = w * std::max(-delta, std::min(delta, r));
      // Constant hessian keeps the Newton leaf refit well-conditioned even in
      // the linear (outlier) region where the true second derivative is 0.
      h[i] = w;
    }
  }

  double EvalMetric(const score_t* F, data_size_t n, const float* labels,
                    const std::vector<int32_t>* /*group_bounds*/ = nullptr) const override {
    // MAE: delta-free, so early stopping monitors a stable robust metric.
    double mae = ChunkedSum(n, 1, [&](data_size_t i) {
      return std::abs(F[i] - labels[i]);
    });
    return mae / n;
  }

  std::string MetricName() const override { return "mae"; }

 private:
  double alpha_;
};

// Pinball (quantile) loss. alpha=0.5 is exactly MAE / median regression.
class RegressionQuantile : public Objective {
 public:
  explicit RegressionQuantile(double alpha) : alpha_(alpha) {}

  void Init(const Dataset&) override {}

  std::vector<score_t> InitScore(const Dataset& ds) const override {
    std::vector<double> y(ds.Labels(), ds.Labels() + ds.NumData());
    double f0 = Quantile(std::move(y), alpha_);
    return std::vector<score_t>(ds.NumData(), f0);
  }

  void GetGradients(const score_t* F, data_size_t n,
                    const float* labels, const float* weights,
                    score_t* g, score_t* h) const override {
    #pragma omp parallel for schedule(static) if (n >= 16384)
    for (data_size_t i = 0; i < n; ++i) {
      double w = weights ? weights[i] : 1.0;
      g[i] = w * ((F[i] >= labels[i]) ? (1.0 - alpha_) : -alpha_);
      h[i] = w;  // surrogate hessian (true one is 0 a.e.)
    }
  }

  double EvalMetric(const score_t* F, data_size_t n, const float* labels,
                    const std::vector<int32_t>* /*group_bounds*/ = nullptr) const override {
    double loss = ChunkedSum(n, 1, [&](data_size_t i) {
      double d = labels[i] - F[i];
      return (d >= 0) ? alpha_ * d : (alpha_ - 1.0) * d;
    });
    return loss / n;
  }

  std::string MetricName() const override { return "quantile"; }

 private:
  double alpha_;
};

// Registration in Create()
static std::unique_ptr<Objective> MakeRegression() {
  return std::make_unique<RegressionL2>();
}

// Forward declaration for Create
extern std::unique_ptr<Objective> MakeBinary(double h_min);
extern std::unique_ptr<Objective> MakeMulticlass(int K, double h_min);
extern std::unique_ptr<Objective> MakeLambdaRank(const Config& cfg);

std::unique_ptr<Objective> Objective::Create(const Config& cfg) {
  if (cfg.objective == "regression" || cfg.objective == "regression_l2" ||
      cfg.objective == "mse") {
    return MakeRegression();
  }
  if (cfg.objective == "huber") {
    return std::make_unique<RegressionHuber>(cfg.huber_alpha);
  }
  if (cfg.objective == "quantile") {
    return std::make_unique<RegressionQuantile>(cfg.quantile_alpha);
  }
  if (cfg.objective == "mae" || cfg.objective == "regression_l1") {
    return std::make_unique<RegressionQuantile>(0.5);
  }
  if (cfg.objective == "binary" || cfg.objective == "binary_crossentropy") {
    return MakeBinary(cfg.h_min);
  }
  if (cfg.objective == "multiclass" || cfg.objective == "softmax") {
    return MakeMulticlass(cfg.num_class, cfg.h_min);
  }
  if (cfg.objective == "lambdarank" || cfg.objective == "rank") {
    return MakeLambdaRank(cfg);
  }
  throw ConfigError("Unknown objective: " + cfg.objective);
}

} // namespace shimaenaga
