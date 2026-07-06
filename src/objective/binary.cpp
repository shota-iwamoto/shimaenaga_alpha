#include "objective.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace shimaenaga {

class BinaryLogloss : public Objective {
 public:
  explicit BinaryLogloss(double h_min) : h_min_(h_min) {}
  void Init(const Dataset&) override {}

  std::vector<score_t> InitScore(const Dataset& ds) const override {
    // F_0 = log(ȳ / (1-ȳ)) (E.51)
    double sw = 0.0, swy = 0.0;
    for (data_size_t i = 0; i < ds.NumData(); ++i) {
      double w = ds.HasWeights() ? ds.Weights()[i] : 1.0;
      sw  += w;
      swy += w * ds.Labels()[i];
    }
    double ybar = (sw > 0) ? swy / sw : 0.5;
    ybar = std::max(1e-7, std::min(1.0 - 1e-7, ybar));
    double f0 = std::log(ybar / (1.0 - ybar));
    return std::vector<score_t>(ds.NumData(), f0);
  }

  void GetGradients(const score_t* F, data_size_t n,
                    const float* labels, const float* weights,
                    score_t* g, score_t* h) const override {
    #pragma omp parallel for schedule(static) if (n >= 16384)
    for (data_size_t i = 0; i < n; ++i) {
      double w = weights ? weights[i] : 1.0;
      double p = 1.0 / (1.0 + std::exp(-F[i]));
      g[i] = w * (p - labels[i]);          // (E.51)
      h[i] = w * std::max(p * (1.0 - p), h_min_);
    }
  }

  double EvalMetric(const score_t* F, data_size_t n, const float* labels,
                    const std::vector<int32_t>* /*group_bounds*/ = nullptr) const override {
    double loss = ChunkedSum(n, 2, [&](data_size_t i) {
      double p = 1.0 / (1.0 + std::exp(-F[i]));
      p = std::max(1e-15, std::min(1.0 - 1e-15, p));
      return -(labels[i] * std::log(p) + (1.0 - labels[i]) * std::log(1.0 - p));
    });
    return loss / n;
  }

  std::string MetricName() const override { return "binary_logloss"; }

 private:
  double h_min_;  // from cfg.h_min (E.51)
};

std::unique_ptr<Objective> MakeBinary(double h_min) {
  return std::make_unique<BinaryLogloss>(h_min);
}

} // namespace shimaenaga
