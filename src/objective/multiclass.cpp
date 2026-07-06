#include "objective.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <cassert>

namespace shimaenaga {

// Multiclass softmax objective (E.52)
// F has shape [N][K], g and h also [N][K]
static constexpr int kMaxSoftmaxK = 64;  // stack buffer bound; larger K heap-allocs

class MulticlassSoftmax : public Objective {
 public:
  MulticlassSoftmax(int K, double h_min) : K_(K), h_min_(h_min) {}

  void Init(const Dataset&) override {}

  std::vector<score_t> InitScore(const Dataset& ds) const override {
    // F_0 = 0 for all classes (log-uniform prior)
    return std::vector<score_t>(ds.NumData() * K_, 0.0);
  }

  void GetGradients(const score_t* F, data_size_t n,
                    const float* labels, const float* weights,
                    score_t* g, score_t* h) const override {
    #pragma omp parallel for schedule(static) if ((int64_t)n * K_ >= 16384)
    for (data_size_t i = 0; i < n; ++i) {
      double w = weights ? weights[i] : 1.0;
      const score_t* fi = F + (size_t)i * K_;
      score_t* gi = g + (size_t)i * K_;
      score_t* hi = h + (size_t)i * K_;

      // Numerically stable softmax (fixed-size stack buffer: K ≤ kMaxSoftmaxK)
      double p[kMaxSoftmaxK];
      double* pp = p;
      std::vector<double> p_dyn;
      if (K_ > kMaxSoftmaxK) { p_dyn.resize(K_); pp = p_dyn.data(); }
      double mx = fi[0];
      for (int k = 1; k < K_; ++k) mx = std::max(mx, fi[k]);
      double sum = 0.0;
      for (int k = 0; k < K_; ++k) { pp[k] = std::exp(fi[k] - mx); sum += pp[k]; }
      for (int k = 0; k < K_; ++k) pp[k] /= (sum + 1e-15);

      int yi = static_cast<int>(labels[i]);
      for (int k = 0; k < K_; ++k) {
        double y_ik = (k == yi) ? 1.0 : 0.0;
        gi[k] = w * (pp[k] - y_ik);                          // (E.52)
        hi[k] = w * std::max(pp[k] * (1.0 - pp[k]), h_min_);
      }
    }
  }

  double EvalMetric(const score_t* F, data_size_t n, const float* labels,
                    const std::vector<int32_t>* /*group_bounds*/ = nullptr) const override {
    // Multiclass logloss: smooth in F, so early stopping sees graded progress
    // (the previous 0/1 error rate only moved when an argmax flipped).
    double loss = ChunkedSum(n, K_, [&](data_size_t i) {
      const score_t* fi = F + (size_t)i * K_;
      double mx = fi[0];
      for (int k = 1; k < K_; ++k) mx = std::max(mx, fi[k]);
      double sum = 0.0;
      for (int k = 0; k < K_; ++k) sum += std::exp(fi[k] - mx);
      int yi = static_cast<int>(labels[i]);
      return -(fi[yi] - mx - std::log(sum));
    });
    return loss / n;
  }

  std::string MetricName() const override { return "multi_logloss"; }
  int NumClasses() const override { return K_; }

 private:
  int K_;
  double h_min_;  // from cfg.h_min (E.52)
};

std::unique_ptr<Objective> MakeMulticlass(int K, double h_min) {
  return std::make_unique<MulticlassSoftmax>(K, h_min);
}

} // namespace shimaenaga
