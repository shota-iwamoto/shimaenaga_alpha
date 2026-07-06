#pragma once
#include <cstdint>
#include <vector>
#include <memory>
#include <string>
#include "../../include/shimaenaga/config.h"
#include "../../include/shimaenaga/dataset.h"

namespace shimaenaga {

// Deterministic parallel sum: fixed-count chunking (boundaries depend only on
// n, partials merged in chunk order) keeps results bit-identical for any
// thread count (T10) while the chunks run in parallel.
template <typename Body>
inline double ChunkedSum(data_size_t n, int64_t work_per_item, Body body) {
  constexpr int kChunks = 64;
  double partial[kChunks];
  #pragma omp parallel for schedule(static) if ((int64_t)n * work_per_item >= 32768)
  for (int c = 0; c < kChunks; ++c) {
    data_size_t s = (data_size_t)((int64_t)n * c / kChunks);
    data_size_t e = (data_size_t)((int64_t)n * (c + 1) / kChunks);
    double acc = 0.0;
    for (data_size_t i = s; i < e; ++i) acc += body(i);
    partial[c] = acc;
  }
  double total = 0.0;
  for (int c = 0; c < kChunks; ++c) total += partial[c];
  return total;
}

// Abstract objective (数学設計書 §8, 詳細設計書 §9)
class Objective {
 public:
  virtual ~Objective() = default;

  virtual void Init(const Dataset& ds) = 0;

  // Initial prediction F_0 (E.50-E.52)
  virtual std::vector<score_t> InitScore(const Dataset& ds) const = 0;

  // Compute gradients g, h (E.24)
  virtual void GetGradients(
      const score_t* F, data_size_t n,
      const float* labels, const float* weights,
      score_t* g, score_t* h) const = 0;

  // Evaluation metric. group_bounds: cumulative query boundaries of the
  // dataset being evaluated (ranking only; others ignore it). Passing the
  // evaluated dataset's own bounds matters — the objective's stored bounds
  // belong to the TRAINING set and must not be applied to a validation set.
  virtual double EvalMetric(
      const score_t* F, data_size_t n,
      const float* labels,
      const std::vector<int32_t>* group_bounds = nullptr) const = 0;

  virtual std::string MetricName() const = 0;
  virtual int NumClasses() const { return 1; }
  virtual bool IsRanking() const { return false; }

  static std::unique_ptr<Objective> Create(const Config& cfg);
};

} // namespace shimaenaga
