#include "objective.h"
#include "../util/random.h"
#include "../util/log.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>

namespace shimaenaga {

// LambdaMART ranking objective (E.54-E.57)
// LightGBM-compatible sign convention: s_i > s_j is correct when y_i > y_j
class LambdaRankObjective : public Objective {
 public:
  explicit LambdaRankObjective(const Config& cfg)
      : sigma_(cfg.sigma_rank),
        trunc_(cfg.ndcg_truncation),
        max_pairs_(cfg.max_pairs_per_group),
        h_min_(cfg.h_min),
        seed_(cfg.seed) {}

  void Init(const Dataset& ds) override {
    // Precompute discount table: 1/log2(rank+2)
    discount_.resize(trunc_ + 2);
    for (int r = 0; r <= trunc_ + 1; ++r)
      discount_[r] = 1.0 / std::log2(r + 2.0);
    // Capture query group boundaries (cumulative) so gradients respect groups.
    group_bounds_ = ds.GroupBoundaries();
  }

  std::vector<score_t> InitScore(const Dataset& ds) const override {
    return std::vector<score_t>(ds.NumData(), 0.0);
  }

  void GetGradients(const score_t* F, data_size_t n,
                    const float* labels, const float* weights,
                    score_t* g, score_t* h) const override {
    #pragma omp parallel for schedule(static) if (n >= 16384)
    for (data_size_t i = 0; i < n; ++i) { g[i] = 0.0; h[i] = 0.0; }
    if (group_bounds_.size() >= 2) {
      int ng = (int)group_bounds_.size() - 1;
      // Groups are disjoint index ranges and each is processed serially inside,
      // so group-parallelism is deterministic (詳細設計書 §14.1).
      #pragma omp parallel for schedule(dynamic, 16)
      for (int qi = 0; qi < ng; ++qi)
        ProcessGroup(F, n, labels, weights, group_bounds_[qi], group_bounds_[qi + 1], g, h);
    } else {
      // No group info: whole dataset is one query.
      ProcessGroup(F, n, labels, weights, 0, n, g, h);
    }
  }

  double EvalMetric(const score_t* F, data_size_t n, const float* labels,
                    const std::vector<int32_t>* group_bounds = nullptr) const override {
    // Mean NDCG@T over query groups. The caller passes the evaluated dataset's
    // OWN group boundaries — the stored bounds belong to the training set and
    // must never be applied to a validation set (different N → OOB reads/NaN).
    const std::vector<int32_t>& gb =
        (group_bounds && group_bounds->size() >= 2) ? *group_bounds : group_bounds_;
    if (gb.size() >= 2) {
      if (gb.back() != (int32_t)n)
        throw DataError("EvalMetric: group boundaries do not match the "
                        "evaluated dataset size");
      int ng = (int)gb.size() - 1;
      double sum = ChunkedSum((data_size_t)ng, 64, [&](data_size_t qi) {
        return ComputeNDCG(F, labels, gb[qi], gb[qi + 1]);
      });
      return ng ? 1.0 - sum / ng : 0.0;  // report as loss (lower is better)
    }
    return 1.0 - ComputeNDCG(F, labels, 0, n);
  }

  std::string MetricName() const override { return "ndcg"; }
  bool IsRanking() const override { return true; }

 private:
  double sigma_;
  int    trunc_;
  int    max_pairs_;
  double h_min_;
  uint64_t seed_;
  std::vector<double> discount_;
  std::vector<int32_t> group_bounds_;

  void ProcessGroup(const score_t* F, data_size_t /*n*/,
                    const float* labels, const float* weights,
                    int start, int end,
                    score_t* g, score_t* h) const {
    int gs = end - start;
    if (gs < 2) return;

    // Ideal DCG (label-sorted).
    std::vector<int> order(gs);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b){ return labels[start + a] > labels[start + b]; });
    double idcg = 0.0;
    for (int r = 0; r < std::min(gs, trunc_); ++r)
      idcg += (std::pow(2.0, labels[start + order[r]]) - 1.0) * discount_[r];
    if (idcg < 1e-15) return;  // skip all-equal groups (IDCG=0)

    // Current rank of each document = position when sorted by score descending.
    // Ties broken by index for determinism (T10).
    std::vector<int> by_score(gs);
    std::iota(by_score.begin(), by_score.end(), 0);
    std::sort(by_score.begin(), by_score.end(), [&](int a, int b){
      double sa = F[start + a], sb = F[start + b];
      if (sa != sb) return sa > sb;
      return a < b;
    });
    std::vector<int> rank(gs);
    for (int r = 0; r < gs; ++r) rank[by_score[r]] = r;

    auto disc = [&](int r){ return r < trunc_ ? discount_[r] : 0.0; };

    auto add_pair = [&](int a, int b) {
        int i = start + a, j = start + b;
        double si = F[i], sj = F[j];
        double rho_ij = 1.0 / (1.0 + std::exp(sigma_ * (si - sj)));  // (E.54)

        // |ΔNDCG| from swapping the two docs' current rank positions.
        double gain_i = std::pow(2.0, labels[i]) - 1.0;
        double gain_j = std::pow(2.0, labels[j]) - 1.0;
        double delta = std::fabs((gain_i - gain_j) * (disc(rank[a]) - disc(rank[b]))) / idcg;

        double lambda_ij = sigma_ * delta * rho_ij;
        double hess_ij = sigma_ * sigma_ * delta * rho_ij * (1.0 - rho_ij);

        double wi = weights ? weights[i] : 1.0;
        double wj = weights ? weights[j] : 1.0;
        g[i] -= wi * lambda_ij;  g[j] += wj * lambda_ij;   // (E.55)
        h[i] += wi * hess_ij;   h[j] += wj * hess_ij;      // (E.56)
    };

    if (max_pairs_ <= 0) {
      // All pairs, no buffering.
      for (int a = 0; a < gs; ++a)
        for (int b = 0; b < gs; ++b)
          if (labels[start + a] > labels[start + b]) add_pair(a, b);
    } else {
      // RANDOM subset of max_pairs_ per group (reservoir sampling, deterministic
      // per (seed, group) — previously the cap silently kept only the first
      // pairs in index order, biasing gradients to low-index documents).
      std::vector<std::pair<int,int>> pairs;
      pairs.reserve(max_pairs_);
      Random rng(seed_ * 0x9e3779b97f4a7c15ull + (uint64_t)start + 1);
      int64_t seen = 0;
      for (int a = 0; a < gs; ++a)
        for (int b = 0; b < gs; ++b) {
          if (labels[start + a] <= labels[start + b]) continue;  // need y_a > y_b
          ++seen;
          if ((int)pairs.size() < max_pairs_) {
            pairs.emplace_back(a, b);
          } else {
            int64_t j = (int64_t)(rng.NextDouble() * seen);
            if (j < max_pairs_) pairs[(size_t)j] = {a, b};
          }
        }
      for (const auto& [a, b] : pairs) add_pair(a, b);
    }
    for (int i = start; i < end; ++i)
      h[i] = std::max(h[i], h_min_);  // (E.57)
  }

  double ComputeNDCG(const score_t* F, const float* labels,
                     int start, int end) const {
    int gs = end - start;
    if (gs == 0) return 0.0;

    // Sort by F descending
    std::vector<int> order(gs);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b){ return F[start + a] > F[start + b]; });

    double dcg = 0.0;
    for (int r = 0; r < std::min(gs, trunc_); ++r)
      dcg += (std::pow(2.0, labels[start + order[r]]) - 1.0) * discount_[r];

    // Sort by label descending for IDCG
    std::sort(order.begin(), order.end(),
              [&](int a, int b){ return labels[start + a] > labels[start + b]; });
    double idcg = 0.0;
    for (int r = 0; r < std::min(gs, trunc_); ++r)
      idcg += (std::pow(2.0, labels[start + order[r]]) - 1.0) * discount_[r];
    return (idcg > 0) ? dcg / idcg : 0.0;
  }
};

std::unique_ptr<Objective> MakeLambdaRank(const Config& cfg) {
  return std::make_unique<LambdaRankObjective>(cfg);
}

} // namespace shimaenaga
