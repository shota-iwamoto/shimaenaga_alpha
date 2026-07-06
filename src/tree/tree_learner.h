#pragma once
#include <cstdint>
#include <vector>
#include <queue>
#include <functional>
#include <algorithm>
#include "../../include/shimaenaga/tree.h"
#include "../../include/shimaenaga/config.h"

namespace shimaenaga {

struct HistEntry {
  double g = 0.0, h = 0.0;
  // Sample count per bin: lets the split sweep enforce min_data_in_leaf at
  // search time. Previously the constraint was only checked after the winning
  // candidate was popped, and a failing candidate silently killed the leaf
  // (no re-search) — losing splits that a compliant threshold could provide.
  int64_t cnt = 0;
};

// Leaf-wise tree learner (LightGBM-compatible, 詳細設計書 §5.2)
class TreeLearner {
 public:
  TreeLearner(const Config& cfg,
              const std::vector<std::vector<bin_t>>& bins,
              data_size_t n_samples,
              int n_features);

  // Grow a tree using weighted gradients g̃, h̃  (C=1)
  Tree Grow(
      const std::vector<double>& g_tilde,
      const std::vector<double>& h_tilde,
      const std::vector<int>& feature_subset,
      int max_leaves,
      const std::vector<bool>& sample_mask = {});

  // Multi-output grow for C>1: g_NC[i*C+k], h_NC[i*C+k]
  // Uses multi-output gain = Σ_k gain_k across all classes
  Tree GrowMultiOutput(
      const std::vector<double>& g_NC,
      const std::vector<double>& h_NC,
      int C,
      const std::vector<int>& feature_subset,
      int max_leaves,
      const std::vector<bool>& sample_mask = {});

 private:
  const Config& cfg_;
  const std::vector<std::vector<bin_t>>& bins_;
  data_size_t n_;
  int F_;
  // Per-feature bin count (max bin + 1, capped [2,256]), precomputed once so the
  // per-split histogram sweep need not rescan samples to find the max bin.
  std::vector<int> feat_nbins_;

  struct SplitCandidate {
    double gain;
    int leaf_idx;
    int feature;
    int threshold_bin;
    int default_left;  // 0=right, 1=left
    bool is_cat;
    uint64_t cat_bitset;

    bool operator<(const SplitCandidate& o) const { return gain < o.gain; }
  };

  // ── Histogram-subtraction growth (数学設計書 §11.2 「histogram 減算則」) ──
  // Per-leaf histograms are kept alive; after a split only the smaller child is
  // re-scanned and the larger child is derived as parent − smaller. `off` maps
  // subset feature index fi to its slice [off[fi], off[fi+1]) in the flat hist.
  // Each growth step runs exactly ONE parallel region with fat per-feature
  // tasks (build + subtract + both sweeps fused): many small regions per split
  // drown in OpenMP barrier spin, which profiling showed dominated runtime.

  // Serial sweep of one feature's bin slice → best candidate for that feature.
  // n_total = number of samples in the leaf (for min_data_in_leaf gating).
  SplitCandidate SweepFeature(
      int leaf_idx, int f, const HistEntry* hs, int nb,
      double G_total, double H_total, int64_t n_total,
      double score_parent, double min_hess) const;

  // Root: build hist from samples and search, one fused parallel region.
  SplitCandidate BuildRootAndSearch(
      const std::vector<data_size_t>& samples,
      double G_total, double H_total,
      const std::vector<int>& feats, const std::vector<int>& off,
      const std::vector<double>& g, const std::vector<double>& h,
      std::vector<HistEntry>& hist) const;

  // Split: per feature — build the smaller child's slice, subtract it from the
  // parent slice in place (parent buffer becomes the larger child's hist), then
  // sweep both children. Returns the children's best candidates.
  void EvalSplitChildren(
      const std::vector<data_size_t>& small_samples, bool left_small,
      int left_leaf, int right_leaf,
      double GL, double HL, double GR, double HR,
      int64_t n_left, int64_t n_right,
      bool eval_left, bool eval_right,
      const std::vector<int>& feats, const std::vector<int>& off,
      const std::vector<double>& g, const std::vector<double>& h,
      std::vector<HistEntry>& parent_hist, std::vector<HistEntry>& small_hist,
      SplitCandidate& out_left, SplitCandidate& out_right) const;

  // Multi-output analogues (flat [off[fi]*C ..] slices, class-major per bin).
  // cs = per-bin sample counts (class-independent), n_total as above.
  SplitCandidate SweepFeatureMO(
      int leaf_idx, int f, const double* gs, const double* hs,
      const int64_t* cs, int nb, int C,
      const std::vector<double>& G_tot, const std::vector<double>& H_tot,
      int64_t n_total, double score_parent, double min_hess) const;

  SplitCandidate BuildRootAndSearchMO(
      const std::vector<data_size_t>& samples,
      const std::vector<double>& G_tot, const std::vector<double>& H_tot,
      const std::vector<int>& feats, const std::vector<int>& off, int C,
      const std::vector<double>& g_NC, const std::vector<double>& h_NC,
      std::vector<double>& g_hist, std::vector<double>& h_hist,
      std::vector<int64_t>& cnt_hist) const;

  void EvalSplitChildrenMO(
      const std::vector<data_size_t>& small_samples, bool left_small,
      int left_leaf, int right_leaf,
      const std::vector<double>& GL, const std::vector<double>& HL,
      const std::vector<double>& GR, const std::vector<double>& HR,
      int64_t n_left, int64_t n_right,
      bool eval_left, bool eval_right,
      const std::vector<int>& feats, const std::vector<int>& off, int C,
      const std::vector<double>& g_NC, const std::vector<double>& h_NC,
      std::vector<double>& parent_g, std::vector<double>& parent_h,
      std::vector<int64_t>& parent_cnt,
      std::vector<double>& small_g, std::vector<double>& small_h,
      std::vector<int64_t>& small_cnt,
      SplitCandidate& out_left, SplitCandidate& out_right) const;

  // L1 soft-thresholding of the gradient sum (LightGBM-style reg_alpha).
  static double ThresholdL1(double G, double l1) {
    if (l1 <= 0.0) return G;
    if (G >  l1) return G - l1;
    if (G < -l1) return G + l1;
    return 0.0;
  }

  double Score(double G, double H) const {
    double tg = ThresholdL1(G, cfg_.lambda_l1);
    double denom = H + cfg_.lambda_v;
    return (denom > 0) ? tg * tg / denom : 0.0;
  }

  // Minimum hessian mass required on each side of a split.
  double MinHess() const {
    return std::max(cfg_.h_min, cfg_.min_sum_hessian_in_leaf);
  }
};

} // namespace shimaenaga
