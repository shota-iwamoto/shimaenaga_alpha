#include "token_planner.h"
#include "../util/random.h"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace shimaenaga {

namespace {

// Correlation between two bin columns (sampled to at most 100k rows).
// Bin indices are (coarse) quantile ranks, so Pearson on bins approximates
// Spearman rank correlation. Returns a properly normalized signed ρ ∈ [-1,1]
// — the previous formula plugged raw bin values into the Spearman
// n(n²−1) normalization, which returned ≈1.0 for every pair and could never
// go negative (anti-correlated features looked maximally distant).
double BinCorr(const bin_t* a, const bin_t* b, data_size_t n) {
  data_size_t sn = std::min(n, (data_size_t)100000);
  if (sn < 2) return 0.0;
  double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
  for (data_size_t i = 0; i < sn; ++i) {
    double x = a[i], y = b[i];
    sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y;
  }
  double va = saa - sa * sa / sn;
  double vb = sbb - sb * sb / sn;
  double cov = sab - sa * sb / sn;
  if (va < 1e-12 || vb < 1e-12) return 0.0;  // constant column
  return std::max(-1.0, std::min(1.0, cov / std::sqrt(va * vb)));
}

// Average-linkage agglomerative clustering of F items into k groups.
// Lance-Williams update: after merging clusters (i,j), the average-linkage
// distance to any other cluster m is the size-weighted mean
//   d(i∪j, m) = (n_i·d(i,m) + n_j·d(j,m)) / (n_i + n_j),
// so the O(|g_i|·|g_j|) pair rescans of the previous implementation
// (O(F⁴)-ish overall) collapse to an O(F) row update per merge.
std::vector<int> HierarchicalCluster(
    std::vector<std::vector<double>> dist,  // F×F dissimilarity (consumed)
    int F, int k) {
  std::vector<int>  size(F, 1);
  std::vector<bool> active(F, true);
  std::vector<int>  label(F);
  std::iota(label.begin(), label.end(), 0);
  std::vector<std::vector<int>> members(F);
  for (int i = 0; i < F; ++i) members[i] = {i};

  int n_clusters = F;
  while (n_clusters > k) {
    double best = 1e18;
    int bi = -1, bj = -1;
    for (int i = 0; i < F; ++i) {
      if (!active[i]) continue;
      for (int j = i + 1; j < F; ++j) {
        if (!active[j]) continue;
        if (dist[i][j] < best) { best = dist[i][j]; bi = i; bj = j; }
      }
    }
    if (bi < 0) break;
    // Merge bj into bi
    for (int m = 0; m < F; ++m) {
      if (!active[m] || m == bi || m == bj) continue;
      double d = (size[bi] * dist[bi][m] + size[bj] * dist[bj][m]) /
                 (double)(size[bi] + size[bj]);
      dist[bi][m] = dist[m][bi] = d;
    }
    for (int x : members[bj]) { members[bi].push_back(x); label[x] = bi; }
    members[bj].clear();
    size[bi] += size[bj];
    active[bj] = false;
    n_clusters--;
  }
  // Relabel 0..k-1
  std::vector<int> remap(F, -1);
  int cnt = 0;
  for (int i = 0; i < F; ++i) if (active[i]) remap[i] = cnt++;
  std::vector<int> out(F);
  for (int i = 0; i < F; ++i) out[i] = remap[label[i]];
  return out;
}

// Correlation clustering is O(F²·sample + F³); above this many features fall
// back to cheap round-robin grouping instead of stalling Dataset::Build.
constexpr int kMaxClusterFeatures = 512;

}  // anonymous namespace

TokenPlan TokenPlanner::Build(
    const std::vector<std::vector<bin_t>>& feature_bins,
    data_size_t n_samples,
    int F,
    const Config& cfg,
    const std::vector<bool>& is_categorical,
    const std::vector<double>& missing_rates) {

  TokenPlan plan;
  plan.feature_subsets.clear();
  plan.types.clear();

  // tier-0 is pure GBDT (P=1, no attention): the single tree MUST see all
  // features, otherwise the auto plan would restrict it to feature-cluster 0 and
  // cripple it — breaking the LightGBM degeneracy theorem (数学設計書 §10).
  // (token_plan != "auto" is rejected by Config::Validate.)
  if (F == 0 || cfg.tier == 0) {
    int P = 1;
    plan.P = P;
    plan.feature_subsets.resize(P);
    plan.types.resize(P, TokenType::FeatureGroup);
    std::vector<int> all_feats(F);
    std::iota(all_feats.begin(), all_feats.end(), 0);
    for (int p = 0; p < P; ++p) plan.feature_subsets[p] = all_feats;
    plan.mask_bits.resize(P, (1u << P) - 1);
    return plan;
  }

  // Separate numerical and categorical features
  std::vector<int> num_feats, cat_feats;
  for (int f = 0; f < F; ++f) {
    if (is_categorical[f]) cat_feats.push_back(f);
    else num_feats.push_back(f);
  }

  int P_target = cfg.num_tokens;
  int n_group = (P_target + 1) / 2;  // ⌈P/2⌉ numeric groups

  // Cluster numerical features by |ρ| of their bin columns (数学設計書 §3.3).
  std::vector<int> cluster_labels(num_feats.size(), 0);
  if (!num_feats.empty()) {
    int nf = static_cast<int>(num_feats.size());
    int k = std::min(n_group, nf);
    if (k > 1 && nf > 1 && nf <= kMaxClusterFeatures) {
      std::vector<std::vector<double>> dist(nf, std::vector<double>(nf, 0.0));
      for (int i = 0; i < nf; ++i)
        for (int j = i + 1; j < nf; ++j) {
          double c = std::abs(BinCorr(
              feature_bins[num_feats[i]].data(),
              feature_bins[num_feats[j]].data(), n_samples));
          dist[i][j] = dist[j][i] = 1.0 - c;
        }
      cluster_labels = HierarchicalCluster(std::move(dist), nf, k);
    } else if (k > 1) {
      // Too many features for O(F²) correlations: round-robin fallback.
      for (int i = 0; i < nf; ++i) cluster_labels[i] = i % k;
    }
    // Build feature groups for each cluster
    std::vector<std::vector<int>> clusters(n_group);
    for (int i = 0; i < nf; ++i)
      clusters[cluster_labels[i] % n_group].push_back(num_feats[i]);
    for (auto& grp : clusters) {
      if (!grp.empty()) {
        plan.feature_subsets.push_back(grp);
        plan.types.push_back(TokenType::FeatureGroup);
      }
    }
  }

  // Categorical token
  if (!cat_feats.empty() && (int)plan.feature_subsets.size() < P_target) {
    plan.feature_subsets.push_back(cat_feats);
    plan.types.push_back(TokenType::FeatureGroup);
  }

  // Missingness token: features with a meaningful missing rate. The NaN-vs-not
  // signal survives binning (bin 0 = missing), so a tree over these features
  // can split purely on missingness patterns.
  double miss_thresh = 0.05;
  std::vector<int> miss_feats;
  for (int f = 0; f < F; ++f)
    if (missing_rates[f] > miss_thresh) miss_feats.push_back(f);
  int missing_token_idx = -1;
  if (!miss_feats.empty() && cfg.missing_token != "off" &&
      (int)plan.feature_subsets.size() < P_target) {
    missing_token_idx = (int)plan.feature_subsets.size();
    plan.feature_subsets.push_back(miss_feats);
    plan.types.push_back(TokenType::Missingness);
  }

  // Fill remaining slots with interaction tokens.
  // The FIRST interaction token gets the FULL feature set: the correlation
  // clustering above is essentially arbitrary on weakly-correlated features and
  // can place interacting features (e.g. an x0·x1 target term) in different
  // tokens — then NO token tree can split on both and the interaction becomes
  // unrepresentable (observed as tier-1/2 losing badly to tier-0 on ~1/3 of
  // Friedman#1 draws). A full token guarantees tier-1/2 capacity ⊇ tier-0.
  // Remaining slots are random subsets, seeded by cfg.seed (was a fixed 42,
  // which made the plan identical across model seeds — no escape hatch).
  Random rng(cfg.seed * 0x9e3779b97f4a7c15ull + 42);
  std::vector<int> all_feats(F);
  std::iota(all_feats.begin(), all_feats.end(), 0);
  bool first_interaction = true;
  while ((int)plan.feature_subsets.size() < P_target) {
    if (first_interaction) {
      plan.feature_subsets.push_back(all_feats);
      first_interaction = false;
    } else {
      int k = std::max(1, static_cast<int>(F * cfg.token_feature_fraction));
      plan.feature_subsets.push_back(rng.SampleK(F, k));
    }
    plan.types.push_back(TokenType::Interaction);
  }

  plan.P = static_cast<int>(plan.feature_subsets.size());

  // ── Attention mask (基本設計書 §5.4) ──
  // full: N(p) = all tokens.
  // feature_local: N(p) = tokens sharing at least one feature with p, plus p
  // itself and the missingness token (missingness context is globally useful).
  // Interaction tokens overlap widely by construction, so they stay connected.
  plan.mask_bits.assign(plan.P, 0u);
  uint32_t full_mask = (plan.P >= 32) ? ~0u : ((1u << plan.P) - 1u);
  if (cfg.attn_mask == "full" || cfg.tier < 2) {
    for (int p = 0; p < plan.P; ++p) plan.mask_bits[p] = full_mask;
  } else {
    // Feature bitsets for overlap tests
    int words = (F + 63) / 64;
    std::vector<std::vector<uint64_t>> fbits(plan.P, std::vector<uint64_t>(words, 0));
    for (int p = 0; p < plan.P; ++p)
      for (int f : plan.feature_subsets[p])
        fbits[p][f >> 6] |= (1ull << (f & 63));
    for (int p = 0; p < plan.P; ++p) {
      uint32_t m = 1u << p;  // self always allowed
      if (missing_token_idx >= 0) m |= 1u << missing_token_idx;
      for (int r = 0; r < plan.P; ++r) {
        if (r == p) continue;
        bool overlap = false;
        for (int w = 0; w < words && !overlap; ++w)
          overlap = (fbits[p][w] & fbits[r][w]) != 0;
        if (overlap) m |= 1u << r;
      }
      plan.mask_bits[p] = m;
    }
    if (missing_token_idx >= 0)
      plan.mask_bits[missing_token_idx] = full_mask;  // missingness sees all
  }

  return plan;
}

} // namespace shimaenaga
