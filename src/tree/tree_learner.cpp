#include "tree_learner.h"
#include <algorithm>
#include <cassert>

namespace shimaenaga {

TreeLearner::TreeLearner(const Config& cfg,
                         const std::vector<std::vector<bin_t>>& bins,
                         data_size_t n_samples,
                         int n_features)
    : cfg_(cfg), bins_(bins), n_(n_samples), F_(n_features) {
  // Precompute per-feature bin count once (O(N·F)); the split search reuses it
  // instead of rescanning each leaf's samples for the max bin on every split.
  feat_nbins_.assign(F_, 2);
  for (int f = 0; f < F_; ++f) {
    int nb = 2;
    const bin_t* col = bins_[f].data();
    for (data_size_t i = 0; i < n_; ++i)
      if ((int)col[i] + 1 > nb) nb = (int)col[i] + 1;
    feat_nbins_[f] = std::min(nb, 256);
  }
}

// ─── Scalar sweep / fused build+search ─────────────────────────────────────

TreeLearner::SplitCandidate TreeLearner::SweepFeature(
    int leaf_idx, int f, const HistEntry* hs, int nb,
    double G_total, double H_total, int64_t n_total,
    double score_parent, double min_hess) const {

  SplitCandidate fb;
  fb.gain = -1e18; fb.feature = -1; fb.leaf_idx = leaf_idx;
  const int64_t min_data = cfg_.min_data_in_leaf;
  const int64_t c0 = hs[0].cnt;  // NaN bin count

  // Sweep bins 1..nb-2 (bin 0 = NaN, assigned via default_left).
  double GL = 0.0, HL = 0.0;
  int64_t CL = 0;
  for (int b = 1; b < nb - 1; ++b) {
    GL += hs[b].g; HL += hs[b].h; CL += hs[b].cnt;
    if (HL < min_hess) continue;
    double GR = G_total - GL - hs[0].g;
    double HR = H_total - HL - hs[0].h;
    int64_t CR = n_total - CL - c0;
    int def_left = (HR >= HL) ? 1 : 0;
    // min_data_in_leaf with NaN routed to the default side
    int64_t CLf = CL + (def_left ? c0 : 0);
    int64_t CRf = CR + (def_left ? 0 : c0);
    if (CLf < min_data || CRf < min_data) continue;
    if (def_left == 1) { GL += hs[0].g; HL += hs[0].h; }
    else               { GR += hs[0].g; HR += hs[0].h; }
    if (HL < min_hess || HR < min_hess) {
      if (def_left == 1) { GL -= hs[0].g; HL -= hs[0].h; }
      else               { GR -= hs[0].g; HR -= hs[0].h; }
      continue;
    }
    double gain = 0.5 * (Score(GL, HL) +
                         Score(GR, HR) -
                         score_parent) - cfg_.min_gain_to_split;
    if (gain > fb.gain) {
      fb.gain = gain; fb.leaf_idx = leaf_idx; fb.feature = f;
      fb.threshold_bin = b; fb.default_left = def_left;
      fb.is_cat = false; fb.cat_bitset = 0;
    }
    if (def_left == 1) { GL -= hs[0].g; HL -= hs[0].h; }
    else               { GR -= hs[0].g; HR -= hs[0].h; }
  }
  return fb;
}

TreeLearner::SplitCandidate TreeLearner::BuildRootAndSearch(
    const std::vector<data_size_t>& samples,
    double G_total, double H_total,
    const std::vector<int>& feats, const std::vector<int>& off,
    const std::vector<double>& g, const std::vector<double>& h,
    std::vector<HistEntry>& hist) const {

  int nf = (int)feats.size();
  double score_parent = Score(G_total, H_total);
  const double min_hess = MinHess();
  std::vector<SplitCandidate> feat_best(nf);

  // One region: each feature builds its own slice then sweeps it immediately
  // (slice is still hot in cache). Deterministic: disjoint writes + fixed merge.
  #pragma omp parallel for schedule(dynamic)
  for (int fi = 0; fi < nf; ++fi) {
    HistEntry* hs = hist.data() + off[fi];
    const bin_t* col = bins_[feats[fi]].data();
    for (data_size_t i : samples) {
      hs[col[i]].g += g[i];
      hs[col[i]].h += h[i];
      hs[col[i]].cnt++;
    }
    feat_best[fi] = SweepFeature(0, feats[fi], hs, off[fi + 1] - off[fi],
                                 G_total, H_total, (int64_t)samples.size(),
                                 score_parent, min_hess);
  }

  SplitCandidate best;
  best.gain = -1e18; best.leaf_idx = 0; best.feature = -1;
  for (int fi = 0; fi < nf; ++fi)
    if (feat_best[fi].feature >= 0 && feat_best[fi].gain > best.gain)
      best = feat_best[fi];
  return best;
}

void TreeLearner::EvalSplitChildren(
    const std::vector<data_size_t>& small_samples, bool left_small,
    int left_leaf, int right_leaf,
    double GL, double HL, double GR, double HR,
    int64_t n_left, int64_t n_right,
    bool eval_left, bool eval_right,
    const std::vector<int>& feats, const std::vector<int>& off,
    const std::vector<double>& g, const std::vector<double>& h,
    std::vector<HistEntry>& parent_hist, std::vector<HistEntry>& small_hist,
    SplitCandidate& out_left, SplitCandidate& out_right) const {

  int nf = (int)feats.size();
  const double min_hess = MinHess();
  double score_l = Score(GL, HL), score_r = Score(GR, HR);
  std::vector<SplitCandidate> best_l(nf), best_r(nf);

  #pragma omp parallel for schedule(dynamic)
  for (int fi = 0; fi < nf; ++fi) {
    HistEntry* ps = parent_hist.data() + off[fi];
    HistEntry* ss = small_hist.data() + off[fi];
    int nb = off[fi + 1] - off[fi];
    // Build the smaller child's slice.
    const bin_t* col = bins_[feats[fi]].data();
    for (data_size_t i : small_samples) {
      ss[col[i]].g += g[i];
      ss[col[i]].h += h[i];
      ss[col[i]].cnt++;
    }
    // Larger child = parent − smaller (in place; parent buffer is repurposed).
    for (int b = 0; b < nb; ++b) {
      ps[b].g -= ss[b].g; ps[b].h -= ss[b].h; ps[b].cnt -= ss[b].cnt;
    }

    const HistEntry* lh = left_small ? ss : ps;
    const HistEntry* rh = left_small ? ps : ss;
    SplitCandidate skip; skip.gain = -1e18; skip.feature = -1;
    best_l[fi] = eval_left
        ? SweepFeature(left_leaf, feats[fi], lh, nb, GL, HL, n_left, score_l, min_hess)
        : skip;
    best_r[fi] = eval_right
        ? SweepFeature(right_leaf, feats[fi], rh, nb, GR, HR, n_right, score_r, min_hess)
        : skip;
  }

  out_left.gain = -1e18;  out_left.feature = -1;  out_left.leaf_idx = left_leaf;
  out_right.gain = -1e18; out_right.feature = -1; out_right.leaf_idx = right_leaf;
  for (int fi = 0; fi < nf; ++fi) {
    if (best_l[fi].feature >= 0 && best_l[fi].gain > out_left.gain)  out_left  = best_l[fi];
    if (best_r[fi].feature >= 0 && best_r[fi].gain > out_right.gain) out_right = best_r[fi];
  }
}

Tree TreeLearner::Grow(
    const std::vector<double>& g_tilde,
    const std::vector<double>& h_tilde,
    const std::vector<int>& feature_subset,
    int max_leaves,
    const std::vector<bool>& sample_mask) {

  Tree tree;
  tree.num_leaves = 1;

  std::vector<data_size_t> root_samples;
  root_samples.reserve(n_);
  for (data_size_t i = 0; i < n_; ++i)
    if (sample_mask.empty() || sample_mask[i]) root_samples.push_back(i);

  if (max_leaves <= 1 || (int)root_samples.size() < cfg_.min_data_in_leaf * 2)
    return tree;

  // Flat histogram layout over the feature subset.
  int nf = (int)feature_subset.size();
  std::vector<int> off(nf + 1, 0);
  for (int fi = 0; fi < nf; ++fi)
    off[fi + 1] = off[fi] + feat_nbins_[feature_subset[fi]];
  const int total_bins = off[nf];

  // Per-leaf state: samples, histogram, G/H totals.
  std::vector<std::vector<data_size_t>> leaf_samples = {root_samples};
  std::vector<std::vector<HistEntry>> leaf_hist(max_leaves + 1);
  std::vector<double> leaf_G(max_leaves + 1, 0.0), leaf_H(max_leaves + 1, 0.0);

  double G0 = 0.0, H0 = 0.0;
  for (data_size_t i : root_samples) { G0 += g_tilde[i]; H0 += h_tilde[i]; }
  leaf_G[0] = G0; leaf_H[0] = H0;

  // leaf_parent[leaf_id] = node index of parent (-1 = root of tree)
  // leaf_is_left[leaf_id] = whether this leaf is the left child of parent
  std::vector<int>  leaf_parent(max_leaves + 1, -1);
  std::vector<bool> leaf_is_left(max_leaves + 1, false);
  std::vector<int>  leaf_depth(max_leaves + 1, 0);

  // Node arrays (tree nodes in order of creation)
  std::vector<int>     node_feature;
  std::vector<bin_t>   node_thresh;
  std::vector<uint8_t> node_def_left;
  std::vector<int>     node_lc, node_rc;  // left/right child
  std::vector<uint8_t> node_is_cat;
  std::vector<uint64_t> node_cat_bs;

  int root_node_idx = -1;  // which node is the tree root (-1 = leaf 0)

  // Priority queue by gain
  auto cmp = [](const SplitCandidate& a, const SplitCandidate& b) {
    if (a.gain != b.gain) return a.gain < b.gain;
    // Tie-break for determinism (T10): feature, then threshold
    if (a.feature != b.feature) return a.feature > b.feature;
    return a.threshold_bin > b.threshold_bin;
  };
  std::priority_queue<SplitCandidate, std::vector<SplitCandidate>, decltype(cmp)> pq(cmp);

  leaf_hist[0].assign(total_bins, HistEntry{});
  auto cand = BuildRootAndSearch(root_samples, G0, H0, feature_subset, off,
                                 g_tilde, h_tilde, leaf_hist[0]);
  if (cand.gain > 0) pq.push(cand);

  while (!pq.empty() && tree.num_leaves < max_leaves) {
    SplitCandidate best = pq.top(); pq.pop();
    if (best.gain <= 0 || best.feature < 0) break;

    int li = best.leaf_idx;
    const auto& smp = leaf_samples[li];
    if ((int)smp.size() < cfg_.min_data_in_leaf * 2) continue;

    // Split samples
    std::vector<data_size_t> left_smp, right_smp;
    const bin_t* col = bins_[best.feature].data();
    for (data_size_t i : smp) {
      bin_t b = col[i];
      bool go_left = (b == 0) ? (best.default_left == 1)
                               : (b <= best.threshold_bin);
      if (go_left) left_smp.push_back(i);
      else          right_smp.push_back(i);
    }
    if ((int)left_smp.size() < cfg_.min_data_in_leaf ||
        (int)right_smp.size() < cfg_.min_data_in_leaf) continue;

    // Create a new node
    int node_id = (int)node_feature.size();
    node_feature.push_back(best.feature);
    node_thresh.push_back(static_cast<bin_t>(best.threshold_bin));
    node_def_left.push_back(static_cast<uint8_t>(best.default_left));
    node_is_cat.push_back(0);
    node_cat_bs.push_back(0);

    int right_leaf = tree.num_leaves;
    tree.num_leaves++;

    // Left child: reuse leaf index li
    // Right child: new leaf index right_leaf
    node_lc.push_back(~li);
    node_rc.push_back(~right_leaf);

    // Update parent's pointer from ~li to this node_id
    if (leaf_parent[li] == -1) {
      // li was the root leaf -> this node becomes tree root
      root_node_idx = node_id;
    } else {
      int pn = leaf_parent[li];
      if (leaf_is_left[li]) node_lc[pn] = node_id;
      else                  node_rc[pn] = node_id;
    }

    // Child totals: sum the smaller side, derive the larger by subtraction.
    const bool left_small = left_smp.size() <= right_smp.size();
    const auto& small_smp = left_small ? left_smp : right_smp;
    double Gs = 0.0, Hs = 0.0;
    for (data_size_t i : small_smp) { Gs += g_tilde[i]; Hs += h_tilde[i]; }
    double GL_c = left_small ? Gs : leaf_G[li] - Gs;
    double HL_c = left_small ? Hs : leaf_H[li] - Hs;
    double GR_c = left_small ? leaf_G[li] - Gs : Gs;
    double HR_c = left_small ? leaf_H[li] - Hs : Hs;

    // Depth/size gating for the children's candidate search.
    int child_depth = leaf_depth[li] + 1;
    bool child_depth_ok = (cfg_.max_depth < 0 || child_depth < cfg_.max_depth);
    bool eval_l = child_depth_ok && (int)left_smp.size()  >= cfg_.min_data_in_leaf * 2;
    bool eval_r = child_depth_ok && (int)right_smp.size() >= cfg_.min_data_in_leaf * 2;

    // Histogram subtraction (§11.2), one fused parallel region per split.
    std::vector<HistEntry> small_hist(total_bins, HistEntry{});
    std::vector<HistEntry> parent_hist = std::move(leaf_hist[li]);
    SplitCandidate cl, cr;
    EvalSplitChildren(small_smp, left_small, li, right_leaf,
                      GL_c, HL_c, GR_c, HR_c,
                      (int64_t)left_smp.size(), (int64_t)right_smp.size(),
                      eval_l, eval_r,
                      feature_subset, off, g_tilde, h_tilde,
                      parent_hist, small_hist, cl, cr);
    if (left_small) {
      leaf_hist[li]         = std::move(small_hist);
      leaf_hist[right_leaf] = std::move(parent_hist);
    } else {
      leaf_hist[li]         = std::move(parent_hist);
      leaf_hist[right_leaf] = std::move(small_hist);
    }
    leaf_G[li] = GL_c; leaf_H[li] = HL_c;
    leaf_G[right_leaf] = GR_c; leaf_H[right_leaf] = HR_c;

    // Update leaf metadata
    leaf_samples[li] = left_smp;
    leaf_samples.push_back(right_smp);  // index = right_leaf

    leaf_parent[right_leaf] = node_id;
    leaf_is_left[right_leaf] = false;
    leaf_parent[li] = node_id;
    leaf_is_left[li] = true;
    leaf_depth[right_leaf] = child_depth;
    leaf_depth[li] = child_depth;

    if (cl.gain > 0 && cl.feature >= 0) pq.push(cl);
    if (cr.gain > 0 && cr.feature >= 0) pq.push(cr);
  }

  // Finalize tree structure
  if (node_feature.empty()) return tree;  // no splits

  tree.split_feature  = node_feature;
  tree.threshold_bin  = node_thresh;
  tree.default_left   = node_def_left;
  tree.is_categorical = node_is_cat;
  tree.cat_bitset     = node_cat_bs;

  // Rotate so that root_node_idx is node 0 (GetLeaf starts at node 0).
  if (root_node_idx != 0 && root_node_idx >= 0) {
    int ri = root_node_idx;
    auto swap_node = [&](int a, int b) {
      std::swap(tree.split_feature[a], tree.split_feature[b]);
      std::swap(tree.threshold_bin[a], tree.threshold_bin[b]);
      std::swap(tree.default_left[a], tree.default_left[b]);
      std::swap(tree.is_categorical[a], tree.is_categorical[b]);
      std::swap(tree.cat_bitset[a], tree.cat_bitset[b]);
      std::swap(node_lc[a], node_lc[b]);
      std::swap(node_rc[a], node_rc[b]);
    };
    swap_node(0, ri);
    // Fix all references to 0 and ri
    for (int n = 0; n < (int)node_lc.size(); ++n) {
      auto fix = [&](int& c) {
        if (c == 0) c = ri;
        else if (c == ri) c = 0;
      };
      if (node_lc[n] >= 0) fix(node_lc[n]);
      if (node_rc[n] >= 0) fix(node_rc[n]);
    }
  }

  tree.left_child  = node_lc;
  tree.right_child = node_rc;
  return tree;
}

// ─── Multi-output sweep / fused build+search ───────────────────────────────

TreeLearner::SplitCandidate TreeLearner::SweepFeatureMO(
    int leaf_idx, int f, const double* gs, const double* hs,
    const int64_t* cs, int nb, int C,
    const std::vector<double>& G_tot, const std::vector<double>& H_tot,
    int64_t n_total, double score_parent, double min_hess) const {

  SplitCandidate fb;
  fb.gain = -1e18; fb.feature = -1; fb.leaf_idx = leaf_idx;
  const int64_t min_data = cfg_.min_data_in_leaf;
  const int64_t c0 = cs[0];  // NaN bin count

  std::vector<double> GL(C, 0.0), HL(C, 0.0), GR(C), HR(C);
  int64_t CL = 0;

  for (int b = 1; b < nb - 1; ++b) {
    CL += cs[b];
    for (int k = 0; k < C; ++k) {
      GL[k] += gs[(size_t)b * C + k];
      HL[k] += hs[(size_t)b * C + k];
    }

    double H_L_sum = 0.0;
    for (int k = 0; k < C; ++k) H_L_sum += HL[k];
    if (H_L_sum < min_hess) continue;

    for (int k = 0; k < C; ++k) {
      GR[k] = G_tot[k] - GL[k] - gs[k];
      HR[k] = H_tot[k] - HL[k] - hs[k];
    }
    double H_R_sum = 0.0;
    for (int k = 0; k < C; ++k) H_R_sum += HR[k];

    // Add NaN (bin 0) to larger side
    int def_left = (H_R_sum >= H_L_sum) ? 1 : 0;
    // min_data_in_leaf with NaN routed to the default side
    {
      int64_t CR = n_total - CL - c0;
      int64_t CLf = CL + (def_left ? c0 : 0);
      int64_t CRf = CR + (def_left ? 0 : c0);
      if (CLf < min_data || CRf < min_data) continue;
    }
    if (def_left == 1)
      for (int k = 0; k < C; ++k) { GL[k] += gs[k]; HL[k] += hs[k]; }
    else
      for (int k = 0; k < C; ++k) { GR[k] += gs[k]; HR[k] += hs[k]; }

    double HL_s = 0.0, HR_s = 0.0;
    for (int k = 0; k < C; ++k) { HL_s += HL[k]; HR_s += HR[k]; }
    if (HL_s < min_hess || HR_s < min_hess) {
      if (def_left == 1)
        for (int k = 0; k < C; ++k) { GL[k] -= gs[k]; HL[k] -= hs[k]; }
      else
        for (int k = 0; k < C; ++k) { GR[k] -= gs[k]; HR[k] -= hs[k]; }
      continue;
    }

    // Multi-output gain = Σ_k [score_L[k] + score_R[k]] - score_parent
    double gain = 0.0;
    for (int k = 0; k < C; ++k)
      gain += Score(GL[k], HL[k]) + Score(GR[k], HR[k]);
    gain = 0.5 * (gain - score_parent) - cfg_.min_gain_to_split;

    if (gain > fb.gain) {
      fb.gain = gain;
      fb.leaf_idx = leaf_idx;
      fb.feature = f;
      fb.threshold_bin = b;
      fb.default_left = def_left;
      fb.is_cat = false;
      fb.cat_bitset = 0;
    }

    if (def_left == 1)
      for (int k = 0; k < C; ++k) { GL[k] -= gs[k]; HL[k] -= hs[k]; }
    else
      for (int k = 0; k < C; ++k) { GR[k] -= gs[k]; HR[k] -= hs[k]; }
  }
  return fb;
}

TreeLearner::SplitCandidate TreeLearner::BuildRootAndSearchMO(
    const std::vector<data_size_t>& samples,
    const std::vector<double>& G_tot, const std::vector<double>& H_tot,
    const std::vector<int>& feats, const std::vector<int>& off, int C,
    const std::vector<double>& g_NC, const std::vector<double>& h_NC,
    std::vector<double>& g_hist, std::vector<double>& h_hist,
    std::vector<int64_t>& cnt_hist) const {

  int nf = (int)feats.size();
  double score_parent = 0.0;
  for (int k = 0; k < C; ++k) score_parent += Score(G_tot[k], H_tot[k]);
  const double min_hess = MinHess();
  std::vector<SplitCandidate> feat_best(nf);

  #pragma omp parallel for schedule(dynamic)
  for (int fi = 0; fi < nf; ++fi) {
    double* gs = g_hist.data() + (size_t)off[fi] * C;
    double* hs = h_hist.data() + (size_t)off[fi] * C;
    int64_t* cs = cnt_hist.data() + off[fi];
    const bin_t* col = bins_[feats[fi]].data();
    for (data_size_t i : samples) {
      int b = col[i];
      cs[b]++;
      for (int k = 0; k < C; ++k) {
        gs[(size_t)b * C + k] += g_NC[(size_t)i * C + k];
        hs[(size_t)b * C + k] += h_NC[(size_t)i * C + k];
      }
    }
    feat_best[fi] = SweepFeatureMO(0, feats[fi], gs, hs, cs, off[fi + 1] - off[fi], C,
                                   G_tot, H_tot, (int64_t)samples.size(),
                                   score_parent, min_hess);
  }

  SplitCandidate best;
  best.gain = -1e18; best.leaf_idx = 0; best.feature = -1;
  for (int fi = 0; fi < nf; ++fi)
    if (feat_best[fi].feature >= 0 && feat_best[fi].gain > best.gain)
      best = feat_best[fi];
  return best;
}

void TreeLearner::EvalSplitChildrenMO(
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
    SplitCandidate& out_left, SplitCandidate& out_right) const {

  int nf = (int)feats.size();
  const double min_hess = MinHess();
  double score_l = 0.0, score_r = 0.0;
  for (int k = 0; k < C; ++k) {
    score_l += Score(GL[k], HL[k]);
    score_r += Score(GR[k], HR[k]);
  }
  std::vector<SplitCandidate> best_l(nf), best_r(nf);

  #pragma omp parallel for schedule(dynamic)
  for (int fi = 0; fi < nf; ++fi) {
    double* pg = parent_g.data() + (size_t)off[fi] * C;
    double* ph = parent_h.data() + (size_t)off[fi] * C;
    int64_t* pc = parent_cnt.data() + off[fi];
    double* sg = small_g.data() + (size_t)off[fi] * C;
    double* sh = small_h.data() + (size_t)off[fi] * C;
    int64_t* sc = small_cnt.data() + off[fi];
    int nb = off[fi + 1] - off[fi];
    const bin_t* col = bins_[feats[fi]].data();
    for (data_size_t i : small_samples) {
      int b = col[i];
      sc[b]++;
      for (int k = 0; k < C; ++k) {
        sg[(size_t)b * C + k] += g_NC[(size_t)i * C + k];
        sh[(size_t)b * C + k] += h_NC[(size_t)i * C + k];
      }
    }
    for (size_t t = 0; t < (size_t)nb * C; ++t) { pg[t] -= sg[t]; ph[t] -= sh[t]; }
    for (int b = 0; b < nb; ++b) pc[b] -= sc[b];

    const double* lg = left_small ? sg : pg;
    const double* lh = left_small ? sh : ph;
    const int64_t* lc = left_small ? sc : pc;
    const double* rg = left_small ? pg : sg;
    const double* rh = left_small ? ph : sh;
    const int64_t* rc = left_small ? pc : sc;
    SplitCandidate skip; skip.gain = -1e18; skip.feature = -1;
    best_l[fi] = eval_left
        ? SweepFeatureMO(left_leaf, feats[fi], lg, lh, lc, nb, C, GL, HL,
                         n_left, score_l, min_hess)
        : skip;
    best_r[fi] = eval_right
        ? SweepFeatureMO(right_leaf, feats[fi], rg, rh, rc, nb, C, GR, HR,
                         n_right, score_r, min_hess)
        : skip;
  }

  out_left.gain = -1e18;  out_left.feature = -1;  out_left.leaf_idx = left_leaf;
  out_right.gain = -1e18; out_right.feature = -1; out_right.leaf_idx = right_leaf;
  for (int fi = 0; fi < nf; ++fi) {
    if (best_l[fi].feature >= 0 && best_l[fi].gain > out_left.gain)  out_left  = best_l[fi];
    if (best_r[fi].feature >= 0 && best_r[fi].gain > out_right.gain) out_right = best_r[fi];
  }
}

// Multi-output Grow (same logic as Grow but with MO histograms)
Tree TreeLearner::GrowMultiOutput(
    const std::vector<double>& g_NC,
    const std::vector<double>& h_NC,
    int C,
    const std::vector<int>& feature_subset,
    int max_leaves,
    const std::vector<bool>& sample_mask) {

  Tree tree;
  tree.num_leaves = 1;

  std::vector<data_size_t> root_samples;
  root_samples.reserve(n_);
  for (data_size_t i = 0; i < n_; ++i)
    if (sample_mask.empty() || sample_mask[i]) root_samples.push_back(i);

  if (max_leaves <= 1 || (int)root_samples.size() < cfg_.min_data_in_leaf * 2)
    return tree;

  int nf = (int)feature_subset.size();
  std::vector<int> off(nf + 1, 0);
  for (int fi = 0; fi < nf; ++fi)
    off[fi + 1] = off[fi] + feat_nbins_[feature_subset[fi]];
  const size_t total = (size_t)off[nf] * C;

  std::vector<std::vector<data_size_t>> leaf_samples = {root_samples};
  std::vector<std::vector<double>> leaf_ghist(max_leaves + 1), leaf_hhist(max_leaves + 1);
  std::vector<std::vector<int64_t>> leaf_chist(max_leaves + 1);
  std::vector<std::vector<double>> leaf_G(max_leaves + 1), leaf_H(max_leaves + 1);

  leaf_G[0].assign(C, 0.0); leaf_H[0].assign(C, 0.0);
  for (data_size_t i : root_samples)
    for (int k = 0; k < C; ++k) {
      leaf_G[0][k] += g_NC[(size_t)i * C + k];
      leaf_H[0][k] += h_NC[(size_t)i * C + k];
    }

  std::vector<int>  leaf_parent(max_leaves + 1, -1);
  std::vector<bool> leaf_is_left(max_leaves + 1, false);
  std::vector<int>  leaf_depth(max_leaves + 1, 0);

  std::vector<int>     node_feature;
  std::vector<bin_t>   node_thresh;
  std::vector<uint8_t> node_def_left;
  std::vector<int>     node_lc, node_rc;
  std::vector<uint8_t> node_is_cat;
  std::vector<uint64_t> node_cat_bs;

  int root_node_idx = -1;

  auto cmp = [](const SplitCandidate& a, const SplitCandidate& b) {
    if (a.gain != b.gain) return a.gain < b.gain;
    if (a.feature != b.feature) return a.feature > b.feature;
    return a.threshold_bin > b.threshold_bin;
  };
  std::priority_queue<SplitCandidate, std::vector<SplitCandidate>, decltype(cmp)> pq(cmp);

  leaf_ghist[0].assign(total, 0.0);
  leaf_hhist[0].assign(total, 0.0);
  leaf_chist[0].assign(off[nf], 0);
  auto cand = BuildRootAndSearchMO(root_samples, leaf_G[0], leaf_H[0], feature_subset,
                                   off, C, g_NC, h_NC, leaf_ghist[0], leaf_hhist[0],
                                   leaf_chist[0]);
  if (cand.gain > 0) pq.push(cand);

  while (!pq.empty() && tree.num_leaves < max_leaves) {
    SplitCandidate best = pq.top(); pq.pop();
    if (best.gain <= 0 || best.feature < 0) break;

    int li = best.leaf_idx;
    const auto& smp = leaf_samples[li];
    if ((int)smp.size() < cfg_.min_data_in_leaf * 2) continue;

    std::vector<data_size_t> left_smp, right_smp;
    const bin_t* col = bins_[best.feature].data();
    for (data_size_t i : smp) {
      bin_t b = col[i];
      bool go_left = (b == 0) ? (best.default_left == 1) : (b <= best.threshold_bin);
      if (go_left) left_smp.push_back(i);
      else          right_smp.push_back(i);
    }
    if ((int)left_smp.size() < cfg_.min_data_in_leaf ||
        (int)right_smp.size() < cfg_.min_data_in_leaf) continue;

    int node_id = (int)node_feature.size();
    node_feature.push_back(best.feature);
    node_thresh.push_back(static_cast<bin_t>(best.threshold_bin));
    node_def_left.push_back(static_cast<uint8_t>(best.default_left));
    node_is_cat.push_back(0);
    node_cat_bs.push_back(0);

    int right_leaf = tree.num_leaves++;
    node_lc.push_back(~li);
    node_rc.push_back(~right_leaf);

    if (leaf_parent[li] == -1) root_node_idx = node_id;
    else {
      int pn = leaf_parent[li];
      if (leaf_is_left[li]) node_lc[pn] = node_id;
      else                  node_rc[pn] = node_id;
    }

    // Child totals: sum the smaller side, derive the larger by subtraction.
    const bool left_small = left_smp.size() <= right_smp.size();
    const auto& small_smp = left_small ? left_smp : right_smp;
    std::vector<double> Gs(C, 0.0), Hs(C, 0.0);
    for (data_size_t i : small_smp)
      for (int k = 0; k < C; ++k) {
        Gs[k] += g_NC[(size_t)i * C + k];
        Hs[k] += h_NC[(size_t)i * C + k];
      }
    std::vector<double> GL_c(C), HL_c(C), GR_c(C), HR_c(C);
    for (int k = 0; k < C; ++k) {
      GL_c[k] = left_small ? Gs[k] : leaf_G[li][k] - Gs[k];
      HL_c[k] = left_small ? Hs[k] : leaf_H[li][k] - Hs[k];
      GR_c[k] = left_small ? leaf_G[li][k] - Gs[k] : Gs[k];
      HR_c[k] = left_small ? leaf_H[li][k] - Hs[k] : Hs[k];
    }

    int child_depth = leaf_depth[li] + 1;
    bool child_depth_ok = (cfg_.max_depth < 0 || child_depth < cfg_.max_depth);
    bool eval_l = child_depth_ok && (int)left_smp.size()  >= cfg_.min_data_in_leaf * 2;
    bool eval_r = child_depth_ok && (int)right_smp.size() >= cfg_.min_data_in_leaf * 2;

    // Histogram subtraction (§11.2), multi-output variant, one fused region.
    std::vector<double> small_g(total, 0.0), small_h(total, 0.0);
    std::vector<int64_t> small_c(off[nf], 0);
    std::vector<double> parent_g = std::move(leaf_ghist[li]);
    std::vector<double> parent_h = std::move(leaf_hhist[li]);
    std::vector<int64_t> parent_c = std::move(leaf_chist[li]);
    SplitCandidate cl, cr;
    EvalSplitChildrenMO(small_smp, left_small, li, right_leaf,
                        GL_c, HL_c, GR_c, HR_c,
                        (int64_t)left_smp.size(), (int64_t)right_smp.size(),
                        eval_l, eval_r,
                        feature_subset, off, C, g_NC, h_NC,
                        parent_g, parent_h, parent_c,
                        small_g, small_h, small_c, cl, cr);
    if (left_small) {
      leaf_ghist[li] = std::move(small_g);          leaf_hhist[li] = std::move(small_h);
      leaf_chist[li] = std::move(small_c);
      leaf_ghist[right_leaf] = std::move(parent_g); leaf_hhist[right_leaf] = std::move(parent_h);
      leaf_chist[right_leaf] = std::move(parent_c);
    } else {
      leaf_ghist[li] = std::move(parent_g);         leaf_hhist[li] = std::move(parent_h);
      leaf_chist[li] = std::move(parent_c);
      leaf_ghist[right_leaf] = std::move(small_g);  leaf_hhist[right_leaf] = std::move(small_h);
      leaf_chist[right_leaf] = std::move(small_c);
    }
    leaf_G[li] = GL_c; leaf_H[li] = HL_c;
    leaf_G[right_leaf] = GR_c; leaf_H[right_leaf] = HR_c;

    leaf_samples[li] = left_smp;
    leaf_samples.push_back(right_smp);
    leaf_parent[right_leaf] = node_id;
    leaf_is_left[right_leaf] = false;
    leaf_parent[li] = node_id;
    leaf_is_left[li] = true;
    leaf_depth[right_leaf] = child_depth;
    leaf_depth[li] = child_depth;

    if (cl.gain > 0 && cl.feature >= 0) pq.push(cl);
    if (cr.gain > 0 && cr.feature >= 0) pq.push(cr);
  }

  if (node_feature.empty()) return tree;

  tree.split_feature  = node_feature;
  tree.threshold_bin  = node_thresh;
  tree.default_left   = node_def_left;
  tree.is_categorical = node_is_cat;
  tree.cat_bitset     = node_cat_bs;

  if (root_node_idx != 0 && root_node_idx >= 0) {
    int ri = root_node_idx;
    auto swap_node = [&](int a, int b) {
      std::swap(tree.split_feature[a], tree.split_feature[b]);
      std::swap(tree.threshold_bin[a], tree.threshold_bin[b]);
      std::swap(tree.default_left[a], tree.default_left[b]);
      std::swap(tree.is_categorical[a], tree.is_categorical[b]);
      std::swap(tree.cat_bitset[a], tree.cat_bitset[b]);
      std::swap(node_lc[a], node_lc[b]);
      std::swap(node_rc[a], node_rc[b]);
    };
    swap_node(0, ri);
    for (int n = 0; n < (int)node_lc.size(); ++n) {
      auto fix = [&](int& c) {
        if (c == 0) c = ri;
        else if (c == ri) c = 0;
      };
      if (node_lc[n] >= 0) fix(node_lc[n]);
      if (node_rc[n] >= 0) fix(node_rc[n]);
    }
  }

  tree.left_child  = node_lc;
  tree.right_child = node_rc;
  return tree;
}

} // namespace shimaenaga
