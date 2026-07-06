#pragma once
#include <memory>
#include <vector>
#include <utility>
#include "../../include/shimaenaga/config.h"
#include "../../include/shimaenaga/dataset.h"
#include "../../include/shimaenaga/model.h"
#include "../data/token_planner.h"
#include "../objective/objective.h"
#include "../tree/tree_learner.h"
#include "../attention/attention_engine.h"
#include "../attention/layer_stack.h"
#include "refitter.h"
#include "line_search.h"

namespace shimaenaga {

class BlockTrainer {
 public:
  BlockTrainer(const Config& cfg, const Dataset& train, Objective& obj);

  // Train one boosting iteration
  // F: current scores [N][C] (updated in-place)
  // beta_prev: previous iteration's beta [N][P] (for Phase A weighting)
  // Returns the block to append to the model
  AttentiveBlock TrainOneIter(int iter,
                               std::vector<score_t>& F,
                               const std::vector<float>& beta_prev);

  // Return β[i,p] learned in the last TrainOneIter call (for next Phase-A weighting)
  const std::vector<float>& GetBeta() const { return work_.beta; }

 private:
  const Config& cfg_;
  const Dataset& train_;
  Objective& obj_;
  data_size_t N_;
  int P_, H_, C_, d_a_;

  std::vector<score_t> g_, h_;

  // Effective warm-up length: min(attn_warmup, num_iterations/5) so short runs
  // don't spend most of their budget with attention frozen at uniform.
  int warmup_iters_;

  // Column-oriented bin matrix, shared with the Dataset (no copy).
  const std::vector<std::vector<bin_t>>& bins_;

  // Stochastic row bagging mask (bagging_fraction), resampled every bagging_freq
  // iterations and reused in between. Empty = use all rows.
  std::vector<bool> bagging_mask_;

  // Per-iteration working state
  BlockWork work_;
  BlockParams params_;
  ReadoutLUT rlut_;
  SelfAttnLUT slut_;

  // Tier-3 (lazy: constructed at the first post-ctx-warmup iteration)
  std::unique_ptr<Tier3Engine> t3_;
  Tier3Params t3params_;
  Tier3Work t3work_;
  int ctx_warmup_iters_ = 0;   // tier-3 activates for iter > ctx_warmup_iters_
  bool t3_active_ = false;     // set per-iteration by TrainOneIter

  // Tree learner persisted across iterations: bins are immutable, so its
  // per-feature bin-count precompute runs once instead of every iteration.
  std::unique_ptr<TreeLearner> tl_;

  // Reusable Phase-A per-token gradient buffers (avoid P× N-vector allocs/iter).
  std::vector<double> g_tilde_, h_tilde_;
  // Reusable multiclass weighted-gradient buffers (N×C, avoid per-token allocs).
  std::vector<double> g_nc_, h_nc_;

  // T6: warn once per training when self-attention heads have collapsed onto
  // each other (cos(Ā_h, Ā_h') > 0.99).
  bool head_sim_warned_ = false;

  // Fills g_tilde_/h_tilde_ (size N_) for token p; returns nothing (buffers reused).
  void MakeTokenGH(int p, const std::vector<float>& beta_prev);
  // Gate pseudo-target (設計書 §7): token-disagreement signal computed from the
  // Newton-initialized leaf values of the freshly grown token trees. Requires
  // token leaf indices already assigned in work_.
  std::vector<double> MakeGateGH(const std::vector<Tree>& token_trees) const;

  AttentiveBlock PackBlock(const std::vector<Tree>& token_trees,
                           const Tree& gate_tree,
                           const BlockParams& params) const;
  // Append the Tier-3 fields (e, layers, WR/WK, V_cls, ...) to a packed block.
  void PackTier3(AttentiveBlock& blk) const;
};

} // namespace shimaenaga
