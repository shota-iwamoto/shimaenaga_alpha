#include "block_trainer.h"
#include "../util/log.h"
#include "../util/random.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cassert>

namespace shimaenaga {

BlockTrainer::BlockTrainer(const Config& cfg, const Dataset& train, Objective& obj)
    : cfg_(cfg), train_(train), obj_(obj),
      N_(train.NumData()),
      P_(cfg.tier == 0 ? 1 : cfg.num_tokens),
      H_(cfg.tier == 0 ? 1 : cfg.num_heads),
      C_(cfg.num_class > 1 ? cfg.num_class : 1),
      d_a_(cfg.d_attn),
      g_(N_ * C_, 0.0), h_(N_ * C_, 0.0),
      warmup_iters_(std::min(cfg.attn_warmup,
                             std::max(1, cfg.num_iterations / 5))),
      bins_(train.AllBins()) {  // shared with the Dataset — no copy

  work_.Alloc(N_, P_, H_, C_, cfg_.tier);
  // Build the tree learner once so its per-feature bin-count precompute (O(N·F))
  // is paid a single time rather than every boosting iteration.
  tl_ = std::make_unique<TreeLearner>(cfg_, bins_, N_, train_.NumFeatures());

  // Tier-3 context warm-up (数学 §8.7): before it expires, tier=3 trains on
  // the plain tier-2 Phase-B path (identical semantics — anchored parity), and
  // the packed blocks carry T_L = 0 so inference short-circuits to tier-2.
  if (cfg_.tier == 3) {
    int ctx = cfg_.ctx_warmup >= 0 ? cfg_.ctx_warmup : cfg_.attn_warmup;
    ctx_warmup_iters_ = std::max(warmup_iters_, ctx);
  }
}

void BlockTrainer::MakeTokenGH(int p, const std::vector<float>& beta_prev) {
  g_tilde_.resize(N_);
  bool use_beta = !beta_prev.empty();
  const double gam = cfg_.beta_uniform_mix, unif = 1.0 / P_;
  for (data_size_t i = 0; i < N_; ++i) {
    // w = (1-γ)β̂ + γ/P: starvation guard — a token whose β̂ collapsed to 0
    // would otherwise see zero gradients forever and can never recover.
    double weight = use_beta
        ? (1.0 - gam) * beta_prev[(size_t)i * P_ + p] + gam * unif
        : unif;
    if (C_ == 1) {
      g_tilde_[i] = weight * g_[(size_t)i];
    } else {
      // Multiclass: Σ_k g_k = 0 (softmax), use diagonal Newton gain proxy
      double score = 0.0;
      for (int k = 0; k < C_; ++k) {
        double gk = g_[(size_t)i * C_ + k];
        double hk = h_[(size_t)i * C_ + k];
        score += gk * gk / (hk + cfg_.eps_damping);
      }
      g_tilde_[i] = weight * score;
    }
  }
}

std::vector<double> BlockTrainer::MakeGateGH(const std::vector<Tree>& token_trees) const {
  // 設計書 §7: gate 木は「token 予測分散を擬似ターゲット」に成長させる。
  // s_p = Σ_k g_ik · v_{p,ℓ_p(i),k}(v は Newton 初期化葉値, φ=0 なので r=g)
  // は β_p 方向の損失方向微分。その p 間標準偏差が大きいサンプルほど
  // ルーティング(gate 分割)の利得が大きい。旧実装の |g| マグニチュードは
  // 「誤差が大きい場所」しか見ず、トークンが同意見の領域まで分割していた。
  // Newton-init leaf values per token (same formula as AttentionEngine::InitParams).
  std::vector<std::vector<double>> v(P_);  // v[p][l*C+k]
  for (int p = 0; p < P_; ++p) {
    int L = token_trees[p].num_leaves;
    std::vector<double> sum_g((size_t)L * C_, 0.0), sum_h((size_t)L * C_, 0.0);
    for (data_size_t i = 0; i < N_; ++i) {
      leaf_t l = work_.leaf_idx[(size_t)p * N_ + i];
      if (l < 0 || l >= L) continue;
      for (int k = 0; k < C_; ++k) {
        sum_g[(size_t)l * C_ + k] += g_[(size_t)i * C_ + k];
        sum_h[(size_t)l * C_ + k] += h_[(size_t)i * C_ + k];
      }
    }
    v[p].assign((size_t)L * C_, 0.0);
    for (int l = 0; l < L; ++l)
      for (int k = 0; k < C_; ++k)
        v[p][(size_t)l * C_ + k] =
            -sum_g[(size_t)l * C_ + k] / (sum_h[(size_t)l * C_ + k] + cfg_.lambda_v);
  }

  std::vector<double> g_gate(N_);
  #pragma omp parallel for schedule(static)
  for (data_size_t i = 0; i < N_; ++i) {
    double s[kMaxTokens];
    double mean = 0.0;
    for (int p = 0; p < P_; ++p) {
      leaf_t l = work_.leaf_idx[(size_t)p * N_ + i];
      double sp = 0.0;
      for (int k = 0; k < C_; ++k)
        sp += g_[(size_t)i * C_ + k] * v[p][(size_t)l * C_ + k];
      s[p] = sp;
      mean += sp;
    }
    mean /= P_;
    double var = 0.0;
    for (int p = 0; p < P_; ++p) var += (s[p] - mean) * (s[p] - mean);
    g_gate[i] = std::sqrt(var / P_);
  }
  return g_gate;
}

AttentiveBlock BlockTrainer::PackBlock(
    const std::vector<Tree>& token_trees, const Tree& gate_tree,
    const BlockParams& params) const {

  AttentiveBlock blk;
  blk.token_trees = token_trees;
  blk.gate_tree   = gate_tree;
  blk.P = params.P; blk.H = params.H; blk.C = params.C; blk.d_a = params.d_a;
  blk.attention_mode = params.mode;
  blk.tier = cfg_.tier;
  blk.gate_num_leaves = params.gate_L;
  blk.v_lsize = params.token_L;

  // v, z_or_q, k are stored flat in BlockParams with exactly the layout the
  // serialized block expects ([p][l][k], [lg][h][d], [p][l][h][d]) — copy directly.
  blk.v = params.v;
  blk.z_or_q = params.z_or_q;
  blk.k = params.k;

  // biases
  blk.b.resize(params.H * params.P);
  for (int h = 0; h < params.H; ++h)
    for (int p = 0; p < params.P; ++p)
      blk.b[h * params.P + p] = params.b[h][p];

  // head weights
  for (int h = 0; h < params.H; ++h) {
    blk.rho[h]  = params.rho[h];
    blk.rhoA[h] = params.rhoA[h];
  }

  // Tier-2 self-attention (qA/kA flat layout [p][l][h][d] matches the block).
  if (cfg_.tier >= 2 && !params.qA.empty()) {
    blk.qA = params.qA;
    blk.kA = params.kA;
    blk.bA.resize(params.H * params.P * params.P);
    for (int h = 0; h < params.H; ++h)
      for (int p = 0; p < params.P; ++p)
        for (int r = 0; r < params.P; ++r)
          blk.bA[h * params.P * params.P + p * params.P + r] = params.bA[h][p][r];
    // Attention mask (feature_local): needed at inference to reproduce the
    // masked softmax. Omitted (empty) for full attention.
    const TokenPlan& plan = train_.GetTokenPlan();
    if (cfg_.attn_mask != "full" && (int)plan.mask_bits.size() == params.P)
      blk.attn_mask = plan.mask_bits;
  }

  return blk;
}

AttentiveBlock BlockTrainer::TrainOneIter(int iter,
                                           std::vector<score_t>& F,
                                           const std::vector<float>& beta_prev) {
  // ── Compute gradients (E.24) ──
  obj_.GetGradients(F.data(), N_,
                    train_.Labels(),
                    train_.HasWeights() ? train_.Weights() : nullptr,
                    g_.data(), h_.data());

  // ── Phase A: grow trees (bins shared with the Dataset) ──
  const std::vector<std::vector<bin_t>>& bins = bins_;
  const TokenPlan& plan = train_.GetTokenPlan();
  bool use_beta = (iter > warmup_iters_) && !beta_prev.empty();

  // Tree learner (persisted across iterations; see ctor).
  TreeLearner& tl = *tl_;

  // All features list for gate tree
  std::vector<int> all_feats(train_.NumFeatures());
  std::iota(all_feats.begin(), all_feats.end(), 0);

  // ── Stochastic subsampling (regularization, 基本設計 §正則化) ──
  // RNG keyed by (seed, iter) so results stay bit-deterministic (T10).
  Random sub_rng(cfg_.seed * 0x9e3779b97f4a7c15ull + (uint64_t)iter * 0x85ebca6b + 1);
  // Row bagging: resample the mask every bagging_freq iterations, reuse between.
  bool do_bag = cfg_.bagging_fraction > 0.0 && cfg_.bagging_fraction < 1.0 - 1e-9;
  if (do_bag) {
    int bfreq = cfg_.bagging_freq > 0 ? cfg_.bagging_freq : 1;
    if (bagging_mask_.empty() || ((iter - 1) % bfreq == 0)) {
      bagging_mask_.assign(N_, false);
      data_size_t kept = 0;
      for (data_size_t i = 0; i < N_; ++i)
        if (sub_rng.NextDouble() < cfg_.bagging_fraction) { bagging_mask_[i] = true; ++kept; }
      if (kept == 0) std::fill(bagging_mask_.begin(), bagging_mask_.end(), true);  // guard
    }
  } else {
    bagging_mask_.clear();
  }
  const std::vector<bool>& row_mask = bagging_mask_;  // empty => all rows
  // Feature colsample (feature_fraction): random subset of a tree's candidate features.
  auto colsample = [&](const std::vector<int>& base) -> std::vector<int> {
    if (cfg_.feature_fraction >= 1.0 - 1e-9 || (int)base.size() <= 1) return base;
    int k = std::max(1, (int)std::lround((double)base.size() * cfg_.feature_fraction));
    std::vector<int> idx = base;
    sub_rng.Shuffle(idx);
    idx.resize(k);
    std::sort(idx.begin(), idx.end());
    return idx;
  };

  std::vector<Tree> token_trees(P_);
  for (int p = 0; p < P_; ++p) {
    const std::vector<int>* subset = &all_feats;
    if (p < (int)plan.feature_subsets.size())
      subset = &plan.feature_subsets[p];
    std::vector<int> feats = colsample(*subset);

    if (C_ == 1) {
      // Scalar: weight g̃/h̃ per token (reused member buffers, no per-token alloc)
      // 設計書 §7 Phase A: warmup は g̃=g/P, h̃=h/P; m≥2 は g̃=β̂g, h̃=β̂²h+h_min.
      // φ は β·v に線形なので v に関する二階項は β² が正しい (E.31 の分母と整合)。
      MakeTokenGH(p, use_beta ? beta_prev : std::vector<float>{});
      h_tilde_.resize(N_);
      const double gam = cfg_.beta_uniform_mix, unif = 1.0 / P_;
      for (data_size_t i = 0; i < N_; ++i) {
        if (use_beta) {
          double w = (1.0 - gam) * beta_prev[(size_t)i * P_ + p] + gam * unif;
          h_tilde_[i] = w * w * h_[(size_t)i] + cfg_.h_min;
        } else {
          h_tilde_[i] = (h_[(size_t)i] + cfg_.h_min) / P_;
        }
      }
      token_trees[p] = tl.Grow(g_tilde_, h_tilde_, feats, cfg_.token_num_leaves, row_mask);
    } else {
      // Multiclass: pass full NxC gradients weighted by token attention beta
      // (reused member buffers — allocating N·C doubles per token was a hotspot)
      g_nc_.resize((size_t)N_ * C_);
      h_nc_.resize((size_t)N_ * C_);
      const double gam = cfg_.beta_uniform_mix, unif = 1.0 / P_;
      for (data_size_t i = 0; i < N_; ++i) {
        double weight = use_beta
            ? (1.0 - gam) * beta_prev[(size_t)i * P_ + p] + gam * unif
            : unif;
        double hw = use_beta ? weight * weight : weight;  // β̂² (§7) / warmup 1/P
        for (int k = 0; k < C_; ++k) {
          g_nc_[(size_t)i * C_ + k] = weight * g_[(size_t)i * C_ + k];
          h_nc_[(size_t)i * C_ + k] = hw * h_[(size_t)i * C_ + k] + cfg_.h_min;
        }
      }
      token_trees[p] = tl.GrowMultiOutput(g_nc_, h_nc_, C_, feats, cfg_.token_num_leaves, row_mask);
    }
  }

  // Assign token leaf indices first: the gate pseudo-target needs them.
  for (int p = 0; p < P_; ++p)
    for (data_size_t i = 0; i < N_; ++i)
      work_.LeafIdx(p, i) = token_trees[p].GetLeaf(bins, i);

  // Gate tree, grown on the token-disagreement signal (設計書 §7).
  // P=1 (tier-0) has zero disagreement → gate degenerates to a stump, which is
  // correct: β ≡ 1 there and the gate is never consulted.
  Tree gate_tree;
  std::vector<int> gate_feats = colsample(all_feats);
  {
    std::vector<double> g_gate = MakeGateGH(token_trees);
    std::vector<double> h_gate(N_, 1.0);
    gate_tree = tl.Grow(g_gate, h_gate, gate_feats, cfg_.gate_num_leaves, row_mask);
  }
  for (data_size_t i = 0; i < N_; ++i)
    work_.LeafIdx(P_, i) = gate_tree.GetLeaf(bins, i);

  // ── Init block params ──
  std::vector<int> token_leaves(P_);
  for (int p = 0; p < P_; ++p)
    token_leaves[p] = token_trees[p].num_leaves;

  params_.Init(P_, H_, C_, d_a_, cfg_.attention_mode,
               gate_tree.num_leaves, token_leaves, cfg_.tier);

  AttentionEngine engine(cfg_, N_, plan);
  engine.InitParams(params_, g_, h_, work_);

  bool warmup = (iter <= warmup_iters_) || (cfg_.tier == 0);
  // Tier-3 activation (基本設計 §8): after the context warm-up the deep path
  // takes over Phase B entirely; before it, tier=3 runs the tier-2 path
  // (anchored parity — 数学 §8.1/§8.7).
  t3_active_ = (cfg_.tier == 3) && !warmup && (iter > ctx_warmup_iters_);

  if (t3_active_) {
    // ── Phase B (Tier-3): deep forward/backward + ladder (E3.28) ──
    if (!t3_) {
      t3_ = std::make_unique<Tier3Engine>(cfg_, N_, plan);
      t3work_.Alloc(N_, P_, H_, C_, cfg_.d_hidden, cfg_.d_ffn, cfg_.attn_layers);
    }
    t3_->InitParams(t3params_, params_);   // anchored init (E3.21)
    t3_->PhaseB(params_, t3params_, g_, h_, work_, t3work_);
  } else {
  // ── Phase B (tier <= 2 path; also tier-3 warm-up iterations) ──
  engine.BuildReadoutLUT(params_, rlut_);
  if (cfg_.tier >= 2) engine.BuildSelfAttnLUT(params_, slut_);

  ForwardOpts first_opts;      // everything on, plus the once-per-iteration
  first_opts.abar = true;      // Ā aggregation for λ_div / T6 (数学§7.5)
  engine.Forward(params_, rlut_, slut_, g_, h_, work_, warmup, first_opts);

  Refitter refitter(cfg_, g_, h_, work_, params_, engine, rlut_, slut_);
  refitter.BuildCSR();
  LineSearch ls(cfg_, refitter, params_, work_, engine, rlut_, slut_, g_, h_);

  for (int s = 0; s < cfg_.inner_refit_steps; ++s) {
    double Q_before = refitter.EvaluateQ();

    // (B1) is exact cyclic coordinate descent — it decreases Q monotonically on
    // its own, so the line-search checkpoint is taken AFTER it (§7.6: rollback
    // target is the "(B1) のみ" state, never discarding B1's guaranteed progress).
    refitter.RefitValues();
    ls.SaveState();

    if (!warmup) {
      refitter.RefitReadout();
      if (cfg_.tier >= 2) refitter.RefitSelfAttn();
      refitter.RefitHeads();
    }
    ls.Check(Q_before);
  }
  }

  // ── Update F ──
  for (data_size_t i = 0; i < N_; ++i)
    for (int k = 0; k < C_; ++k)
      F[(size_t)i * C_ + k] += cfg_.learning_rate * work_.Phi(i, k);

  // T6: head diversity check — near-identical mean attention maps mean the
  // extra heads are wasted capacity. Ā is maintained by Forward when
  // lambda_div > 0 (the same statistic drives the (E.41) regularizer).
  // Cosine is measured on Ā − 1/P (deviation from uniform): right after
  // warm-up every head is still ≈uniform, which has raw cosine ≈1 trivially
  // and would fire a meaningless warning.
  if (!head_sim_warned_ && work_.abar_valid && H_ > 1) {
    const int PP = P_ * P_;
    const double unif = 1.0 / P_;
    for (int h1 = 0; h1 < H_ && !head_sim_warned_; ++h1)
      for (int h2 = h1 + 1; h2 < H_ && !head_sim_warned_; ++h2) {
        double d = 0, n1 = 0, n2 = 0;
        for (int t = 0; t < PP; ++t) {
          double a = work_.a_bar[(size_t)h1 * PP + t] - unif;
          double b = work_.a_bar[(size_t)h2 * PP + t] - unif;
          d += a * b; n1 += a * a; n2 += b * b;
        }
        // Require the heads to have actually moved away from uniform before
        // judging their similarity.
        if (n1 > 1e-4 && n2 > 1e-4 && d / std::sqrt(n1 * n2) > 0.99) {
          SHIMAENAGA_LOG_WARN("attention heads %d/%d are near-identical "
                              "(cos>0.99) at iteration %d — consider fewer "
                              "heads or larger lambda_div (T6)", h1, h2, iter);
          head_sim_warned_ = true;
        }
      }
  }

  AttentiveBlock blk = PackBlock(token_trees, gate_tree, params_);
  if (t3_active_) PackTier3(blk);
  return blk;
}

void BlockTrainer::PackTier3(AttentiveBlock& blk) const {
  const Tier3Params& t3 = t3params_;
  blk.T_L = t3.T_L;
  blk.d_u = t3.d_u;
  blk.d_f = t3.d_f;
  blk.e = t3.e;
  blk.e_gate = t3.e_gate;
  blk.WR = t3.WR;
  blk.WK = t3.WK;
  blk.V_cls = t3.V_cls;
  for (int h = 0; h < t3.H; ++h) blk.theta_R[h] = t3.theta_R[h];
  blk.layers.resize(t3.T_L);
  for (int t = 0; t < t3.T_L; ++t) {
    const auto& L = t3.layers[t];
    auto& M = blk.layers[t];
    M.Wq = L.Wq; M.Wk = L.Wk; M.Wv = L.Wv;
    M.a_q = L.a_q; M.a_k = L.a_k;
    M.bA3 = L.bA3;
    M.W1 = L.W1; M.W2 = L.W2; M.c1 = L.c1;
    for (int h = 0; h < t3.H; ++h) {
      M.rho3[h] = L.rho3[h];
      M.theta[h] = L.theta[h];
    }
    M.gamma_c = L.gamma_c;
  }
  // Tier-3 blocks always need the mask rows at inference (row-mask + CLS
  // reconstruction); pack them even for attn_mask="full" (empty = full).
  if (cfg_.attn_mask != "full") {
    const TokenPlan& plan = train_.GetTokenPlan();
    if ((int)plan.mask_bits.size() == t3.P) blk.attn_mask = plan.mask_bits;
  }
}

} // namespace shimaenaga
