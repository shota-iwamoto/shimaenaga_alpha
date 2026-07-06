#pragma once
#include <cstdint>
#include <string>
#include <map>
#include <stdexcept>
#include "export.h"

namespace shimaenaga {

// Section 2: Basic types
using data_size_t = int32_t;
using bin_t       = uint8_t;
using leaf_t      = int16_t;
using score_t     = double;
using param_t     = float;

constexpr int    kMaxTokens = 16;
constexpr int    kMaxHeads  = 4;
constexpr int    kMaxDAttn  = 8;
constexpr int    kMaxLeaves = 64;
constexpr double kEps       = 1e-15;
// Tier-3 (Tier-3 詳細設計書 §2)
constexpr int    kMaxLayers  = 4;   // T_L
constexpr int    kMaxDHidden = 16;  // d_u
constexpr int    kMaxDFFN    = 32;  // d_f

// Section 3: Config (基本設計書 §7)
struct SHIMAENAGA_EXPORT Config {
  // Task
  std::string objective;        // regression/binary/multiclass/lambdarank
  int  num_class       = 1;
  // Boosting
  int    num_iterations = 1000;
  double learning_rate  = 0.05;
  // Attention architecture
  int    tier          = 2;     // 0=pure GBDT, 1=Tier-1, 2=Tier-2, 3=Tier-3
  int    num_tokens    = 8;     // P
  int    num_heads     = 2;     // H
  std::string attention_mode = "qk_leaf"; // score_tree / qk_leaf
  int    d_attn        = 4;     // d_a
  double eta_attn      = 0.5;   // η_attn (Tier-2)
  std::string attn_mask = "full"; // full / feature_local
  // Tree structure
  int    token_num_leaves = 31;
  int    gate_num_leaves  = 31;
  double token_feature_fraction = 0.7;
  std::string token_plan  = "auto";
  std::string missing_token = "auto";
  bool   prev_score_token = false;
  // Training
  int    inner_refit_steps = 2;
  int    min_data_in_leaf  = 20;
  int    max_depth         = -1;    // -1 = unlimited (leaf-wise depth cap)
  double min_sum_hessian_in_leaf = 1e-3;
  // Objective options
  double huber_alpha    = 0.9;      // adaptive Huber: delta = alpha-quantile of |residual|
  double quantile_alpha = 0.5;      // pinball quantile (0.5 = MAE)
  // Regularization
  double lambda_l1  = 0.0;
  double lambda_v   = 1.0;
  double lambda_q   = 0.1;
  double lambda_k   = 0.1;
  double lambda_z   = 0.1;
  double lambda_ent = 1e-3;
  double lambda_div = 1e-3;
  double min_gain_to_split = 0.0;
  double h_min        = 1e-6;
  double eps_damping  = 1e-6;
  // Data
  int    max_bin = 255;
  double bagging_fraction = 1.0;
  int    bagging_freq     = 0;
  double feature_fraction = 1.0;
  // Early stopping
  int    early_stopping_rounds = 0;
  // Ranking
  double sigma_rank       = 1.0;
  int    ndcg_truncation  = 10;
  int    max_pairs_per_group = 0;
  // System
  int      num_threads        = 0;   // 0 = auto
  uint64_t seed               = 0;
  bool     force_deterministic = true;
  // Internal
  int    attn_warmup   = 10;
  double beta_ls       = 0.5;
  int    max_backtrack = 5;
  // Attention stabilization (collapse guards)
  // Max |Δω| / |Δb| / |ΔbA| per scalar-Newton update (0 = off). Tiny hessians
  // otherwise produce divergent steps that saturate the softmax to exact 0/1,
  // after which gradients vanish and heads/tokens die irrecoverably.
  double attn_step_clip = 1.0;
  // Phase A token weight w = (1-γ)β̂ + γ/P: keeps starved tokens (β̂→0) able
  // to grow trees and recover instead of degenerating to 1-leaf stumps.
  double beta_uniform_mix = 0.1;

  // ─── Tier-3 (Tier-3 基本設計書 §7, tier=3 のみ有効) ───
  int    attn_layers   = 2;      // T_L: deep attention layers, 1..kMaxLayers
  int    d_hidden      = 8;      // d_u: hidden-stream dim, 1..kMaxDHidden
  int    d_ffn         = 16;     // d_f: Tree-FFN inner dim, 0..kMaxDFFN (0=off)
  double eta_u         = 0.5;    // hidden residual mix (E3.14)
  double eta_ffn       = 0.5;    // FFN residual mix (E3.16)
  double eta_cls       = 0.1;    // CLS head mix (E3.18); 0 keeps the convex hull
  bool   use_cls_token = true;   // CLS participates in layer self-attention
  bool   tau_learnable = true;   // learnable temperature τ=exp(θ), |θ|<=ln4
  double lambda_e      = 0.1;    // L2 on leaf hidden embeddings e (E3.29)
  double lambda_W      = 0.1;    // L2 on all context/FFN projection matrices
  double lambda_c      = 0.1;    // L2 on Tree-FFN leaf bias c
  double lambda_cls    = 1.0;    // L2 on CLS head V_cls
  double lambda_tau    = 1e-2;   // L2 on log-temperature θ
  double spectral_max  = 2.0;    // spectral projection cap for Wv/W2 (0=off)
  double drop_token    = 0.0;    // reserved (v2.0 では未実装、範囲検査のみ)
  double drop_layer    = 0.0;    // reserved (v2.0 では未実装、範囲検査のみ)
  int    ctx_warmup    = -1;     // -1 = attn_warmup と同値 (数学 §8.7)
  bool   phase_b_streaming = true;   // reserved (v2.0 は全行ワーク; 詳細設計 §5 参照)
  int    tier3_block_rows  = 65536;  // reserved (streaming block rows)
  double norm_eps      = 1e-6;   // RMSNorm ε_n (E3.2)

  static Config FromMap(const std::map<std::string, std::string>& m);
  void Validate() const;
};

} // namespace shimaenaga
