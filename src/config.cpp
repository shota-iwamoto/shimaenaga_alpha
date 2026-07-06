#include "../include/shimaenaga/config.h"
#include "util/log.h"
#include <stdexcept>
#include <sstream>
#include <algorithm>

namespace shimaenaga {

Config Config::FromMap(const std::map<std::string, std::string>& m) {
  Config cfg;
  auto get_str = [&](const std::string& k, std::string& out) {
    auto it = m.find(k);
    if (it != m.end()) out = it->second;
  };
  auto get_int = [&](const std::string& k, int& out) {
    auto it = m.find(k);
    if (it != m.end()) out = std::stoi(it->second);
  };
  auto get_dbl = [&](const std::string& k, double& out) {
    auto it = m.find(k);
    if (it != m.end()) out = std::stod(it->second);
  };
  auto get_bool = [&](const std::string& k, bool& out) {
    auto it = m.find(k);
    if (it != m.end()) out = (it->second == "true" || it->second == "1");
  };
  auto get_u64 = [&](const std::string& k, uint64_t& out) {
    auto it = m.find(k);
    if (it != m.end()) out = std::stoull(it->second);
  };

  get_str("objective",           cfg.objective);
  get_int("num_class",           cfg.num_class);
  get_int("num_iterations",      cfg.num_iterations);
  get_dbl("learning_rate",       cfg.learning_rate);
  get_int("tier",                cfg.tier);
  get_int("num_tokens",          cfg.num_tokens);
  get_int("num_heads",           cfg.num_heads);
  get_str("attention_mode",      cfg.attention_mode);
  get_int("d_attn",              cfg.d_attn);
  get_dbl("eta_attn",            cfg.eta_attn);
  get_str("attn_mask",           cfg.attn_mask);
  get_int("token_num_leaves",    cfg.token_num_leaves);
  get_int("gate_num_leaves",     cfg.gate_num_leaves);
  get_dbl("token_feature_fraction", cfg.token_feature_fraction);
  get_str("token_plan",          cfg.token_plan);
  get_str("missing_token",       cfg.missing_token);
  get_bool("prev_score_token",   cfg.prev_score_token);
  get_int("inner_refit_steps",   cfg.inner_refit_steps);
  get_int("min_data_in_leaf",    cfg.min_data_in_leaf);
  get_int("max_depth",           cfg.max_depth);
  get_dbl("min_sum_hessian_in_leaf", cfg.min_sum_hessian_in_leaf);
  get_dbl("huber_alpha",         cfg.huber_alpha);
  get_dbl("quantile_alpha",      cfg.quantile_alpha);
  get_dbl("lambda_l1",           cfg.lambda_l1);
  get_dbl("lambda_v",            cfg.lambda_v);
  get_dbl("lambda_q",            cfg.lambda_q);
  get_dbl("lambda_k",            cfg.lambda_k);
  get_dbl("lambda_z",            cfg.lambda_z);
  get_dbl("lambda_ent",          cfg.lambda_ent);
  get_dbl("lambda_div",          cfg.lambda_div);
  get_dbl("min_gain_to_split",   cfg.min_gain_to_split);
  get_dbl("h_min",               cfg.h_min);
  get_dbl("eps_damping",         cfg.eps_damping);
  get_int("max_bin",             cfg.max_bin);
  get_dbl("bagging_fraction",    cfg.bagging_fraction);
  get_int("bagging_freq",        cfg.bagging_freq);
  get_dbl("feature_fraction",    cfg.feature_fraction);
  // sklearn / XGBoost-style aliases (applied after the canonical names above).
  get_dbl("subsample",           cfg.bagging_fraction);
  get_dbl("colsample_bytree",    cfg.feature_fraction);
  get_dbl("reg_alpha",           cfg.lambda_l1);
  get_dbl("reg_lambda",          cfg.lambda_v);
  get_dbl("min_child_weight",    cfg.min_sum_hessian_in_leaf);
  get_int("min_child_samples",   cfg.min_data_in_leaf);
  get_dbl("alpha",               cfg.quantile_alpha);
  get_int("early_stopping_rounds", cfg.early_stopping_rounds);
  get_dbl("sigma_rank",          cfg.sigma_rank);
  get_int("ndcg_truncation",     cfg.ndcg_truncation);
  get_int("max_pairs_per_group", cfg.max_pairs_per_group);
  get_int("num_threads",         cfg.num_threads);
  get_u64("seed",                cfg.seed);
  get_bool("force_deterministic", cfg.force_deterministic);
  get_int("attn_warmup",         cfg.attn_warmup);
  get_dbl("beta_ls",             cfg.beta_ls);
  get_int("max_backtrack",       cfg.max_backtrack);
  get_dbl("attn_step_clip",      cfg.attn_step_clip);
  get_dbl("beta_uniform_mix",    cfg.beta_uniform_mix);

  // Tier-3 (Tier-3 基本設計書 §7)
  get_int("attn_layers",         cfg.attn_layers);
  get_int("d_hidden",            cfg.d_hidden);
  get_int("d_ffn",               cfg.d_ffn);
  get_dbl("eta_u",               cfg.eta_u);
  get_dbl("eta_ffn",             cfg.eta_ffn);
  get_dbl("eta_cls",             cfg.eta_cls);
  get_bool("use_cls_token",      cfg.use_cls_token);
  get_bool("tau_learnable",      cfg.tau_learnable);
  get_dbl("lambda_e",            cfg.lambda_e);
  get_dbl("lambda_W",            cfg.lambda_W);
  get_dbl("lambda_c",            cfg.lambda_c);
  get_dbl("lambda_cls",          cfg.lambda_cls);
  get_dbl("lambda_tau",          cfg.lambda_tau);
  get_dbl("spectral_max",        cfg.spectral_max);
  get_dbl("drop_token",          cfg.drop_token);
  get_dbl("drop_layer",          cfg.drop_layer);
  get_int("ctx_warmup",          cfg.ctx_warmup);
  get_bool("phase_b_streaming",  cfg.phase_b_streaming);
  get_int("tier3_block_rows",    cfg.tier3_block_rows);
  get_dbl("norm_eps",            cfg.norm_eps);

  // Aliases
  if (m.count("n_estimators")) cfg.num_iterations = std::stoi(m.at("n_estimators"));
  if (m.count("lr"))           cfg.learning_rate  = std::stod(m.at("lr"));

  return cfg;
}

void Config::Validate() const {
  if (objective.empty())
    throw ConfigError("objective must be set");
  if (num_iterations < 1)
    throw ConfigError("num_iterations must be >= 1");
  if (learning_rate <= 0 || learning_rate > 1)
    throw ConfigError("learning_rate must be in (0, 1]");
  if (tier < 0 || tier > 3)
    throw ConfigError("tier must be 0, 1, 2, or 3");
  if (num_tokens < 1 || num_tokens > kMaxTokens)
    throw ConfigError("num_tokens must be in [1, " + std::to_string(kMaxTokens) + "]");
  if (num_heads < 1 || num_heads > kMaxHeads)
    throw ConfigError("num_heads must be in [1, " + std::to_string(kMaxHeads) + "]");
  if (d_attn < 1 || d_attn > kMaxDAttn)
    throw ConfigError("d_attn must be in [1, " + std::to_string(kMaxDAttn) + "]");
  if (token_num_leaves < 2 || token_num_leaves > kMaxLeaves)
    throw ConfigError("token_num_leaves must be in [2, 64]");
  if (gate_num_leaves < 2 || gate_num_leaves > kMaxLeaves)
    throw ConfigError("gate_num_leaves must be in [2, 64]");
  // bin_t is uint8_t: bins occupy 0..max_bin-1, so anything above 256 would
  // silently wrap and corrupt histograms/predictions.
  if (max_bin < 2 || max_bin > 256)
    throw ConfigError("max_bin must be in [2, 256]");
  if (max_depth == 0 || max_depth < -1)
    throw ConfigError("max_depth must be -1 (unlimited) or >= 1");
  if (lambda_l1 < 0.0)
    throw ConfigError("lambda_l1/reg_alpha must be >= 0");
  if (min_sum_hessian_in_leaf < 0.0)
    throw ConfigError("min_sum_hessian_in_leaf/min_child_weight must be >= 0");
  if (huber_alpha <= 0.0 || huber_alpha > 1.0)
    throw ConfigError("huber_alpha must be in (0, 1]");
  if (quantile_alpha <= 0.0 || quantile_alpha >= 1.0)
    throw ConfigError("quantile_alpha/alpha must be in (0, 1)");
  if (attn_step_clip < 0.0)
    throw ConfigError("attn_step_clip must be >= 0 (0 = off)");
  if (beta_uniform_mix < 0.0 || beta_uniform_mix > 1.0)
    throw ConfigError("beta_uniform_mix must be in [0, 1]");
  if (inner_refit_steps < 1 || inner_refit_steps > 3)
    throw ConfigError("inner_refit_steps must be in [1, 3]");
  if (num_class < 1)
    throw ConfigError("num_class must be >= 1");
  if (bagging_fraction <= 0.0 || bagging_fraction > 1.0)
    throw ConfigError("bagging_fraction/subsample must be in (0, 1]");
  if (feature_fraction <= 0.0 || feature_fraction > 1.0)
    throw ConfigError("feature_fraction/colsample_bytree must be in (0, 1]");
  if (eta_attn < 0.0 || eta_attn > 1.0)
    throw ConfigError("eta_attn must be in [0, 1]");
  if (lambda_v < 0 || lambda_q < 0 || lambda_k < 0 || lambda_z < 0 ||
      lambda_ent < 0 || lambda_div < 0)
    throw ConfigError("lambda_v/q/k/z/ent/div must be >= 0");
  if (attention_mode != "qk_leaf" && attention_mode != "score_tree")
    throw ConfigError("attention_mode must be 'qk_leaf' or 'score_tree'");
  if (attn_mask != "full" && attn_mask != "feature_local")
    throw ConfigError("attn_mask must be 'full' or 'feature_local'");
  // Manual token plans are not implemented yet; failing loudly beats silently
  // collapsing to a single all-features token (which is what used to happen).
  if (token_plan != "auto")
    throw ConfigError("token_plan='" + token_plan +
                      "' is not supported (only 'auto' is implemented)");
  if (token_feature_fraction <= 0.0 || token_feature_fraction > 1.0)
    throw ConfigError("token_feature_fraction must be in (0, 1]");

  // Tier-0 enforcement (LightGBM degenerate mode, T1)
  // Note: we can't modify const Config, validate only checks
  if (tier == 0) {
    SHIMAENAGA_LOG_INFO("tier=0: LightGBM-compatible degenerate mode");
  }

  // ─── Tier-3 (Tier-3 基本設計書 §7 の Validate 規則) ───
  if (tier == 3) {
    if (attention_mode != "qk_leaf")
      throw ConfigError("tier=3 requires attention_mode='qk_leaf' "
                        "(score_tree is tier<=2 only)");
    if (attn_layers < 1 || attn_layers > kMaxLayers)
      throw ConfigError("attn_layers must be in [1, " +
                        std::to_string(kMaxLayers) + "]");
    if (d_hidden < 1 || d_hidden > kMaxDHidden)
      throw ConfigError("d_hidden must be in [1, " +
                        std::to_string(kMaxDHidden) + "]");
    if (d_ffn < 0 || d_ffn > kMaxDFFN)
      throw ConfigError("d_ffn must be in [0, " + std::to_string(kMaxDFFN) + "]");
    if (eta_u < 0.0 || eta_u > 1.0 || eta_ffn < 0.0 || eta_ffn > 1.0 ||
        eta_cls < 0.0 || eta_cls > 1.0)
      throw ConfigError("eta_u/eta_ffn/eta_cls must be in [0, 1]");
    if (eta_cls > 0.5)
      SHIMAENAGA_LOG_WARN("eta_cls=%.2f > 0.5: the CLS head dominates the "
                          "convex-hull carrier; attention diagnostics (T6) "
                          "become hard to interpret", eta_cls);
    if (lambda_e < 0 || lambda_W < 0 || lambda_c < 0 || lambda_cls < 0 ||
        lambda_tau < 0)
      throw ConfigError("lambda_e/W/c/cls/tau must be >= 0");
    if (spectral_max < 0.0)
      throw ConfigError("spectral_max must be >= 0 (0 = off)");
    if (drop_token < 0.0 || drop_token > 0.5 ||
        drop_layer < 0.0 || drop_layer > 0.5)
      throw ConfigError("drop_token/drop_layer must be in [0, 0.5]");
    if (drop_token > 0.0 || drop_layer > 0.0)
      SHIMAENAGA_LOG_WARN("drop_token/drop_layer are reserved in v2.0-alpha "
                          "and currently ignored");
    if (norm_eps <= 0.0)
      throw ConfigError("norm_eps must be > 0");
    if (tier3_block_rows < 4096)
      throw ConfigError("tier3_block_rows must be >= 4096");
  }

  // Memory estimate (基本設計書 §11)
  // readout LUT: H * L_g * P * L_p * 4 bytes per block
  // self-attn LUT (tier-2): H * P^2 * L_p^2 * 4 bytes per block — this term
  // dominates and was previously missing from the estimate.
  size_t lut_per_block = (size_t)num_heads * gate_num_leaves *
                          num_tokens * token_num_leaves * 4;
  if (tier >= 2)
    lut_per_block += (size_t)num_heads * num_tokens * num_tokens *
                     token_num_leaves * token_num_leaves * 4;
  size_t lut_total_mb = lut_per_block * num_iterations / (1024 * 1024);
  if (lut_total_mb > 512) {
    SHIMAENAGA_LOG_WARN("Estimated LUT memory ~%zu MB > 512 MB. "
                "Consider feature_local mask or smaller L, P.", lut_total_mb);
  }
}

} // namespace shimaenaga
