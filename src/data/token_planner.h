#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include "../../include/shimaenaga/config.h"

namespace shimaenaga {

enum class TokenType {
  FeatureGroup,   // numeric/category cluster
  Interaction,    // random subset
  Missingness,    // NaN indicator features
  PrevScore,
  RankingQuery,
  RankingDocument,
};

struct TokenPlan {
  int P = 0;
  std::vector<std::vector<int>> feature_subsets;  // S_p per token
  std::vector<TokenType> types;
  // attn mask: mask_bits[p] is a bitmask of tokens p can attend to
  // (bit r set = token p can attend to token r)
  std::vector<uint32_t> mask_bits;  // full or feature_local
};

// Builds token plan from feature correlation structure (数学設計書 §3.3)
class TokenPlanner {
 public:
  static TokenPlan Build(
      const std::vector<std::vector<bin_t>>& feature_bins,  // [F][N]
      data_size_t n_samples,
      int F,
      const Config& cfg,
      const std::vector<bool>& is_categorical,
      const std::vector<double>& missing_rates);
};

} // namespace shimaenaga
