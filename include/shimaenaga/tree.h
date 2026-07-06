#pragma once
#include <vector>
#include <cstdint>
#include "config.h"

namespace shimaenaga {

class BinMapper;  // forward declare for GetLeafRaw

// Decision tree - immutable after growth (詳細設計書 §5.1)
struct Tree {
  int num_leaves = 1;
  // Internal nodes (size = num_leaves-1)
  std::vector<int>     split_feature;
  std::vector<bin_t>   threshold_bin;
  std::vector<uint8_t> default_left;    // 1=left when NaN
  std::vector<int>     left_child;      // <0 means ~leaf_index
  std::vector<int>     right_child;
  // Categorical splits
  std::vector<uint8_t> is_categorical;
  std::vector<uint64_t> cat_bitset;     // one-vs-rest bitset

  // Traverse with column-oriented bin matrix
  leaf_t GetLeaf(const std::vector<std::vector<bin_t>>& cols, data_size_t i) const {
    if (num_leaves == 1) return 0;
    int node = 0;
    while (node >= 0) {
      int f = split_feature[node];
      bin_t b = cols[f][i];
      bool go_left;
      if (b == 0) {
        go_left = (default_left[node] == 1);
      } else if (is_categorical[node]) {
        uint64_t bit = (b < 64) ? (1ull << b) : 0ull;
        go_left = (cat_bitset[node] & bit) != 0;
      } else {
        go_left = (b <= threshold_bin[node]);
      }
      node = go_left ? left_child[node] : right_child[node];
    }
    return static_cast<leaf_t>(~node);
  }

  // Traverse a single pre-binned row (bin per feature). Inference fast path:
  // the row is binned once, instead of re-running BinMapper's binary search at
  // every visited node of every tree in every block.
  leaf_t GetLeafBinned(const bin_t* row_bins) const {
    if (num_leaves == 1) return 0;
    int node = 0;
    while (node >= 0) {
      bin_t b = row_bins[split_feature[node]];
      bool go_left;
      if (b == 0) {
        go_left = (default_left[node] == 1);
      } else if (is_categorical[node]) {
        uint64_t bit = (b < 64) ? (1ull << b) : 0ull;
        go_left = (cat_bitset[node] & bit) != 0;
      } else {
        go_left = (b <= threshold_bin[node]);
      }
      node = go_left ? left_child[node] : right_child[node];
    }
    return static_cast<leaf_t>(~node);
  }

  // Traverse with raw float values (inference)
  leaf_t GetLeafRaw(const float* row, const std::vector<BinMapper>& mappers) const;
};

} // namespace shimaenaga
