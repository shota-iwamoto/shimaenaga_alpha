#include "../../include/shimaenaga/tree.h"
#include "../data/bin_mapper.h"

namespace shimaenaga {

leaf_t Tree::GetLeafRaw(const float* row, const std::vector<BinMapper>& mappers) const {
  if (num_leaves == 1) return 0;
  int node = 0;
  while (node >= 0) {
    int f = split_feature[node];
    bin_t b = mappers[f].MapValue(row[f]);
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

} // namespace shimaenaga
