#include "bin_mapper.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>

namespace shimaenaga {

void BinMapper::Fit(const float* values, data_size_t n, int max_bin, FeatureType ftype) {
  ftype_ = ftype;
  if (ftype == FeatureType::Categorical) {
    // Map integer category values to bins
    std::unordered_map<int, int> freq;
    for (data_size_t i = 0; i < n; ++i) {
      if (!std::isnan(values[i])) {
        int c = static_cast<int>(values[i]);
        freq[c]++;
      }
    }
    // Sort by frequency descending
    std::vector<std::pair<int,int>> pairs(freq.begin(), freq.end());
    std::sort(pairs.begin(), pairs.end(),
              [](const auto& a, const auto& b){ return a.second > b.second; });
    // bin 0 = NaN/unknown. Categories get bins 1..K
    int limit = std::min((int)pairs.size(), max_bin - 1);
    max_cat_id_ = 0;
    int next_bin = 1;
    for (auto& p : pairs) {
      if (next_bin > limit) break;
      int c = p.first;
      if (c < 0) continue;
      if (c >= static_cast<int>(cat_map_.size()))
        cat_map_.resize(c + 1, 0);
      cat_map_[c] = next_bin++;
      max_cat_id_ = std::max(max_cat_id_, c);
    }
    return;
  }

  // Numerical: compute quantile thresholds
  std::vector<float> sorted;
  sorted.reserve(n);
  for (data_size_t i = 0; i < n; ++i)
    if (!std::isnan(values[i])) sorted.push_back(values[i]);
  if (sorted.empty()) { thresholds_ = {0.0f}; return; }
  std::sort(sorted.begin(), sorted.end());

  // Deduplicate and pick at most max_bin-1 thresholds
  int nb = std::min(max_bin - 1, static_cast<int>(sorted.size()));
  thresholds_.clear();
  double step = static_cast<double>(sorted.size()) / nb;
  for (int i = 0; i < nb; ++i) {
    int idx = static_cast<int>((i + 1) * step) - 1;
    idx = std::max(0, std::min(idx, (int)sorted.size() - 1));
    float t = sorted[idx];
    if (thresholds_.empty() || t != thresholds_.back())
      thresholds_.push_back(t);
  }
  // Ensure max value is covered
  if (thresholds_.empty() || thresholds_.back() < sorted.back())
    thresholds_.push_back(sorted.back());
}

bin_t BinMapper::MapValue(float v) const {
  if (std::isnan(v)) return 0;  // bin 0 = missing
  if (ftype_ == FeatureType::Categorical) {
    int c = static_cast<int>(v);
    if (c < 0 || c >= static_cast<int>(cat_map_.size())) return 0;
    return static_cast<bin_t>(cat_map_[c]);
  }
  // Binary search for threshold
  int lo = 0, hi = static_cast<int>(thresholds_.size()) - 1;
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (v <= thresholds_[mid]) hi = mid;
    else lo = mid + 1;
  }
  return static_cast<bin_t>(lo + 1);  // bin 1..K
}

void BinMapper::MapArray(const float* in, bin_t* out, data_size_t n) const {
  for (data_size_t i = 0; i < n; ++i) out[i] = MapValue(in[i]);
}

} // namespace shimaenaga
