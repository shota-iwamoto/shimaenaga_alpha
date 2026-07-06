#include "../../include/shimaenaga/dataset.h"
#include "bin_mapper.h"
#include "token_planner.h"
#include "../util/log.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstring>

namespace shimaenaga {

Dataset::~Dataset() = default;

int Dataset::NumBins(int f) const {
  return GetBinMapper(f).NumBins();
}

const bin_t* Dataset::FeatureBins(int f) const {
  return feature_bins_[f].data();
}

const TokenPlan& Dataset::GetTokenPlan() const {
  return *token_plan_;
}

const BinMapper& Dataset::GetBinMapper(int f) const {
  const auto& mp = owns_mappers_ ? mappers_ : *mappers_ptr_;
  return mp[f];
}

const std::vector<BinMapper>& Dataset::GetBinMappers() const {
  return owns_mappers_ ? mappers_ : *mappers_ptr_;
}

int Dataset::NumBinsForFeature(int f) const { return GetBinMapper(f).NumBins(); }
bool Dataset::IsCategorical(int f) const { return GetBinMapper(f).IsCategorical(); }
double Dataset::MissingRate(int f) const {
  if (num_data_ == 0) return 0.0;
  int miss = 0;
  for (data_size_t i = 0; i < num_data_; ++i)
    if (feature_bins_[f][i] == 0) miss++;
  return (double)miss / num_data_;
}

void Dataset::ValidateGroups(data_size_t n, const int32_t* groups, int ng) {
  if (ng == 0) return;
  int32_t total = 0;
  for (int i = 0; i < ng; ++i) {
    if (groups[i] <= 0) throw DataError("Group size must be positive");
    total += groups[i];
  }
  if (total != n) throw DataError("Sum of group sizes must equal num_data");
}

void Dataset::BuildBins(const float* X, const Config& cfg, bool fit_mappers) {
  const auto& mp = owns_mappers_ ? mappers_ : *mappers_ptr_;
  feature_bins_.resize(num_features_);
  for (int f = 0; f < num_features_; ++f) {
    feature_bins_[f].resize(num_data_);
    // Extract column f
    std::vector<float> col(num_data_);
    for (data_size_t i = 0; i < num_data_; ++i) col[i] = X[(size_t)i * num_features_ + f];
    if (fit_mappers) {
      // Mappers already fitted externally before this call
    }
    mp[f].MapArray(col.data(), feature_bins_[f].data(), num_data_);
  }
}

std::unique_ptr<Dataset> Dataset::Build(
    const float* X, data_size_t n, int num_features,
    const float* y, const float* weights,
    const int32_t* group_boundaries, int num_groups,
    const Config& cfg) {

  if (n <= 0) throw DataError("num_data must be > 0");
  if (num_features <= 0) throw DataError("num_features must be > 0");

  auto ds = std::make_unique<Dataset>();
  ds->num_data_     = n;
  ds->num_features_ = num_features;
  ds->owns_mappers_ = true;

  // Labels
  ds->labels_.assign(y, y + n);

  // Weights
  if (weights) ds->weights_.assign(weights, weights + n);

  // Groups (ranking)
  if (group_boundaries && num_groups > 0) {
    ds->ValidateGroups(n, group_boundaries, num_groups);
    // Convert sizes to boundaries (cumulative)
    ds->group_boundaries_.resize(num_groups + 1);
    ds->group_boundaries_[0] = 0;
    for (int g = 0; g < num_groups; ++g)
      ds->group_boundaries_[g+1] = ds->group_boundaries_[g] + group_boundaries[g];
  }

  // Fit bin mappers
  ds->mappers_.resize(num_features);
  for (int f = 0; f < num_features; ++f) {
    std::vector<float> col(n);
    for (data_size_t i = 0; i < n; ++i) col[i] = X[(size_t)i * num_features + f];
    // For now, treat all as numerical (categorical detection can be added)
    ds->mappers_[f].Fit(col.data(), n, cfg.max_bin, FeatureType::Numerical);
  }
  ds->mappers_ptr_ = &ds->mappers_;

  // Build bin matrix
  ds->BuildBins(X, cfg, false);

  // Compute missing rates
  std::vector<double> miss_rates(num_features, 0.0);
  std::vector<bool>   is_cat(num_features, false);
  for (int f = 0; f < num_features; ++f) {
    miss_rates[f] = ds->MissingRate(f);
    is_cat[f]     = ds->IsCategorical(f);
  }

  // Build token plan
  ds->token_plan_ = std::make_unique<TokenPlan>(
      TokenPlanner::Build(ds->feature_bins_, n, num_features, cfg, is_cat, miss_rates));

  SHIMAENAGA_LOG_INFO("Dataset built: %d samples, %d features, %d tokens",
              (int)n, num_features, ds->token_plan_->P);
  return ds;
}

std::unique_ptr<Dataset> Dataset::BuildLike(
    const Dataset& train,
    const float* X, data_size_t n,
    const float* y, const float* weights,
    const int32_t* group_boundaries, int num_groups) {

  auto ds = std::make_unique<Dataset>();
  ds->num_data_     = n;
  ds->num_features_ = train.num_features_;
  ds->owns_mappers_ = false;
  ds->mappers_ptr_  = &train.mappers_;

  ds->labels_.assign(y, y + n);
  if (weights) ds->weights_.assign(weights, weights + n);

  if (group_boundaries && num_groups > 0) {
    // Check train/valid group split (T5)
    ds->ValidateGroups(n, group_boundaries, num_groups);
    ds->group_boundaries_.resize(num_groups + 1);
    ds->group_boundaries_[0] = 0;
    for (int g = 0; g < num_groups; ++g)
      ds->group_boundaries_[g+1] = ds->group_boundaries_[g] + group_boundaries[g];
  }

  // Use train's mappers (no refitting - leakage prevention I1-I3)
  Config dummy_cfg;
  ds->BuildBins(X, dummy_cfg, false);

  // Copy token plan from train
  ds->token_plan_ = std::make_unique<TokenPlan>(*train.token_plan_);

  return ds;
}

} // namespace shimaenaga
