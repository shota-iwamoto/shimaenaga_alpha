#pragma once
#include <vector>
#include <memory>
#include <cstdint>
#include "config.h"
#include "export.h"

namespace shimaenaga {

struct TokenPlan;
class BinMapper;

class SHIMAENAGA_EXPORT Dataset {
 public:
  Dataset() = default;
  ~Dataset();

  // Build from training data - fits bin mappers and token plan
  static std::unique_ptr<Dataset> Build(
      const float* X, data_size_t n, int num_features,
      const float* y, const float* weights,
      const int32_t* group_boundaries, int num_groups,
      const Config& cfg);

  // Build validation/test data using train's mappers (I1-I3)
  static std::unique_ptr<Dataset> BuildLike(
      const Dataset& train,
      const float* X, data_size_t n,
      const float* y, const float* weights,
      const int32_t* group_boundaries, int num_groups);

  data_size_t NumData()     const { return num_data_; }
  int         NumFeatures() const { return num_features_; }
  int         NumBins(int f)const;

  // Column-oriented bin access: feature_bins_[f][i]
  const bin_t* FeatureBins(int f) const;
  // Whole bin matrix (shared by TreeLearner / valid scoring — no copies)
  const std::vector<std::vector<bin_t>>& AllBins() const { return feature_bins_; }
  const float* Labels()   const { return labels_.data(); }
  const float* Weights()  const { return weights_.data(); }
  bool HasWeights()       const { return !weights_.empty(); }
  const std::vector<int32_t>& GroupBoundaries() const { return group_boundaries_; }
  bool IsRanking()        const { return !group_boundaries_.empty(); }

  const TokenPlan& GetTokenPlan() const;
  const BinMapper& GetBinMapper(int f) const;
  const std::vector<BinMapper>& GetBinMappers() const;

  int  NumBinsForFeature(int f) const;
  bool IsCategorical(int f) const;
  double MissingRate(int f) const;

 private:
  data_size_t num_data_    = 0;
  int         num_features_ = 0;

  std::vector<std::vector<bin_t>> feature_bins_;  // [F][N]
  std::vector<float> labels_;
  std::vector<float> weights_;
  std::vector<int32_t> group_boundaries_;

  // Owned by train, shared (pointer) by valid/test
  std::vector<BinMapper>         mappers_;
  bool                           owns_mappers_ = true;
  const std::vector<BinMapper>*  mappers_ptr_  = nullptr;

  std::unique_ptr<TokenPlan>     token_plan_;

  void BuildBins(const float* X, const Config& cfg, bool fit_mappers);
  void ValidateGroups(data_size_t n, const int32_t* groups, int ng);
};

} // namespace shimaenaga
