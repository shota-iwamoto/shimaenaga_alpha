#pragma once
#include <vector>
#include <cstdint>
#include "config.h"
#include "export.h"

namespace shimaenaga {

enum class FeatureType { Numerical, Categorical };

// Maps raw float feature values to bin indices (uint8)
// bin 0 = missing/NaN
class SHIMAENAGA_EXPORT BinMapper {
 public:
  BinMapper() = default;

  void Fit(const float* values, data_size_t n, int max_bin, FeatureType ftype);
  bin_t MapValue(float v) const;
  void  MapArray(const float* in, bin_t* out, data_size_t n) const;

  int  NumBins()       const { return static_cast<int>(thresholds_.size()) + 1; }
  FeatureType Type()   const { return ftype_; }
  bool IsCategorical() const { return ftype_ == FeatureType::Categorical; }

  const std::vector<float>&    Thresholds() const { return thresholds_; }
  const std::vector<uint32_t>& CatMap()     const { return cat_map_; }
  int MaxCatId()               const { return max_cat_id_; }

  // Reconstruction from serialized state (used by Serializer)
  void SetState(FeatureType ft, std::vector<float> thr,
                std::vector<uint32_t> cat, int max_cat) {
    ftype_       = ft;
    thresholds_  = std::move(thr);
    cat_map_     = std::move(cat);
    max_cat_id_  = max_cat;
  }

 private:
  FeatureType ftype_ = FeatureType::Numerical;
  std::vector<float>    thresholds_;
  std::vector<uint32_t> cat_map_;
  int max_cat_id_ = 0;

  // Grant access to the internal implementation
  friend class BinMapperImpl;
};

} // namespace shimaenaga
