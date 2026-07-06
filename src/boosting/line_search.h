#pragma once
#include "refitter.h"

namespace shimaenaga {

// Line search and monotonicity guarantee (詳細設計書 §7.6)
class LineSearch {
 public:
  LineSearch(const Config& cfg, Refitter& refitter, BlockParams& params,
             BlockWork& work, AttentionEngine& engine,
             ReadoutLUT& rlut, SelfAttnLUT& slut,
             const std::vector<score_t>& g, const std::vector<score_t>& h);

  // Save current state before a sweep
  void SaveState();

  // Check Q after sweep; backtrack if needed
  // Returns true if Q improved (or stayed the same)
  bool Check(double Q_prev);

 private:
  const Config& cfg_;
  Refitter& refitter_;
  BlockParams& params_;
  BlockWork& work_;
  AttentionEngine& engine_;
  ReadoutLUT& rlut_;
  SelfAttnLUT& slut_;
  const std::vector<score_t>& g_;
  const std::vector<score_t>& h_;

  // Saved state for rollback (φ/r are recomputed by a full forward on
  // rollback, so only the parameters need saving)
  BlockParams saved_params_;
};

} // namespace shimaenaga
