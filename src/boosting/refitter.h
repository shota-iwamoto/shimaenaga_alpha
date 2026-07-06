#pragma once
#include "../attention/attention_engine.h"
#include "../../include/shimaenaga/config.h"

namespace shimaenaga {

// CSR index: for each token p, maps leaf l -> sample indices
struct LeafCSR {
  int P;
  std::vector<int> num_leaves;   // L_p per token, +1 for gate
  // For token p, leaf l: samples are sample_ids[p][leaf_begin[p][l]..leaf_begin[p][l+1]]
  std::vector<std::vector<int>> leaf_begin;    // [P+1][L_p+1]
  std::vector<std::vector<data_size_t>> sample_ids;  // [P+1][N]

  void Build(const BlockWork& work, int P, int gate_leaves,
             const std::vector<int>& token_leaves, data_size_t N);
};

// Phase B Newton updaters (詳細設計書 §7)
class Refitter {
 public:
  Refitter(const Config& cfg,
           const std::vector<score_t>& g_orig,
           const std::vector<score_t>& h_orig,
           BlockWork& work,
           BlockParams& params,
           AttentionEngine& engine,
           ReadoutLUT& rlut,
           SelfAttnLUT& slut);

  // Build CSR from current leaf assignments
  void BuildCSR();

  // (B1): Newton update of leaf values v (E.31)
  void RefitValues();

  // (B2): Newton update of readout score params (E.35-E.38)
  void RefitReadout();

  // (B3): Newton update of self-attention params (E.39-E.40)
  void RefitSelfAttn();

  // (B4): Newton update of head weights rho (E.41)
  void RefitHeads();

  // Evaluate Q_m objective (E.25)
  double EvaluateQ() const;

 private:
  const Config& cfg_;
  const std::vector<score_t>& g_;
  const std::vector<score_t>& h_;
  BlockWork& work_;
  BlockParams& params_;
  AttentionEngine& engine_;
  ReadoutLUT& rlut_;
  SelfAttnLUT& slut_;
  LeafCSR csr_;
  data_size_t N_;

  void RebuildLUTAndForward();
  void UpdatePhiIncremental(int p, const std::vector<std::vector<float>>& delta_v);
};

} // namespace shimaenaga
