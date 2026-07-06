#pragma once
#include <vector>
#include <array>
#include <algorithm>
#include "../../include/shimaenaga/config.h"
#include "../../include/shimaenaga/dataset.h"

namespace shimaenaga {

// Working memory for one boosting iteration (詳細設計書 §6.1)
struct BlockWork {
  data_size_t N = 0;
  int P = 0, H = 0, C = 0, tier = 0;

  std::vector<leaf_t> leaf_idx;   // [P+1][N]  (P token trees + 1 gate)
  std::vector<float>  alpha;      // [N][H][P]
  std::vector<float>  log_alpha;  // [N][H][P]  cached log α (entropy reg, E.35/E.41)
  std::vector<float>  beta;       // [N][P]
  std::vector<float>  A_self;     // [N][H][P][P]  tier=2 only
  std::vector<float>  kappa;      // [N][P]
  std::vector<float>  y1;         // [N][P][C]  carrier (E.22); stored for partial-forward reuse
  std::vector<score_t> phi;       // [N][C]
  std::vector<score_t> r;         // [N][C]  residual gradient (E.26)

  // c[i][h][p], d[i][h][p] for readout score backward
  std::vector<float>  c_read;     // [N][H][P]
  std::vector<float>  d_read;     // [N][H][P]
  // c^A[i][h][p][r], d^A[i][h][p][r] for self-attn (tier=2)
  std::vector<float>  c_self;     // [N][H][P][P]
  std::vector<float>  d_self;     // [N][H][P][P]

  // Ā_h = (1/N)Σ_i A_{ih} (E.41 head-diversity regularizer, tier-2 only).
  // Aggregated once per forward that refreshes A; consumed LAGGED by the next
  // backward (数学設計書 §7.5: per-iteration collection, 1st-order only).
  std::vector<double> a_bar;      // [H][P][P]
  bool abar_valid = false;

  void Alloc(data_size_t n, int p, int h, int c, int tier_level) {
    N = n; P = p; H = h; C = c; tier = tier_level;
    leaf_idx.assign((size_t)(p + 1) * n, 0);
    alpha.assign((size_t)n * h * p, 0.0f);
    log_alpha.assign((size_t)n * h * p, 0.0f);
    beta.assign((size_t)n * p, 0.0f);
    kappa.assign((size_t)n * p, 0.0f);
    y1.assign((size_t)n * p * c, 0.0f);
    phi.assign((size_t)n * c, 0.0);
    r.assign((size_t)n * c, 0.0);
    c_read.assign((size_t)n * h * p, 0.0f);
    d_read.assign((size_t)n * h * p, 0.0f);
    if (tier_level >= 2) {
      A_self.assign((size_t)n * h * p * p, 0.0f);
      c_self.assign((size_t)n * h * p * p, 0.0f);
      d_self.assign((size_t)n * h * p * p, 0.0f);
      a_bar.assign((size_t)h * p * p, 0.0);
      abar_valid = false;
    }
  }

  // Accessors
  leaf_t& LeafIdx(int p_or_g, data_size_t i) {
    return leaf_idx[(size_t)p_or_g * N + i];
  }
  float& Alpha(data_size_t i, int h, int p) {
    return alpha[(size_t)i * H * P + h * P + p];
  }
  float& LogAlpha(data_size_t i, int h, int p) {
    return log_alpha[(size_t)i * H * P + h * P + p];
  }
  float& Beta(data_size_t i, int p) { return beta[(size_t)i * P + p]; }
  float& Kappa(data_size_t i, int p) { return kappa[(size_t)i * P + p]; }
  float& Y1(data_size_t i, int p, int k) { return y1[((size_t)i * P + p) * C + k]; }
  score_t& Phi(data_size_t i, int k) { return phi[(size_t)i * C + k]; }
  score_t& R(data_size_t i, int k) { return r[(size_t)i * C + k]; }
  float& Aself(data_size_t i, int h, int p, int q) {
    return A_self[(size_t)i * H * P * P + h * P * P + p * P + q];
  }
  float& Cread(data_size_t i, int h, int p) { return c_read[(size_t)i*H*P + h*P + p]; }
  float& Dread(data_size_t i, int h, int p) { return d_read[(size_t)i*H*P + h*P + p]; }
  float& Cself(data_size_t i, int h, int p, int q) {
    return c_self[(size_t)i * H * P * P + h * P * P + p * P + q];
  }
  float& Dself(data_size_t i, int h, int p, int q) {
    return d_self[(size_t)i * H * P * P + h * P * P + p * P + q];
  }
};

// Readout LUT: S^R[h][p][gate_leaf][token_leaf]  (E.40)
struct ReadoutLUT {
  int H, P, L_g, L_p_max;
  std::vector<int> L_p;             // L_p per token
  std::vector<float> data;          // [H][P][L_g][L_p_max]

  void Alloc(int h, int p, int lg, const std::vector<int>& lp) {
    H = h; P = p; L_g = lg; L_p = lp;
    L_p_max = *std::max_element(lp.begin(), lp.end());
    data.assign((size_t)H * P * L_g * L_p_max, 0.0f);
  }
  float& At(int h, int p, int lg, int lp) {
    return data[(size_t)h * P * L_g * L_p_max + p * L_g * L_p_max + lg * L_p_max + lp];
  }
  float  At(int h, int p, int lg, int lp) const {
    return data[(size_t)h * P * L_g * L_p_max + p * L_g * L_p_max + lg * L_p_max + lp];
  }
};

// Self-attention LUT: S^A[h][p][r][l_p][l_r]  (E.40)
struct SelfAttnLUT {
  int H, P, L_p_max;
  std::vector<int> L_p;
  std::vector<float> data;  // [H][P][P][L_p_max][L_p_max]

  void Alloc(int h, int p, const std::vector<int>& lp) {
    H = h; P = p; L_p = lp;
    L_p_max = *std::max_element(lp.begin(), lp.end());
    data.assign((size_t)H * P * P * L_p_max * L_p_max, 0.0f);
  }
  float& At(int h, int p, int r, int lp, int lr) {
    return data[(size_t)h * P * P * L_p_max * L_p_max +
                p * P * L_p_max * L_p_max +
                r * L_p_max * L_p_max +
                lp * L_p_max + lr];
  }
  float At(int h, int p, int r, int lp, int lr) const {
    return data[(size_t)h * P * P * L_p_max * L_p_max +
                p * P * L_p_max * L_p_max +
                r * L_p_max * L_p_max +
                lp * L_p_max + lr];
  }
};

// Block parameters (mutable during Phase B)
//
// Leaf-indexed parameter tensors (v, z_or_q, k, qA, kA) are stored as flat
// contiguous buffers rather than nested std::vector. The Phase-B refit kernels
// gather these per sample inside O(N·H·P) loops; nested vectors there cost a
// pointer-chase (cache miss) per access and made RefitSelfAttn memory-bound.
// Flat storage + index arithmetic keeps the hot loops contiguous.
struct BlockParams {
  int P, H, C, d_a, tier;
  std::string mode;  // score_tree / qk_leaf
  int gate_L;
  int gate_inner;            // = d_a (qk_leaf) or P (score_tree) — z_or_q stride
  std::vector<int> token_L;  // L_p per token
  std::vector<size_t> tok_off;  // tok_off[p] = base index of token p in v/k/qA/kA

  // v[p][l][k]  → v_flat[ tok_off_v[p] + l*C + k ]
  std::vector<float> v;
  std::vector<size_t> v_off;
  // Gate: z_or_q[lg][h][d] → [ (lg*H + h)*gate_inner + d ]   (d in 0..gate_inner)
  std::vector<float> z_or_q;
  // Key: k[p][l][h][d] → k_flat[ tok_off[p] + (l*H + h)*d_a + d ]
  std::vector<float> k;
  // Tier-2: qA[p][l][h][d], kA[p][l][h][d]  (same layout/offsets as k)
  std::vector<float> qA, kA;

  // Biases
  std::vector<std::vector<float>> b;    // [H][P]
  std::vector<std::vector<std::vector<float>>> bA;  // [H][P][P]

  // Head weights
  std::vector<float> rho;   // [H] - readout
  std::vector<float> rhoA;  // [H] - self-attn
  std::vector<float> omega;  // [H] - raw params for rho = softmax(omega)
  std::vector<float> omegaA;

  // Flat accessors (return pointer to the innermost contiguous run).
  float*       Vp(int p, int l)            { return v.data() + v_off[p] + (size_t)l * C; }
  const float* Vp(int p, int l) const      { return v.data() + v_off[p] + (size_t)l * C; }
  float*       Qg(int lg, int h)           { return z_or_q.data() + ((size_t)lg * H + h) * gate_inner; }
  const float* Qg(int lg, int h) const     { return z_or_q.data() + ((size_t)lg * H + h) * gate_inner; }
  float*       Kp(int p, int l, int h)         { return k.data()  + tok_off[p] + ((size_t)l * H + h) * d_a; }
  const float* Kp(int p, int l, int h) const   { return k.data()  + tok_off[p] + ((size_t)l * H + h) * d_a; }
  float*       QAp(int p, int l, int h)        { return qA.data() + tok_off[p] + ((size_t)l * H + h) * d_a; }
  const float* QAp(int p, int l, int h) const  { return qA.data() + tok_off[p] + ((size_t)l * H + h) * d_a; }
  float*       KAp(int p, int l, int h)        { return kA.data() + tok_off[p] + ((size_t)l * H + h) * d_a; }
  const float* KAp(int p, int l, int h) const  { return kA.data() + tok_off[p] + ((size_t)l * H + h) * d_a; }

  void Init(int p, int h, int c, int da, const std::string& m,
            int gate_leaves, const std::vector<int>& token_leaves, int tier);
  void UpdateRho();   // rho = softmax(omega)
  void UpdateRhoA();
};

// Selective recomputation flags for Forward. A full sweep does not need every
// quantity refreshed after each coordinate refit; skipping the unchanged ones
// removes ~2/3 of the O(N·H·P²) self-attention work per Phase-B sweep.
// beta, y0, phi, r are always recomputed (cheap, ≤ O(N·H·P)).
struct ForwardOpts {
  bool alpha     = true;  // readout softmax α (skip if readout params unchanged)
  bool selfattn  = true;  // self-attn A matrix + carrier y1 (skip if A unchanged)
  bool kappa     = true;  // κ (E.29/E.30)
  bool back_read = true;  // readout backward c_read/d_read
  bool back_self = true;  // self-attn backward c_self/d_self
  bool abar      = false; // aggregate Ā for λ_div/T6 — once per iteration
                          // (数学§7.5), set only by the iteration's first forward
};

// Main attention engine (詳細設計書 §6)
class AttentionEngine {
 public:
  AttentionEngine(const Config& cfg, data_size_t N, const TokenPlan& plan);

  // Initialize block params (v = Newton init, q,k ~ N(0,0.01), b=0)
  void InitParams(BlockParams& params,
                  const std::vector<score_t>& g,
                  const std::vector<score_t>& h,
                  const BlockWork& work) const;

  // Build LUT from current params (E.40)
  void BuildReadoutLUT(const BlockParams& params, ReadoutLUT& lut) const;
  void BuildSelfAttnLUT(const BlockParams& params, SelfAttnLUT& lut) const;

  // Forward pass (E.11-E.23)
  void Forward(const BlockParams& params,
               const ReadoutLUT& rlut,
               const SelfAttnLUT& slut,
               const std::vector<score_t>& g,
               const std::vector<score_t>& h,
               BlockWork& work,
               bool warmup = false,
               const ForwardOpts& opts = ForwardOpts{}) const;

  // Compute backward coefficients c, d (E.32-E.34, E.39)
  // (already done inside Forward for efficiency)

 private:
  const Config& cfg_;
  data_size_t N_;
  const TokenPlan& plan_;
};

} // namespace shimaenaga
