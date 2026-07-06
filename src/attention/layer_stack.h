#pragma once
// Tier-3 Deep Attentive Tree Block — training engine (Tier-3 詳細設計書 §5-§9).
//
// Implements the Tier-3 数学設計書 (docs/04) equations:
//   (E3.1)-(E3.3)   hidden stream init + RMSNorm
//   (E3.10)-(E3.16) contextual QK / carrier & hidden mixing / Tree-FFN
//   (E3.17)-(E3.18) deep readout + CLS head
//   (E3.20)         κ (value-sensitivity, exact — κ 保存定理)
//   (E3.21)         anchored initialization (all Tier-3 params = 0)
//   (E3.22)-(E3.27) adjoint backward + GN statistics
//   (E3.28)         line-search ladder (implemented by Tier3Engine::PhaseB)
//
// v2.0-alpha deviation from the design doc (documented in docs/05 §3.1): the
// StreamingExecutor is not implemented yet — activations are held for all N
// rows like the tier<=2 BlockWork. Correctness-first; the row-block streaming
// refactor is a follow-up (T19 scope).
//
// Token index convention: τ ∈ 0..P-1 are token trees, τ = P is the CLS token
// (gate tree) — identical to the BlockWork::leaf_idx layout.
#include <vector>
#include <array>
#include <string>
#include "../../include/shimaenaga/config.h"
#include "../../include/shimaenaga/model.h"
#include "attention_engine.h"

namespace shimaenaga {

struct TokenPlan;

// ─── Tier-3 trainable parameters (Tier-3 詳細設計書 §4 training-side) ───
struct Tier3Params {
  int P = 0, H = 0, C = 0, d_a = 0, d_u = 0, d_f = 0, T_L = 0;
  int gate_L = 0;
  std::vector<int> token_L;

  // Leaf hidden embeddings (E3.1): e[e_off[p] + l*d_u + j], e_gate[l*d_u + j]
  std::vector<size_t> e_off;   // size P+1 (prefix)
  std::vector<float> e;        // [P][L_p][d_u]
  std::vector<float> e_gate;   // [L_g][d_u]

  // FFN leaf-bias offsets shared across layers: token p leaf l lives at
  // (c1_off[p] + l) * d_f; the gate (CLS) leaves at (c1_off[P] + l) * d_f.
  std::vector<size_t> c1_off;  // size P+2 (prefix over P tokens + gate)

  struct Layer {
    std::vector<float> Wq, Wk;      // [H][d_a][d_u]
    std::vector<float> Wv;          // [H][d_u][d_u]
    std::vector<float> a_q, a_k;    // [H][P+1][d_a] (layer 0: CLS row only)
    std::vector<float> bA3;         // [H][(P+1)*(P+1)]
    std::vector<float> W1;          // [d_f][d_u]
    std::vector<float> W2;          // [d_u][d_f]
    std::vector<float> c1;          // [(Σ_p L_p) + L_g][d_f]
    std::vector<float> omega3;      // [H] (ρ3 = softmax(ω3))
    std::vector<float> rho3;        // [H]
    std::vector<float> theta;       // [H] log-temperature, |θ| <= ln 4
    // Carrier-mix gate for layers t >= 1: η_a^{(t)} = eta_attn·clamp01(γ).
    // γ = 0 at anchored init keeps deeper layers a carrier identity, so the
    // Phase-B start point equals Tier-2 for ANY T_L (数学 §8.1 の強化)。
    float gamma_c = 0.0f;
  };
  std::vector<Layer> layers;        // size T_L

  std::vector<float> WR, WK;        // [H][d_a][d_u] readout context projections
  std::vector<float> V_cls;         // [C][d_u]
  std::vector<float> theta_R;       // [H]

  // Anchored init (E3.21): everything zero (ρ3 uniform). With e ≡ 0 the whole
  // stack is exactly the Tier-2 forward — Phase B starts at the Tier-2 point.
  void Init(int P_, int H_, int C_, int d_a_, int d_u_, int d_f_, int T_L_,
            int gate_L_, const std::vector<int>& token_L_);
  void UpdateRho3(int t);

  float*       Ep(int p, int l)       { return e.data() + e_off[p] + (size_t)l * d_u; }
  const float* Ep(int p, int l) const { return e.data() + e_off[p] + (size_t)l * d_u; }
  float*       Eg(int l)              { return e_gate.data() + (size_t)l * d_u; }
  const float* Eg(int l) const        { return e_gate.data() + (size_t)l * d_u; }
  // c1 for token τ (τ == P → gate leaves)
  float*       C1(int t, int tau, int l)       { return layers[t].c1.data() + (c1_off[tau] + l) * d_f; }
  const float* C1(int t, int tau, int l) const { return layers[t].c1.data() + (c1_off[tau] + l) * d_f; }
};

// ─── Tier-3 per-iteration work (all-N activations; §5.2 の全行版) ───
struct Tier3Work {
  data_size_t N = 0;
  int P = 0, H = 0, C = 0, d_u = 0, d_f = 0, T_L = 0, T = 0;  // T = P+1

  // Activations (forward), layer index t: u/un/inv_rms have T_L+1 slots
  // (un[T_L] is the readout-time ũ), the rest have T_L slots.
  std::vector<float> u;         // [T_L+1][N][T][d_u]
  std::vector<float> un;        // [T_L+1][N][T][d_u]
  std::vector<float> inv_rms;   // [T_L+1][N][T]
  std::vector<float> unm;       // [T_L][N][T][d_u]   ũ of u_mid (FFN input)
  std::vector<float> inv_rms_m; // [T_L][N][T]
  std::vector<float> zf;        // [T_L][N][T][d_f]   FFN pre-activation
  std::vector<float> A3;        // [T_L][N][H][T][T]  layer attention
  std::vector<float> y;         // [T_L+1][N][P][C]   carrier per level

  // Backward products (consumed by the leaf-CSR / chunked solves)
  std::vector<float> cs;        // [T_L][N][H][T][T]  score adjoint (E3.26)
  std::vector<float> ds;        // [T_L][N][H][T][T]  score curvature (diag GN)
  std::vector<float> lam_u0;    // [N][T][d_u]        adjoint at u^(0) → e refit
  std::vector<float> mu_u0;     // [N][T][d_u]        curvature at u^(0)
  std::vector<float> lam_z;     // [T_L][N][T][d_f]   adjoint at z → c1 refit
  std::vector<float> mu_z;      // [T_L][N][T][d_f]

  void Alloc(data_size_t n, int P_, int H_, int C_, int d_u_, int d_f_, int T_L_);

  size_t UIdx(int t, data_size_t i, int tau) const {
    return (((size_t)t * N + i) * T + tau) * d_u;
  }
  size_t RmsIdx(int t, data_size_t i, int tau) const {
    return ((size_t)t * N + i) * T + tau;
  }
  size_t ZIdx(int t, data_size_t i, int tau) const {
    return (((size_t)t * N + i) * T + tau) * d_f;
  }
  size_t AIdx(int t, data_size_t i, int h, int tp, int tr) const {
    return ((((size_t)t * N + i) * H + h) * T + tp) * T + tr;
  }
  size_t YIdx(int t, data_size_t i, int p) const {
    return (((size_t)t * N + i) * P + p) * C;
  }
};

// ─── GN statistics from BackwardT3 (chunk-reduced, deterministic) ───
// G_* are pure data gradients (Σ_i r·∂φ/∂θ, entropy included on the readout
// side); regularization gradients are added inside the solves.
struct Tier3Stats {
  // Per layer
  struct LayerStats {
    std::vector<double> G_Wq, D_Wq, G_Wk, D_Wk;   // [H][d_a][d_u]
    std::vector<double> G_Wv, D_Wv;               // [H][d_u][d_u]
    std::vector<double> G_aq, D_aq, G_ak, D_ak;   // [H][T][d_a]
    std::vector<double> G_bA, D_bA;               // [H][T*T]
    std::vector<double> G_W1, D_W1;               // [d_f][d_u]
    std::vector<double> G_W2, D_W2;               // [d_u][d_f]
    std::vector<double> G_th, D_th;               // [H] temperature
    std::vector<double> G_om, D_om;               // [H] ω3
    double G_gc = 0.0, D_gc = 0.0;                // carrier gate γ (t >= 1)
  };
  std::vector<LayerStats> layer;

  std::vector<double> G_WR, D_WR, G_WK, D_WK;     // [H][d_a][d_u]
  std::vector<double> G_thR, D_thR;               // [H]
  std::vector<double> G_V;                        // [C][d_u]  (E3.22)
  std::vector<double> A_V;                        // [C][d_u][d_u] exact ridge
  std::vector<double> G_b, D_b;                   // [H][P] readout bias
  std::vector<double> G_omR, D_omR;               // [H] readout head ω

  void Alloc(int H, int T, int d_a, int d_u, int d_f, int C, int T_L);
  void Zero();
  void Add(const Tier3Stats& o);
};

// ─── Tier-3 engine: forward / backward / Phase-B refit + ladder ───
class Tier3Engine {
 public:
  Tier3Engine(const Config& cfg, data_size_t N, const TokenPlan& plan);

  // Anchored init of the Tier-3 side. The Tier-2 side (v/q/k/qA/kA/b/bA/ρ)
  // stays whatever AttentionEngine::InitParams produced.
  void InitParams(Tier3Params& t3p, const BlockParams& params) const;

  // Full forward (E3.1-E3.18 + κ E3.20). Fills work.{Alpha,LogAlpha,Beta,
  // Kappa,Phi,R} (so Refitter::RefitValues / EvaluateQ patterns stay valid)
  // and t3w activations.
  void Forward(const BlockParams& params, const Tier3Params& t3p,
               const std::vector<score_t>& g, const std::vector<score_t>& h,
               BlockWork& work, Tier3Work& t3w) const;

  // Adjoint backward (E3.23-E3.27): fills work.Cread/Dread, t3w.cs/ds/
  // lam_u0/mu_u0/lam_z/mu_z and the chunk-reduced global stats.
  void Backward(const BlockParams& params, const Tier3Params& t3p,
                const std::vector<score_t>& g, const std::vector<score_t>& h,
                BlockWork& work, Tier3Work& t3w, Tier3Stats& stats) const;

  // Q_m (E.25) + readout entropy + all L2 terms incl. Ω₃ (E3.29).
  double EvaluateQ(const BlockParams& params, const Tier3Params& t3p,
                   const std::vector<score_t>& g, const std::vector<score_t>& h,
                   const BlockWork& work) const;

  // Phase B (Tier-3 数学設計書 §8.2): inner_refit_steps sweeps of
  //   B1 (values, exact) → backward → B2'/B3/B4/B5/B6 solves → ladder (E3.28).
  // g/h fixed for the iteration. Returns final Q.
  double PhaseB(BlockParams& params, Tier3Params& t3p,
                const std::vector<score_t>& g, const std::vector<score_t>& h,
                BlockWork& work, Tier3Work& t3w);

  const Tier3Stats& Stats() const { return stats_; }  // (T14 gradient tests)

  // Row masks over columns 0..P (bit P = CLS), from the plan + use_cls_token.
  uint32_t RowMask(int tau) const { return row_mask_[tau]; }

 private:
  const Config& cfg_;
  data_size_t N_;
  std::vector<uint32_t> row_mask_;   // [P+1] built from the plan at ctor
  Tier3Stats stats_;

  // Solve+apply all GN updates from stats_ / work / t3w (B2'-B6).
  void ApplyUpdates(BlockParams& params, Tier3Params& t3p,
                    const std::vector<score_t>& g, const std::vector<score_t>& h,
                    const BlockWork& work, const Tier3Work& t3w) const;

  // B1: exact leaf-value Newton with κ (reuses the (E.31) kernel via CSR).
  void RefitValues(BlockParams& params, BlockWork& work,
                   const std::vector<score_t>& g,
                   const std::vector<score_t>& h,
                   const std::vector<std::vector<int>>& leaf_begin,
                   const std::vector<std::vector<data_size_t>>& sample_ids) const;

  void SpectralProject(Tier3Params& t3p) const;
};

} // namespace shimaenaga
