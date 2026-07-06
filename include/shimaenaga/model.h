#pragma once
#include <vector>
#include <array>
#include <string>
#include <cstdint>
#include "config.h"
#include "export.h"
#include "tree.h"
#include "bin_mapper.h"

namespace shimaenaga {
struct TokenPlan;

// Tier-3: per-layer global parameters (Tier-3 詳細設計書 §4).
// Token index convention matches the training work arrays: p ∈ 0..P-1 are the
// token trees, p = P is the CLS token (gate tree).
struct Tier3Layer {
  // Context projections, flat row-major:
  //   Wq/Wk: [H][d_a][d_u], Wv: [H][d_u][d_u]
  std::vector<param_t> Wq, Wk, Wv;
  // Anchor vectors a_q/a_k: [H][P+1][d_a]. Layer 0 uses the qA/kA LEAF anchors
  // for p < P (E3.10) — its a_q/a_k rows are consulted only for the CLS token.
  std::vector<param_t> a_q, a_k;
  // Pair bias: [H][(P+1)*(P+1)] (CLS row/col included)
  std::vector<param_t> bA3;
  // Tree-FFN: W1 [d_f][d_u], W2 [d_u][d_f]; empty when d_f == 0
  std::vector<param_t> W1, W2;
  // FFN leaf bias c: token p leaves then gate leaves, [Σ_p L_p + L_g][d_f]
  std::vector<param_t> c1;
  // Head weights ρ^{(t)} and log-temperature θ^{(t)} (τ = exp(θ))
  std::array<float, kMaxHeads> rho3  = {};
  std::array<float, kMaxHeads> theta = {};
  // Carrier-mix gate (layers t >= 1): η_a^{(t)} = eta_attn·clamp01(γ)
  float gamma_c = 0.0f;
};

// Leaf parameters for one boosting block (詳細設計書 §10.1)
struct AttentiveBlock {
  // Tree structures (P token trees + 1 gate tree)
  std::vector<Tree> token_trees;
  Tree gate_tree;

  int P, H, C, d_a;
  std::string attention_mode;  // score_tree / qk_leaf
  int tier;

  // value  v[p][l][k]  shape: P * L_p * C
  // Flattened as v[p * L_max * C + l * C + k]
  std::vector<param_t> v;        // [P][L_p][C]
  std::vector<int>     v_lsize;  // L_p per token p

  // Gate: score_tree: z[L_g][H][P], qk_leaf: q[L_g][H][d_a]
  std::vector<param_t> z_or_q;  // gate leaf params
  int gate_num_leaves;

  // Key: k[p][l][h][d_a]
  std::vector<param_t> k;        // [P][L_p][H][d_a]

  // Tier-2 self-attention QA, KA: [p][l][h][d_a]
  std::vector<param_t> qA, kA;

  // Biases: b[h][p] (readout), bA[h][p][p'] (self-attn, flattened)
  std::vector<param_t> b;   // H * P
  std::vector<param_t> bA;  // H * P * P

  // Head weights: rho[h] (readout), rhoA[h] (self-attn)
  std::array<float, kMaxHeads> rho  = {};
  std::array<float, kMaxHeads> rhoA = {};

  // Self-attention mask rows: bit r of attn_mask[p] = token p may attend to r.
  // Empty = full attention (backward compatible with format v2 models).
  std::vector<uint32_t> attn_mask;  // size P when set

  // ─── Tier-3 (Tier-3 詳細設計書 §4; T_L == 0 ⇒ tier<=2 evaluation path) ───
  int T_L = 0, d_u = 0, d_f = 0;
  // Leaf hidden embeddings e: [P][L_p][d_u] (KOffset-like layout with d_u
  // instead of H*d_a), e_gate: [L_g][d_u]
  std::vector<param_t> e, e_gate;
  std::vector<Tier3Layer> layers;   // size T_L
  // Readout context projections WR/WK: [H][d_a][d_u]
  std::vector<param_t> WR, WK;
  // CLS head V_cls: [C][d_u] (E3.18)
  std::vector<param_t> V_cls;
  // Readout log-temperature θ^R (τ^R = exp(θ^R))
  std::array<float, kMaxHeads> theta_R = {};
  // e ≡ 0 && all projections ≡ 0 → the block evaluates exactly on the tier-2
  // path (anchored degenerate block; Tier-3 数学設計書 §11)
  bool ctx_frozen = false;

  size_t EOffset(int p) const {
    size_t off = 0;
    for (int pp = 0; pp < p; ++pp) off += (size_t)v_lsize[pp] * d_u;
    return off;
  }

  // Offsets into v for each token p
  size_t VOffset(int p) const {
    size_t off = 0;
    for (int pp = 0; pp < p; ++pp) off += v_lsize[pp] * C;
    return off;
  }
  size_t KOffset(int p) const {
    size_t off = 0;
    for (int pp = 0; pp < p; ++pp) off += v_lsize[pp] * H * d_a;
    return off;
  }
  size_t QAOffset(int p) const { return KOffset(p); }  // same shape
  size_t KAOffset(int p) const { return KOffset(p); }
};

// Full model (詳細設計書 §10.1)
struct SHIMAENAGA_EXPORT Model {
  int C = 1;
  std::vector<score_t> F0;  // initial prediction, size C
  std::vector<AttentiveBlock> blocks;
  Config train_cfg;
  // BinMappers saved at training time for portable prediction
  std::vector<BinMapper> bin_mappers;
};

} // namespace shimaenaga
