#include "refitter.h"
#include "../util/simd.h"
#include "../util/log.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cassert>

namespace shimaenaga {

// ─────────────────────────── LeafCSR ────────────────────────────

void LeafCSR::Build(const BlockWork& work, int P, int gate_leaves,
                    const std::vector<int>& token_leaves, data_size_t N) {
  this->P = P;
  num_leaves.resize(P + 1);
  for (int p = 0; p < P; ++p) num_leaves[p] = token_leaves[p];
  num_leaves[P] = gate_leaves;

  leaf_begin.resize(P + 1);
  sample_ids.resize(P + 1);

  for (int p = 0; p <= P; ++p) {
    int L = num_leaves[p];
    leaf_begin[p].assign(L + 1, 0);
    sample_ids[p].resize(N);

    // Count samples per leaf
    for (data_size_t i = 0; i < N; ++i) {
      int l = work.leaf_idx[(size_t)p * N + i];
      if (l >= 0 && l < L) leaf_begin[p][l + 1]++;
    }
    // Prefix sum
    for (int l = 0; l < L; ++l) leaf_begin[p][l + 1] += leaf_begin[p][l];
    // Fill sample_ids (counting sort)
    std::vector<int> cur(leaf_begin[p].begin(), leaf_begin[p].end());
    for (data_size_t i = 0; i < N; ++i) {
      int l = work.leaf_idx[(size_t)p * N + i];
      if (l >= 0 && l < L) sample_ids[p][cur[l]++] = i;
    }
  }
}

// ─────────────────────────── Refitter ────────────────────────────

Refitter::Refitter(const Config& cfg,
                   const std::vector<score_t>& g,
                   const std::vector<score_t>& h,
                   BlockWork& work,
                   BlockParams& params,
                   AttentionEngine& engine,
                   ReadoutLUT& rlut,
                   SelfAttnLUT& slut)
    : cfg_(cfg), g_(g), h_(h), work_(work), params_(params),
      engine_(engine), rlut_(rlut), slut_(slut),
      N_(work.N) {}

void Refitter::BuildCSR() {
  csr_.Build(work_, params_.P, params_.gate_L, params_.token_L, N_);
}

// Scalar-Newton step clamp (attn_step_clip): with a near-zero pseudo-hessian
// the raw step diverges, saturating softmax(ω)/softmax(s+b) to exact 0/1 —
// gradients then vanish and the head/token dies irrecoverably (observed as
// frozen training on some data draws). Clipping keeps updates finite while
// leaving well-conditioned steps untouched.
static inline double ClipStep(double step, double clip) {
  if (clip <= 0.0) return step;
  return std::max(-clip, std::min(clip, step));
}

// Fixed-count chunking: boundaries depend only on N, partials are merged in
// chunk order, so results are bit-identical for ANY thread count (T10) while
// the chunks themselves run in parallel.
static constexpr int kQChunks = 64;
static inline data_size_t ChunkBegin(data_size_t N, int c) {
  return (data_size_t)((int64_t)N * c / kQChunks);
}

double Refitter::EvaluateQ() const {
  // Q = Σ_i [g_i^T φ_i + ½ φ_i^T diag(h_i) φ_i] + Ω  (E.25 + E.41)
  int C = params_.C, H = params_.H, P = params_.P;
  double partial[kQChunks];
  const bool ent = cfg_.lambda_ent > 0;
  #pragma omp parallel for schedule(static) if ((int64_t)N_ * C * (1 + (ent ? H * P : 0)) >= 32768)
  for (int c = 0; c < kQChunks; ++c) {
    data_size_t s = ChunkBegin(N_, c), e = ChunkBegin(N_, c + 1);
    double q = 0.0;
    for (data_size_t i = s; i < e; ++i) {
      for (int k = 0; k < C; ++k) {
        double phi = work_.Phi(i, k);
        q += g_[(size_t)i * C + k] * phi + 0.5 * h_[(size_t)i * C + k] * phi * phi;
      }
      if (ent) {
        // λ_ent Σ_{h,p} α log α  (E.41 readout entropy) — logα is the cached
        // softmax_log output, so this is a pure multiply-add scan.
        for (int h2 = 0; h2 < H; ++h2)
          for (int p = 0; p < P; ++p)
            q += cfg_.lambda_ent * work_.Alpha(i, h2, p) * work_.LogAlpha(i, h2, p);
      }
    }
    partial[c] = q;
  }
  double Q = 0.0;
  for (int c = 0; c < kQChunks; ++c) Q += partial[c];

  // Ω L2 terms (E.41): λ_v‖v‖² + (qk_leaf: λ_q‖q‖², λ_k‖k‖² / score_tree:
  // λ_z‖z‖²) + Tier-2 λ_q‖qA‖², λ_k‖kA‖². These enter the B2/B3 Newton systems,
  // so the line-search accept test (§7.6) must see them too.
  auto sq_sum = [](const std::vector<float>& v) {
    double s = 0.0;
    for (float x : v) s += (double)x * x;
    return s;
  };
  Q += 0.5 * cfg_.lambda_v * sq_sum(params_.v);
  if (params_.mode == "score_tree") {
    Q += 0.5 * cfg_.lambda_z * sq_sum(params_.z_or_q);
  } else {
    Q += 0.5 * cfg_.lambda_q * sq_sum(params_.z_or_q);
    Q += 0.5 * cfg_.lambda_k * sq_sum(params_.k);
  }
  if (cfg_.tier >= 2) {
    Q += 0.5 * cfg_.lambda_q * sq_sum(params_.qA);
    Q += 0.5 * cfg_.lambda_k * sq_sum(params_.kA);
  }
  return Q;
}

// (B1): Newton update of leaf values v (E.31)
void Refitter::RefitValues() {
  int P = params_.P, C = params_.C;

  for (int p = 0; p < P; ++p) {  // token p: sequential (cyclic CD)
    int L = params_.token_L[p];

    // One region per token: each leaf computes its Newton delta AND applies the
    // incremental φ/r update to its own samples (disjoint across leaves). The
    // previous two back-to-back regions doubled the OpenMP barrier count.
    #pragma omp parallel for schedule(dynamic)
    for (int l = 0; l < L; ++l) {
      int beg = csr_.leaf_begin[p][l], end = csr_.leaf_begin[p][l + 1];
      if (beg == end) continue;

      float dv_buf[64];  // C bounded by kMaxLeaves-scale? no: C = num_class; heap fallback below
      std::vector<float> dv_dyn;
      float* dv = dv_buf;
      if (C > 64) { dv_dyn.resize(C); dv = dv_dyn.data(); }

      float* vp = params_.Vp(p, l);
      for (int k = 0; k < C; ++k) {
        double num = cfg_.lambda_v * vp[k];
        double den = cfg_.lambda_v;
        for (int idx = beg; idx < end; ++idx) {
          data_size_t i = csr_.sample_ids[p][idx];
          double kap = work_.Kappa(i, p);
          num += kap * work_.R(i, k);         // (E.31) numerator
          den += kap * kap * h_[(size_t)i * C + k];  // denominator
        }
        den = std::max(den, cfg_.eps_damping);
        dv[k] = static_cast<float>(-num / den);
        vp[k] += dv[k];
      }

      // Incremental phi update: φ_ik += κ_{ip} * Δv_{pℓ,k}; recompute r (E.26).
      for (int idx = beg; idx < end; ++idx) {
        data_size_t i = csr_.sample_ids[p][idx];
        double kap = work_.Kappa(i, p);
        for (int k = 0; k < C; ++k) {
          work_.Phi(i, k) += kap * dv[k];
          work_.R(i, k) = g_[(size_t)i * C + k] + h_[(size_t)i * C + k] * work_.Phi(i, k);
        }
      }
    }
  }
}

// (B2): Newton update of readout score parameters (E.35-E.38)
void Refitter::RefitReadout() {
  int P = params_.P, H = params_.H, d_a = params_.d_a;
  int gate_L = params_.gate_L;
  const double inv_sqrt = 1.0 / std::sqrt((double)d_a);
  assert(d_a <= simd::kMaxDa);

  if (params_.mode == "score_tree") {
    // Update z_{hp,ℓ} (E.35): Δz = -(Σ c + λ z) / (Σ d + λ)
    for (int l = 0; l < gate_L; ++l) {
      int beg = csr_.leaf_begin[P][l], end = csr_.leaf_begin[P][l + 1];
      if (beg == end) continue;
      for (int h = 0; h < H; ++h) {
        float* z = params_.Qg(l, h);  // gate_inner == P in score_tree mode
        for (int p = 0; p < P; ++p) {
          double num = cfg_.lambda_z * z[p];
          double den = cfg_.lambda_z;
          for (int idx = beg; idx < end; ++idx) {
            data_size_t i = csr_.sample_ids[P][idx];
            num += work_.Cread(i, h, p);
            den += work_.Dread(i, h, p);
          }
          den = std::max(den, cfg_.eps_damping);
          z[p] += static_cast<float>(-num / den);
        }
      }
    }
  } else {
    // qk_leaf: update q_{h,ℓ} for gate leaves (E.36-E.37).
    // (h,l) flattened into ONE parallel region — per-h regions were mostly
    // barrier overhead at H·gate_L tiny tasks each.
    #pragma omp parallel for schedule(dynamic)
    for (int hl = 0; hl < H * gate_L; ++hl) {
        int h = hl / gate_L, l = hl % gate_L;
        int beg = csr_.leaf_begin[P][l], end = csr_.leaf_begin[P][l + 1];
        if (beg == end) continue;

        // Build G_q ∈ R^{d_a}, A_q ∈ R^{d_a × d_a} (stack scratch, no heap)
        float G[simd::kMaxDa] = {};
        float A[simd::kMaxDa * simd::kMaxDa];
        std::fill(A, A + d_a * d_a, 0.0f);
        // A_q initialized with λ_q * I
        for (int d = 0; d < d_a; ++d) A[d * d_a + d] = static_cast<float>(cfg_.lambda_q);

        for (int idx = beg; idx < end; ++idx) {
          data_size_t i = csr_.sample_ids[P][idx];
          // Sum over all tokens p (E.36 includes Σ_p)
          for (int p = 0; p < P; ++p) {
            leaf_t lp_i = work_.leaf_idx[(size_t)p * N_ + i];
            const float* kp = params_.Kp(p, lp_i, h);
            float c_ihp = work_.Cread(i, h, p);
            float d_ihp = work_.Dread(i, h, p);
            for (int d = 0; d < d_a; ++d)
              G[d] += static_cast<float>(c_ihp * kp[d] * inv_sqrt);
            simd::outer_acc(A, d_ihp / d_a, kp, kp, d_a);
          }
        }
        // λ_q * q (regularization gradient)
        float* qg = params_.Qg(l, h);
        for (int d = 0; d < d_a; ++d)
          G[d] += static_cast<float>(cfg_.lambda_q * qg[d]);

        // Solve A_q Δq = -G (E.37)
        float delta_q[simd::kMaxDa];
        if (!simd::cholesky_solve(A, G, delta_q, d_a)) {
          // Fallback: diagonal Newton
          for (int d = 0; d < d_a; ++d)
            delta_q[d] = -G[d] / std::max(A[d * d_a + d], (float)cfg_.eps_damping);
        } else {
          for (int d = 0; d < d_a; ++d) delta_q[d] = -delta_q[d];
        }
        for (int d = 0; d < d_a; ++d) qg[d] += delta_q[d];
    }

    // Update k_{h,p,ℓ} for token leaves (E.38).
    // (h,p,l) flattened into ONE region — the per-(h,p) regions (H·P of them,
    // each over ≤64 leaves) were dominated by barrier spin.
    std::vector<int> tl_off(P + 1, 0);
    for (int p = 0; p < P; ++p) tl_off[p + 1] = tl_off[p] + params_.token_L[p];
    const int total_tl = tl_off[P];
    #pragma omp parallel for schedule(dynamic)
    for (int hpl = 0; hpl < H * total_tl; ++hpl) {
        int h = hpl / total_tl, t = hpl % total_tl;
        int p = 0;
        while (t >= tl_off[p + 1]) ++p;
        int l = t - tl_off[p];
        {
          int beg = csr_.leaf_begin[p][l], end = csr_.leaf_begin[p][l + 1];
          if (beg == end) continue;

          float G[simd::kMaxDa] = {};
          float A[simd::kMaxDa * simd::kMaxDa];
          std::fill(A, A + d_a * d_a, 0.0f);
          for (int d = 0; d < d_a; ++d) A[d * d_a + d] = static_cast<float>(cfg_.lambda_k);

          for (int idx = beg; idx < end; ++idx) {
            data_size_t i = csr_.sample_ids[p][idx];
            leaf_t lg_i = work_.leaf_idx[(size_t)P * N_ + i];
            const float* q = params_.Qg(lg_i, h);
            float c_ihp = work_.Cread(i, h, p);
            float d_ihp = work_.Dread(i, h, p);
            for (int d = 0; d < d_a; ++d)
              G[d] += static_cast<float>(c_ihp * q[d] * inv_sqrt);
            simd::outer_acc(A, d_ihp / d_a, q, q, d_a);
          }
          float* kp = params_.Kp(p, l, h);
          for (int d = 0; d < d_a; ++d)
            G[d] += static_cast<float>(cfg_.lambda_k * kp[d]);

          float delta_k[simd::kMaxDa];
          if (!simd::cholesky_solve(A, G, delta_k, d_a)) {
            for (int d = 0; d < d_a; ++d)
              delta_k[d] = -G[d] / std::max(A[d * d_a + d], (float)cfg_.eps_damping);
          } else {
            for (int d = 0; d < d_a; ++d) delta_k[d] = -delta_k[d];
          }
          for (int d = 0; d < d_a; ++d) kp[d] += delta_k[d];
        }
    }
  }

  // Update token bias b_{hp}: scalar Newton, Δb = -Σ_i c_{ihp} / (Σ_i d_{ihp} + ε_d).
  // Single parallel pass over samples with per-chunk partials (the previous
  // serial H·P × N re-scan was a hotspot); chunk merge order is fixed (T10).
  {
    std::vector<double> pnum((size_t)kQChunks * H * P, 0.0);
    std::vector<double> pden((size_t)kQChunks * H * P, 0.0);
    #pragma omp parallel for schedule(static) if ((int64_t)N_ * H * P >= 65536)
    for (int c = 0; c < kQChunks; ++c) {
      data_size_t s = ChunkBegin(N_, c), e = ChunkBegin(N_, c + 1);
      double* nm = pnum.data() + (size_t)c * H * P;
      double* dn = pden.data() + (size_t)c * H * P;
      for (data_size_t i = s; i < e; ++i)
        for (int h = 0; h < H; ++h)
          for (int p = 0; p < P; ++p) {
            nm[h * P + p] += work_.Cread(i, h, p);
            dn[h * P + p] += work_.Dread(i, h, p);
          }
    }
    for (int h = 0; h < H; ++h)
      for (int p = 0; p < P; ++p) {
        double num = 0.0, den = cfg_.eps_damping;
        for (int c = 0; c < kQChunks; ++c) {
          num += pnum[(size_t)c * H * P + h * P + p];
          den += pden[(size_t)c * H * P + h * P + p];
        }
        params_.b[h][p] += static_cast<float>(ClipStep(-num / den, cfg_.attn_step_clip));
      }
  }

  // Readout params changed → α changes; A/y1 are unaffected (reuse). The only
  // backward needed next is c_self for RefitSelfAttn (tier-2). c_read/κ are not
  // consumed again until the next sweep's RefitHeads forward recomputes them.
  engine_.BuildReadoutLUT(params_, rlut_);
  ForwardOpts opts;
  opts.alpha = true;
  opts.selfattn = false;
  opts.kappa = false;
  opts.back_read = false;
  opts.back_self = (cfg_.tier >= 2);
  engine_.Forward(params_, rlut_, slut_, g_, h_, work_, false, opts);
}

// (B3): Newton update of self-attention params (E.39-E.40)
void Refitter::RefitSelfAttn() {
  if (cfg_.tier < 2) return;

  int P = params_.P, H = params_.H, d_a = params_.d_a;
  const double inv_sqrt = 1.0 / std::sqrt((double)d_a);
  assert(d_a <= simd::kMaxDa);

  // Flat (h,p,l) task index shared by the qA and kA updates below: one region
  // each instead of H·P small ones (barrier spin dominated otherwise).
  std::vector<int> tl_off(P + 1, 0);
  for (int p = 0; p < P; ++p) tl_off[p + 1] = tl_off[p] + params_.token_L[p];
  const int total_tl = tl_off[P];

  // Update qA[p][l][h][d_a] (query for self-attention)
  #pragma omp parallel for schedule(dynamic)
  for (int hpl = 0; hpl < H * total_tl; ++hpl) {
      int h = hpl / total_tl, t = hpl % total_tl;
      int p = 0;
      while (t >= tl_off[p + 1]) ++p;
      int l = t - tl_off[p];
      {
        int beg = csr_.leaf_begin[p][l], end = csr_.leaf_begin[p][l + 1];
        if (beg == end) continue;

        float G[simd::kMaxDa] = {};
        float A[simd::kMaxDa * simd::kMaxDa];
        std::fill(A, A + d_a * d_a, 0.0f);
        for (int d = 0; d < d_a; ++d) A[d * d_a + d] = static_cast<float>(cfg_.lambda_q);

        for (int idx = beg; idx < end; ++idx) {
          data_size_t i = csr_.sample_ids[p][idx];
          // Sum over r in N(p) (E.40 for q^A)
          for (int r = 0; r < P; ++r) {
            leaf_t lr_i = work_.leaf_idx[(size_t)r * N_ + i];
            const float* kA = params_.KAp(r, lr_i, h);
            float cA = work_.Cself(i, h, p, r);
            float dA = work_.Dself(i, h, p, r);
            for (int d = 0; d < d_a; ++d)
              G[d] += static_cast<float>(cA * kA[d] * inv_sqrt);
            simd::outer_acc(A, dA / d_a, kA, kA, d_a);
          }
        }
        float* qAp = params_.QAp(p, l, h);
        for (int d = 0; d < d_a; ++d)
          G[d] += static_cast<float>(cfg_.lambda_q * qAp[d]);

        float delta[simd::kMaxDa];
        if (!simd::cholesky_solve(A, G, delta, d_a)) {
          for (int d = 0; d < d_a; ++d)
            delta[d] = -G[d] / std::max(A[d * d_a + d], (float)cfg_.eps_damping);
        } else {
          for (int d = 0; d < d_a; ++d) delta[d] = -delta[d];
        }
        for (int d = 0; d < d_a; ++d) qAp[d] += delta[d];
      }
  }

  // Update kA[r][l][h][d_a] - key for self-attention (E.40 with Σ_p)
  #pragma omp parallel for schedule(dynamic)
  for (int hrl = 0; hrl < H * total_tl; ++hrl) {
      int h = hrl / total_tl, t = hrl % total_tl;
      int r = 0;
      while (t >= tl_off[r + 1]) ++r;
      int l = t - tl_off[r];
      {
        int beg = csr_.leaf_begin[r][l], end = csr_.leaf_begin[r][l + 1];
        if (beg == end) continue;

        float G[simd::kMaxDa] = {};
        float A[simd::kMaxDa * simd::kMaxDa];
        std::fill(A, A + d_a * d_a, 0.0f);
        for (int d = 0; d < d_a; ++d) A[d * d_a + d] = static_cast<float>(cfg_.lambda_k);

        for (int idx = beg; idx < end; ++idx) {
          data_size_t i = csr_.sample_ids[r][idx];
          // Sum over p: r ∈ N(p) (full mask -> all p)  (E.40 M1 fix)
          for (int p = 0; p < P; ++p) {
            leaf_t lp_i = work_.leaf_idx[(size_t)p * N_ + i];
            const float* qA = params_.QAp(p, lp_i, h);
            float cA = work_.Cself(i, h, p, r);
            float dA = work_.Dself(i, h, p, r);
            for (int d = 0; d < d_a; ++d)
              G[d] += static_cast<float>(cA * qA[d] * inv_sqrt);
            simd::outer_acc(A, dA / d_a, qA, qA, d_a);
          }
        }
        float* kAp = params_.KAp(r, l, h);
        for (int d = 0; d < d_a; ++d)
          G[d] += static_cast<float>(cfg_.lambda_k * kAp[d]);

        float delta[simd::kMaxDa];
        if (!simd::cholesky_solve(A, G, delta, d_a)) {
          for (int d = 0; d < d_a; ++d)
            delta[d] = -G[d] / std::max(A[d * d_a + d], (float)cfg_.eps_damping);
        } else {
          for (int d = 0; d < d_a; ++d) delta[d] = -delta[d];
        }
        for (int d = 0; d < d_a; ++d) kAp[d] += delta[d];
      }
  }

  // Update bA[h][p][r] (pair bias). Same chunked single-pass reduction as the
  // readout bias: the serial H·P² × N loop dominated tier-2 Phase B.
  {
    const size_t HPP = (size_t)H * P * P;
    std::vector<double> pnum(kQChunks * HPP, 0.0);
    std::vector<double> pden(kQChunks * HPP, 0.0);
    #pragma omp parallel for schedule(static) if ((int64_t)N_ * (int64_t)HPP >= 65536)
    for (int c = 0; c < kQChunks; ++c) {
      data_size_t s = ChunkBegin(N_, c), e = ChunkBegin(N_, c + 1);
      double* nm = pnum.data() + (size_t)c * HPP;
      double* dn = pden.data() + (size_t)c * HPP;
      for (data_size_t i = s; i < e; ++i)
        for (int h = 0; h < H; ++h)
          for (int p = 0; p < P; ++p)
            for (int r = 0; r < P; ++r) {
              nm[(size_t)h * P * P + p * P + r] += work_.Cself(i, h, p, r);
              dn[(size_t)h * P * P + p * P + r] += work_.Dself(i, h, p, r);
            }
    }
    for (int h = 0; h < H; ++h)
      for (int p = 0; p < P; ++p)
        for (int r = 0; r < P; ++r) {
          size_t idx = (size_t)h * P * P + p * P + r;
          double num = 0.0, den = cfg_.eps_damping;
          for (int c = 0; c < kQChunks; ++c) {
            num += pnum[(size_t)c * HPP + idx];
            den += pden[(size_t)c * HPP + idx];
          }
          params_.bA[h][p][r] += static_cast<float>(ClipStep(-num / den, cfg_.attn_step_clip));
        }
  }

  // Self-attn params changed → recompute A + y1 only. α unchanged (reuse). The
  // next op (RefitHeads) needs only α/β/r, so all backward is skipped here.
  engine_.BuildSelfAttnLUT(params_, slut_);
  ForwardOpts opts;
  opts.alpha = false;
  opts.selfattn = true;
  opts.kappa = false;
  opts.back_read = false;
  opts.back_self = false;
  engine_.Forward(params_, rlut_, slut_, g_, h_, work_, false, opts);
}

// (B4): Newton update of head weights rho (E.41)
void Refitter::RefitHeads() {
  int P = params_.P, H = params_.H, C = params_.C;
  // rho = softmax(omega); update omega_h via scalar Newton
  // ∂φ/∂ω_h = Σ_p (∂φ/∂β_p)(∂β_p/∂ω_h)
  // ∂β_p/∂ω_h = ρ_h(α_{hp} - β_p)

  // c_h = Σ_p Σ_k r_{ik} * (∂φ_{ik}/∂β_p) * (∂β_p/∂ω_h)
  // ∂φ_{ik}/∂β_p = y¹_{ipk} (the carrier; (E.32) footnote). For Tier-1 the
  // carrier equals v_{pℓ}, for Tier-2 it is the self-attention mix — Y1 holds
  // the correct value for both, so this is exact, not the prior v-approximation.
  // All heads are accumulated in ONE chunk-parallel pass over samples (ρ is only
  // refreshed after the loop, so per-h reads are consistent); merge order fixed.
  {
    std::vector<double> pgrad((size_t)kQChunks * H, 0.0);
    std::vector<double> phess((size_t)kQChunks * H, 0.0);
    #pragma omp parallel for schedule(static) if ((int64_t)N_ * H * P * C >= 65536)
    for (int c = 0; c < kQChunks; ++c) {
      data_size_t s = ChunkBegin(N_, c), e = ChunkBegin(N_, c + 1);
      double* gr = pgrad.data() + (size_t)c * H;
      double* hs = phess.data() + (size_t)c * H;
      for (data_size_t i = s; i < e; ++i)
        for (int h = 0; h < H; ++h)
          for (int p = 0; p < P; ++p) {
            double dbeta_domega = params_.rho[h] * (work_.Alpha(i, h, p) - work_.Beta(i, p));
            for (int k = 0; k < C; ++k) {
              double dphi_dbeta = work_.Y1(i, p, k);
              gr[h] += work_.R(i, k) * dphi_dbeta * dbeta_domega;
              hs[h] += h_[(size_t)i * C + k] * dphi_dbeta * dphi_dbeta *
                       dbeta_domega * dbeta_domega;
            }
          }
    }
    for (int h = 0; h < H; ++h) {
      double grad_omega = 0.0, hess_omega = 0.0;
      for (int c = 0; c < kQChunks; ++c) {
        grad_omega += pgrad[(size_t)c * H + h];
        hess_omega += phess[(size_t)c * H + h];
      }
      hess_omega = std::max(hess_omega, cfg_.eps_damping);
      params_.omega[h] += static_cast<float>(
          ClipStep(-grad_omega / hess_omega, cfg_.attn_step_clip));
    }
  }
  // ρ^A (self-attention head weights, Tier-2): scalar Newton on ω^A — the same
  // (B4) update the readout heads get. Was previously missing entirely, leaving
  // ρ^A frozen at uniform. With ρ^A = softmax(ω^A):
  //   ∂φ_ik/∂ω^A_h = η ρ^A_h Σ_p β_ip (m_ihpk − m̄_ipk),
  //   m_ihpk = Σ_r A_ihpr y⁰_irk (head-h mix), m̄ = Σ_h' ρ^A_h' m (E.22 の合成).
  const bool tier2 = (cfg_.tier >= 2) && !params_.qA.empty();
  if (tier2 && cfg_.eta_attn > 0.0) {
    const double eta = cfg_.eta_attn;
    std::vector<double> pgrad((size_t)kQChunks * H, 0.0);
    std::vector<double> phess((size_t)kQChunks * H, 0.0);
    #pragma omp parallel for schedule(static) if ((int64_t)N_ * H * P * P * C >= 65536)
    for (int c = 0; c < kQChunks; ++c) {
      data_size_t s = ChunkBegin(N_, c), e = ChunkBegin(N_, c + 1);
      double* gr = pgrad.data() + (size_t)c * H;
      double* hs = phess.data() + (size_t)c * H;
      std::vector<double> y0((size_t)P * C), m((size_t)H * P * C), mbar((size_t)P * C);
      for (data_size_t i = s; i < e; ++i) {
        for (int p = 0; p < P; ++p) {
          const float* vp = params_.Vp(p, work_.leaf_idx[(size_t)p * N_ + i]);
          for (int k = 0; k < C; ++k) y0[(size_t)p * C + k] = vp[k];
        }
        std::fill(m.begin(), m.end(), 0.0);
        std::fill(mbar.begin(), mbar.end(), 0.0);
        for (int h = 0; h < H; ++h)
          for (int p = 0; p < P; ++p)
            for (int r = 0; r < P; ++r) {
              double a = work_.Aself(i, h, p, r);
              for (int k = 0; k < C; ++k)
                m[((size_t)h * P + p) * C + k] += a * y0[(size_t)r * C + k];
            }
        for (int h = 0; h < H; ++h)
          for (int p = 0; p < P; ++p)
            for (int k = 0; k < C; ++k)
              mbar[(size_t)p * C + k] += params_.rhoA[h] * m[((size_t)h * P + p) * C + k];
        for (int h = 0; h < H; ++h) {
          for (int k = 0; k < C; ++k) {
            double dphi = 0.0;
            for (int p = 0; p < P; ++p)
              dphi += work_.Beta(i, p) *
                      (m[((size_t)h * P + p) * C + k] - mbar[(size_t)p * C + k]);
            dphi *= eta * params_.rhoA[h];
            gr[h] += work_.R(i, k) * dphi;
            hs[h] += h_[(size_t)i * C + k] * dphi * dphi;
          }
        }
      }
    }
    for (int h = 0; h < H; ++h) {
      double grad = 0.0, hess = 0.0;
      for (int c = 0; c < kQChunks; ++c) {
        grad += pgrad[(size_t)c * H + h];
        hess += phess[(size_t)c * H + h];
      }
      hess = std::max(hess, cfg_.eps_damping);
      params_.omegaA[h] += static_cast<float>(
          ClipStep(-grad / hess, cfg_.attn_step_clip));
    }
  }

  params_.UpdateRho();
  params_.UpdateRhoA();
  // ρ changed → β/φ/κ change; ρ^A changed → y1 too (recompute A/y1 for tier-2).
  // This is the last forward of the sweep, so it must refresh κ and c_read for
  // the next sweep's RefitValues / RefitReadout. c_self is regenerated by next F1.
  ForwardOpts opts;
  opts.alpha = false;
  opts.selfattn = tier2;
  opts.kappa = true;
  opts.back_read = true;
  opts.back_self = false;
  engine_.Forward(params_, rlut_, slut_, g_, h_, work_, false, opts);
}

} // namespace shimaenaga
