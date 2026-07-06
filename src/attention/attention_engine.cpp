#include "attention_engine.h"
#include "../data/token_planner.h"
#include "../util/simd.h"
#include "../util/random.h"
#include "../util/log.h"
#include <cmath>
#include <algorithm>
#include <cassert>
#include <cstring>

namespace shimaenaga {

// ─────────────────────────── BlockParams ────────────────────────────

void BlockParams::Init(int p, int h, int c, int da, const std::string& m,
                       int gate_leaves, const std::vector<int>& token_leaves,
                       int tier) {
  P = p; H = h; C = c; d_a = da; mode = m;
  gate_L = gate_leaves;
  token_L = token_leaves;
  gate_inner = (m == "qk_leaf") ? da : p;  // d_a or P

  // Per-token base offsets for the [P][L_p][...] flat tensors.
  tok_off.assign(P + 1, 0);
  v_off.assign(P + 1, 0);
  for (int pp = 0; pp < P; ++pp) {
    tok_off[pp + 1] = tok_off[pp] + (size_t)token_L[pp] * H * d_a;  // k/qA/kA stride
    v_off[pp + 1]   = v_off[pp]   + (size_t)token_L[pp] * C;        // v stride
  }

  // v[p][l][k] = 0 initially (Newton init is done by InitParams)
  v.assign(v_off[P], 0.0f);

  // gate params z_or_q[lg][h][d]
  z_or_q.assign((size_t)gate_leaves * H * gate_inner, 0.0f);

  // k[p][l][h][d_a]
  k.assign(tok_off[P], 0.0f);

  // Tier-2 self-attn
  if (tier >= 2) {
    qA.assign(tok_off[P], 0.0f);
    kA.assign(tok_off[P], 0.0f);
    bA.assign(h, {});
    for (int hh = 0; hh < h; ++hh)
      bA[hh].assign(p, std::vector<float>(p, 0.0f));
  }

  b.assign(h, std::vector<float>(p, 0.0f));

  // Uniform head weights
  rho.assign(h, 1.0f / h);
  rhoA.assign(h, 1.0f / h);
  omega.assign(h, 0.0f);
  omegaA.assign(h, 0.0f);
}

void BlockParams::UpdateRho() {
  // rho = softmax(omega)
  simd::softmax(omega.data(), rho.data(), H);
}
void BlockParams::UpdateRhoA() {
  simd::softmax(omegaA.data(), rhoA.data(), H);
}

// ─────────────────────────── AttentionEngine ────────────────────────────

AttentionEngine::AttentionEngine(const Config& cfg, data_size_t N, const TokenPlan& plan)
    : cfg_(cfg), N_(N), plan_(plan) {}

void AttentionEngine::InitParams(
    BlockParams& params,
    const std::vector<score_t>& g,
    const std::vector<score_t>& h,
    const BlockWork& work) const {

  int P = params.P, C = params.C, H = params.H;
  (void)H;
  double lambda_v = cfg_.lambda_v;

  // Newton-initialized leaf values: v_{pℓ,k} = -Σg / (Σh + λ)
  // Using uniform kappa = 1/P for initialization
  for (int p = 0; p < P; ++p) {
    int L = params.token_L[p];
    // Collect sum_g and sum_h per leaf per class
    std::vector<std::vector<double>> sum_g(L, std::vector<double>(C, 0.0));
    std::vector<std::vector<double>> sum_h(L, std::vector<double>(C, 0.0));
    std::vector<int> leaf_count(L, 0);

    for (data_size_t i = 0; i < N_; ++i) {
      leaf_t li = work.leaf_idx[(size_t)p * N_ + i];
      if (li < 0 || li >= L) continue;
      leaf_count[li]++;
      for (int k = 0; k < C; ++k) {
        sum_g[li][k] += g[(size_t)i * C + k];
        sum_h[li][k] += h[(size_t)i * C + k];
      }
    }
    for (int l = 0; l < L; ++l) {
      if (leaf_count[l] == 0) continue;
      float* vp = params.Vp(p, l);
      for (int k = 0; k < C; ++k)
        vp[k] = static_cast<float>(-sum_g[l][k] / (sum_h[l][k] + lambda_v));
    }
  }

  // Small random init for q, k (N(0, 0.01))
  if (params.mode == "qk_leaf") {
    Random rng(cfg_.seed + 1);
    for (int l = 0; l < params.gate_L; ++l)
      for (int hh = 0; hh < params.H; ++hh) {
        float* q = params.Qg(l, hh);
        for (int d = 0; d < params.d_a; ++d)
          q[d] = static_cast<float>((rng.NextDouble() - 0.5) * 0.02);
      }
    for (int p = 0; p < P; ++p)
      for (int l = 0; l < params.token_L[p]; ++l)
        for (int hh = 0; hh < params.H; ++hh) {
          float* kk = params.Kp(p, l, hh);
          for (int d = 0; d < params.d_a; ++d)
            kk[d] = static_cast<float>((rng.NextDouble() - 0.5) * 0.02);
        }
  }
}

void AttentionEngine::BuildReadoutLUT(const BlockParams& params, ReadoutLUT& lut) const {
  // S^R[h][p][gate_leaf][token_leaf] = q_{h,gate_leaf}^T k_{h,p,token_leaf} / sqrt(d_a) + b[h][p]
  lut.Alloc(params.H, params.P, params.gate_L, params.token_L);

  if (params.mode == "score_tree") {
    // z_or_q[gate_leaf][h][p] is the score directly
    for (int h = 0; h < params.H; ++h)
      for (int p = 0; p < params.P; ++p)
        for (int lg = 0; lg < params.gate_L; ++lg)
          for (int lp = 0; lp < params.token_L[p]; ++lp)
            lut.At(h, p, lg, lp) = params.Qg(lg, h)[p] + params.b[h][p];
    return;
  }

  // qk_leaf: inner product
  double inv_sqrt_da = 1.0 / std::sqrt((double)params.d_a);
  for (int h = 0; h < params.H; ++h)
    for (int p = 0; p < params.P; ++p)
      for (int lg = 0; lg < params.gate_L; ++lg) {
        const float* q = params.Qg(lg, h);
        for (int lp = 0; lp < params.token_L[p]; ++lp) {
          const float* kk = params.Kp(p, lp, h);
          float dot = simd::dot(q, kk, params.d_a);
          lut.At(h, p, lg, lp) = dot * (float)inv_sqrt_da + params.b[h][p];
        }
      }
}

void AttentionEngine::BuildSelfAttnLUT(const BlockParams& params, SelfAttnLUT& lut) const {
  // S^A[h][p][r][l_p][l_r] = qA_{h,p,l_p}^T kA_{h,r,l_r} / sqrt(d_a) + bA[h][p][r]
  lut.Alloc(params.H, params.P, params.token_L);
  double inv_sqrt_da = 1.0 / std::sqrt((double)params.d_a);
  for (int h = 0; h < params.H; ++h)
    for (int p = 0; p < params.P; ++p)
      for (int r = 0; r < params.P; ++r)
        for (int lp = 0; lp < params.token_L[p]; ++lp)
          for (int lr = 0; lr < params.token_L[r]; ++lr) {
            const float* q = params.QAp(p, lp, h);
            const float* k = params.KAp(r, lr, h);
            float dot = simd::dot(q, k, params.d_a);
            lut.At(h, p, r, lp, lr) = dot * (float)inv_sqrt_da + params.bA[h][p][r];
          }
}

void AttentionEngine::Forward(
    const BlockParams& params,
    const ReadoutLUT& rlut,
    const SelfAttnLUT& slut,
    const std::vector<score_t>& g,
    const std::vector<score_t>& h,
    BlockWork& work,
    bool warmup,
    const ForwardOpts& opts) const {

  int P = params.P, H = params.H, C = params.C;
  double eta = (cfg_.tier >= 2) ? cfg_.eta_attn : 0.0;
  bool tier2 = (cfg_.tier >= 2) && !warmup;

  // Self-attention mask rows (基本設計書 §5.4): bit r of mask[p] = r ∈ N(p).
  // full mask → all bits set (no-op).
  const uint32_t kFullMask = ~0u;
  const bool masked = tier2 && (int)plan_.mask_bits.size() == P &&
                      cfg_.attn_mask != "full";

  // λ_div head-diversity (E.41): G_h[p][r] = ∂Ω_div/∂Ā_h[p][r], computed from
  // the LAGGED Ā of the previous forward (1st-order only, per 数学§7.5). The
  // per-sample chain through the row softmax is applied in step 8b.
  const bool use_div = tier2 && opts.back_self && cfg_.lambda_div > 0 &&
                       H > 1 && work.abar_valid;
  std::vector<double> Gdiv;
  if (use_div) {
    const int PP = P * P;
    Gdiv.assign((size_t)H * PP, 0.0);
    std::vector<double> norm(H, 0.0);
    for (int h2 = 0; h2 < H; ++h2) {
      double s = 0.0;
      for (int t = 0; t < PP; ++t) {
        double v = work.a_bar[(size_t)h2 * PP + t];
        s += v * v;
      }
      norm[h2] = std::sqrt(s);
    }
    for (int h2 = 0; h2 < H; ++h2) {
      if (norm[h2] < 1e-12) continue;
      for (int h3 = 0; h3 < H; ++h3) {
        if (h3 == h2 || norm[h3] < 1e-12) continue;
        double dotv = 0.0;
        for (int t = 0; t < PP; ++t)
          dotv += work.a_bar[(size_t)h2 * PP + t] * work.a_bar[(size_t)h3 * PP + t];
        double cos_hh = dotv / (norm[h2] * norm[h3]);
        // ∂cos²/∂a_h = 2cos·( a_h' /(‖a_h‖‖a_h'‖) − cos·a_h/‖a_h‖² )
        for (int t = 0; t < PP; ++t) {
          double ah  = work.a_bar[(size_t)h2 * PP + t];
          double ahp = work.a_bar[(size_t)h3 * PP + t];
          Gdiv[(size_t)h2 * PP + t] += cfg_.lambda_div * 2.0 * cos_hh *
              (ahp / (norm[h2] * norm[h3]) - cos_hh * ah / (norm[h2] * norm[h2]));
        }
      }
    }
  }
  const double inv_N = 1.0 / std::max<data_size_t>(N_, 1);

  // Per-sample work is row-independent (writes only sample i's slots), so the
  // loop parallelizes cleanly and deterministically. `opts` selects which stages
  // to recompute; skipped stages reuse the values persisted in `work`.
  //
  // Temporaries are hoisted to per-thread buffers allocated once per thread
  // (not once per sample): with N up to millions and P·H per-sample heap
  // allocations, the previous in-loop std::vector churn dominated runtime.
  #pragma omp parallel
  {
    std::vector<float> s_raw(H * P);
    std::vector<float> alpha_i(H * P);
    std::vector<float> y0(P * C);
    std::vector<float> row(P);
    std::vector<double> Ybar(H * C);
    std::vector<double> yhat(C);
    std::vector<leaf_t> lp_loc(P + 1);  // per-sample leaf indices (P tokens + gate)

  #pragma omp for schedule(static)
  for (data_size_t i = 0; i < N_; ++i) {
    // Gather this sample's leaf indices once. leaf_idx is laid out [P+1][N], so
    // re-reading leaf_idx[p*N+i] inside the p/r loops below costs a strided
    // (cache-missy) load each time; the local copy keeps them hot in L1.
    for (int pp = 0; pp <= P; ++pp) lp_loc[pp] = work.leaf_idx[(size_t)pp * N_ + i];
    leaf_t lg_i = lp_loc[P];  // gate leaf

    // ── 1. Readout scores → alpha (E.11/E.12/E.13) ──
    // log α is cached alongside α (softmax_log): the entropy regularizer in
    // 8a and EvaluateQ then never calls log per (i,h,p).
    if (warmup) {
      const float la0 = -std::log((float)P);
      for (int h2 = 0; h2 < H; ++h2)
        for (int p = 0; p < P; ++p) {
          work.Alpha(i, h2, p)    = 1.0f / P;
          work.LogAlpha(i, h2, p) = la0;
        }
    } else if (opts.alpha) {
      for (int h2 = 0; h2 < H; ++h2) {
        for (int p = 0; p < P; ++p) {
          s_raw[h2 * P + p] = rlut.At(h2, p, lg_i, lp_loc[p]);
        }
        simd::softmax_log(s_raw.data() + h2 * P, alpha_i.data() + h2 * P,
                          s_raw.data() + h2 * P, P);  // s_raw reused as logα out
        for (int p = 0; p < P; ++p) {
          work.Alpha(i, h2, p)    = alpha_i[h2 * P + p];
          work.LogAlpha(i, h2, p) = s_raw[h2 * P + p];
        }
      }
    }  // else: reuse persisted work.Alpha / work.LogAlpha

    // ── 2. beta (E.14) — always (cheap; depends on α and ρ) ──
    for (int p = 0; p < P; ++p) {
      float beta_p = 0.0f;
      for (int h2 = 0; h2 < H; ++h2)
        beta_p += params.rho[h2] * work.Alpha(i, h2, p);
      work.Beta(i, p) = beta_p;
    }

    // ── 3. y0 = v[p][leaf_p(i)] (E.15 basis) — always (cheap gather) ──
    for (int p = 0; p < P; ++p) {
      const float* vp = params.Vp(p, lp_loc[p]);
      for (int k = 0; k < C; ++k)
        y0[(size_t)p * C + k] = vp[k];
    }

    // ── 4. Tier-2 self-attention A matrix + carrier y1 (E.20-E.22) ──
    if (tier2 && opts.selfattn) {
      for (int h2 = 0; h2 < H; ++h2)
        for (int p = 0; p < P; ++p) {
          const uint32_t mrow = masked ? plan_.mask_bits[p] : kFullMask;
          for (int r = 0; r < P; ++r) {
            row[r] = ((mrow >> r) & 1u)
                         ? slut.At(h2, p, r, lp_loc[p], lp_loc[r])
                         : -1e30f;  // masked → exp(·)=0 after max shift
          }
          simd::softmax(row.data(), row.data(), P);
          for (int r = 0; r < P; ++r)
            work.Aself(i, h2, p, r) = row[r];
        }
      for (int p = 0; p < P; ++p)
        for (int k = 0; k < C; ++k) {
          double mix = 0.0;
          for (int h2 = 0; h2 < H; ++h2)
            for (int r = 0; r < P; ++r)
              mix += params.rhoA[h2] * work.Aself(i, h2, p, r) * y0[(size_t)r * C + k];
          work.Y1(i, p, k) = (float)((1.0 - eta) * y0[p * C + k] + eta * mix);
        }
    } else if (!tier2) {
      // Tier-1 / warmup: carrier is the raw value.
      for (int p = 0; p < P; ++p)
        for (int k = 0; k < C; ++k)
          work.Y1(i, p, k) = y0[(size_t)p * C + k];
    }  // else (tier2 && !opts.selfattn): reuse persisted work.Y1 and work.Aself

    // ── 5. phi (E.15/E.23) — always ──
    for (int k = 0; k < C; ++k) {
      double phi_k = 0.0;
      for (int p = 0; p < P; ++p)
        phi_k += work.Beta(i, p) * work.Y1(i, p, k);
      work.Phi(i, k) = phi_k;
    }

    // ── 6. kappa (E.29/E.30) ──
    if (opts.kappa) {
      for (int p = 0; p < P; ++p) {
        if (tier2) {
          double kap = (1.0 - eta) * work.Beta(i, p);
          for (int pp = 0; pp < P; ++pp)
            for (int h2 = 0; h2 < H; ++h2)
              kap += eta * params.rhoA[h2] * work.Beta(i, pp) * work.Aself(i, h2, pp, p);
          work.Kappa(i, p) = static_cast<float>(kap);
        } else {
          work.Kappa(i, p) = work.Beta(i, p);  // (E.29)
        }
      }
    }  // else: reuse persisted work.Kappa

    // ── 7. residual r (E.26) — always ──
    for (int k = 0; k < C; ++k)
      work.R(i, k) = g[(size_t)i * C + k] + h[(size_t)i * C + k] * work.Phi(i, k);

    // ── 8a. readout backward c_read/d_read (E.32-E.34) ──
    if (!warmup && opts.back_read) {
      std::fill(Ybar.begin(), Ybar.end(), 0.0);
      for (int h2 = 0; h2 < H; ++h2)
        for (int p = 0; p < P; ++p)
          for (int k = 0; k < C; ++k)
            Ybar[h2 * C + k] += work.Alpha(i, h2, p) * work.Y1(i, p, k);

      const bool ent = cfg_.lambda_ent > 0;
      for (int h2 = 0; h2 < H; ++h2) {
        // Entropy regularizer 1st-order term (E.35 note):
        // ∂Ω_ent/∂s_{ihp} = λ_ent α (log α + 1 − Σ_{p'} α_{p'}(log α_{p'}+1)).
        // Added to c so every B2 consumer (z / q / k / b) sees it uniformly.
        // log α comes from the softmax_log cache — no log calls here.
        double S_ent = 0.0;
        if (ent) {
          for (int p = 0; p < P; ++p)
            S_ent += work.Alpha(i, h2, p) * (work.LogAlpha(i, h2, p) + 1.0);
        }
        for (int p = 0; p < P; ++p) {
          float cip = 0.0f, dip = 0.0f;
          double alpha_rho = params.rho[h2] * work.Alpha(i, h2, p);
          for (int k = 0; k < C; ++k) {
            double dphids = alpha_rho * (work.Y1(i, p, k) - Ybar[h2 * C + k]);
            cip += static_cast<float>(work.R(i, k) * dphids);     // (E.33)
            dip += static_cast<float>(h[(size_t)i * C + k] * dphids * dphids);  // (E.34)
          }
          if (ent) {
            cip += static_cast<float>(
                cfg_.lambda_ent * work.Alpha(i, h2, p) *
                (work.LogAlpha(i, h2, p) + 1.0 - S_ent));
          }
          dip += static_cast<float>(cfg_.eps_damping);
          work.Cread(i, h2, p) = cip;
          work.Dread(i, h2, p) = dip;
        }
      }
    }

    // ── 8b. self-attention backward c_self/d_self (E.39) ──
    if (tier2 && opts.back_self) {
      for (int h2 = 0; h2 < H; ++h2)
        for (int p = 0; p < P; ++p) {
          std::fill(yhat.begin(), yhat.end(), 0.0);
          for (int r = 0; r < P; ++r)
            for (int k = 0; k < C; ++k)
              yhat[k] += work.Aself(i, h2, p, r) * y0[(size_t)r * C + k];

          // λ_div chain through the row softmax (row p of head h2):
          // ∂Ω/∂s_r = (1/N)·A_r·(G[p][r] − Σ_{r'} G[p][r'] A_{r'})
          double gdot = 0.0;
          const double* Gp = use_div ? Gdiv.data() + ((size_t)h2 * P + p) * P : nullptr;
          if (use_div) {
            for (int r = 0; r < P; ++r)
              gdot += Gp[r] * work.Aself(i, h2, p, r);
          }

          for (int r = 0; r < P; ++r) {
            float cA = 0.0f, dA = 0.0f;
            double coeff = eta * work.Beta(i, p) * params.rhoA[h2] * work.Aself(i, h2, p, r);
            for (int k = 0; k < C; ++k) {
              double dphids = coeff * (y0[(size_t)r * C + k] - yhat[k]);
              cA += static_cast<float>(work.R(i, k) * dphids);
              dA += static_cast<float>(h[(size_t)i * C + k] * dphids * dphids);
            }
            if (use_div)
              cA += static_cast<float>(
                  inv_N * work.Aself(i, h2, p, r) * (Gp[r] - gdot));
            dA += static_cast<float>(cfg_.eps_damping);
            work.Cself(i, h2, p, r) = cA;
            work.Dself(i, h2, p, r) = dA;
          }
        }
    }
  }  // omp for
  }  // omp parallel

  // ── Ā aggregation (E.41): deterministic fixed-chunk reduction ──
  // Once per iteration (opts.abar, set by the iteration's first full forward);
  // consumed lagged by the sweeps' backward passes (数学§7.5, 1st-order only).
  if (tier2 && opts.selfattn && opts.abar && cfg_.lambda_div > 0 && H > 1) {
    constexpr int kChunks = 64;
    const int PP = P * P;
    std::vector<double> partial((size_t)kChunks * H * PP, 0.0);
    #pragma omp parallel for schedule(static) if ((int64_t)N_ * H * PP >= 65536)
    for (int c = 0; c < kChunks; ++c) {
      data_size_t s = (data_size_t)((int64_t)N_ * c / kChunks);
      data_size_t e = (data_size_t)((int64_t)N_ * (c + 1) / kChunks);
      double* acc = partial.data() + (size_t)c * H * PP;
      for (data_size_t i = s; i < e; ++i)
        for (int h2 = 0; h2 < H; ++h2)
          for (int t = 0; t < PP; ++t)
            acc[(size_t)h2 * PP + t] +=
                work.A_self[(size_t)i * H * PP + (size_t)h2 * PP + t];
    }
    std::fill(work.a_bar.begin(), work.a_bar.end(), 0.0);
    for (int c = 0; c < kChunks; ++c)
      for (size_t t = 0; t < (size_t)H * PP; ++t)
        work.a_bar[t] += partial[(size_t)c * H * PP + t];
    for (auto& v : work.a_bar) v *= inv_N;
    work.abar_valid = true;
  }
}

} // namespace shimaenaga
