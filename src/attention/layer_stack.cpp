#include "layer_stack.h"
#include "../data/token_planner.h"
#include "../boosting/refitter.h"   // LeafCSR (leaf → samples, counting sort)
#include "../util/simd.h"
#include "../util/log.h"
#include <cmath>
#include <algorithm>
#include <cassert>
#include <cstring>

namespace shimaenaga {

// Fixed-count chunking (T10/T18 determinism): boundaries depend only on N and
// partials merge in chunk order — bit-identical for any thread count.
static constexpr int kChunks = 64;
static inline data_size_t ChunkBegin(data_size_t N, int c) {
  return (data_size_t)((int64_t)N * c / kChunks);
}

static inline double ClipStep(double step, double clip) {
  if (clip <= 0.0) return step;
  return std::max(-clip, std::min(clip, step));
}

static const double kLnTauMax = std::log(4.0);  // |θ| <= ln 4 (M6 再訪の安全弁)

// out[c] += Σ_r W[r*cols+c] x[r]
static inline void MatTVecAcc(const float* W, const double* x, double* out,
                              int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    const float* Wr = W + (size_t)r * cols;
    double xr = x[r];
    for (int c = 0; c < cols; ++c) out[c] += (double)Wr[c] * xr;
  }
}
// out[c] += Σ_r W[r*cols+c]^2 x[r]   (squared-Jacobian μ propagation)
static inline void MatT2VecAcc(const float* W, const double* x, double* out,
                               int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    const float* Wr = W + (size_t)r * cols;
    double xr = x[r];
    for (int c = 0; c < cols; ++c) out[c] += (double)Wr[c] * Wr[c] * xr;
  }
}

// (E3.2): ũ = u / sqrt(mean(u²) + ε); returns 1/RMS.
static inline float RmsNormF(const float* u, float* un, int d, double eps) {
  double s = 0.0;
  for (int j = 0; j < d; ++j) s += (double)u[j] * u[j];
  double inv = 1.0 / std::sqrt(s / d + eps);
  for (int j = 0; j < d; ++j) un[j] = (float)(u[j] * inv);
  return (float)inv;
}

// (E3.3) transpose-apply: λu += Jᵀ λũ = (λũ − ũ (ũ·λũ)/d) * inv_rms
static inline void RmsJacTAcc(const float* un, float inv_rms, const double* lam_un,
                              double* lam_u, int d) {
  double dot = 0.0;
  for (int j = 0; j < d; ++j) dot += (double)un[j] * lam_un[j];
  dot /= d;
  for (int j = 0; j < d; ++j)
    lam_u[j] += (lam_un[j] - (double)un[j] * dot) * inv_rms;
}

// Double-precision Cholesky solve for the exact V_cls ridge (E3.22), d <= 16.
static bool CholSolveD(const double* A, const double* b, double* x, int d) {
  double L[kMaxDHidden * kMaxDHidden];
  for (int i = 0; i < d; ++i)
    for (int j = 0; j <= i; ++j) {
      double s = A[i * d + j];
      for (int k = 0; k < j; ++k) s -= L[i * d + k] * L[j * d + k];
      if (i == j) {
        if (s <= 0.0) return false;
        L[i * d + i] = std::sqrt(s);
      } else {
        L[i * d + j] = s / L[j * d + j];
      }
    }
  double y[kMaxDHidden];
  for (int i = 0; i < d; ++i) {
    double s = b[i];
    for (int k = 0; k < i; ++k) s -= L[i * d + k] * y[k];
    y[i] = s / L[i * d + i];
  }
  for (int i = d - 1; i >= 0; --i) {
    double s = y[i];
    for (int k = i + 1; k < d; ++k) s -= L[k * d + i] * x[k];
    x[i] = s / L[i * d + i];
  }
  return true;
}

// ─────────────────────────── Tier3Params ────────────────────────────

void Tier3Params::Init(int P_, int H_, int C_, int d_a_, int d_u_, int d_f_,
                       int T_L_, int gate_L_, const std::vector<int>& token_L_) {
  P = P_; H = H_; C = C_; d_a = d_a_; d_u = d_u_; d_f = d_f_; T_L = T_L_;
  gate_L = gate_L_; token_L = token_L_;
  const int T = P + 1;

  e_off.assign(P + 1, 0);
  for (int p = 0; p < P; ++p) e_off[p + 1] = e_off[p] + (size_t)token_L[p] * d_u;
  e.assign(e_off[P], 0.0f);
  e_gate.assign((size_t)gate_L * d_u, 0.0f);

  c1_off.assign(P + 2, 0);
  for (int p = 0; p < P; ++p) c1_off[p + 1] = c1_off[p] + token_L[p];
  c1_off[P + 1] = c1_off[P] + gate_L;

  layers.assign(T_L, Layer{});
  for (auto& L : layers) {
    L.Wq.assign((size_t)H * d_a * d_u, 0.0f);
    L.Wk.assign((size_t)H * d_a * d_u, 0.0f);
    L.Wv.assign((size_t)H * d_u * d_u, 0.0f);
    L.a_q.assign((size_t)H * T * d_a, 0.0f);
    L.a_k.assign((size_t)H * T * d_a, 0.0f);
    L.bA3.assign((size_t)H * T * T, 0.0f);
    if (d_f > 0) {
      L.W1.assign((size_t)d_f * d_u, 0.0f);
      L.W2.assign((size_t)d_u * d_f, 0.0f);
      L.c1.assign(c1_off[P + 1] * (size_t)d_f, 0.0f);
    }
    L.omega3.assign(H, 0.0f);
    L.rho3.assign(H, 1.0f / H);
    L.theta.assign(H, 0.0f);
  }

  WR.assign((size_t)H * d_a * d_u, 0.0f);
  WK.assign((size_t)H * d_a * d_u, 0.0f);
  V_cls.assign((size_t)C * d_u, 0.0f);
  theta_R.assign(H, 0.0f);
}

void Tier3Params::UpdateRho3(int t) {
  simd::softmax(layers[t].omega3.data(), layers[t].rho3.data(), H);
}

// ─────────────────────────── Tier3Work ────────────────────────────

void Tier3Work::Alloc(data_size_t n, int P_, int H_, int C_, int d_u_, int d_f_,
                      int T_L_) {
  N = n; P = P_; H = H_; C = C_; d_u = d_u_; d_f = d_f_; T_L = T_L_;
  T = P + 1;
  u.assign((size_t)(T_L + 1) * N * T * d_u, 0.0f);
  un.assign((size_t)(T_L + 1) * N * T * d_u, 0.0f);
  inv_rms.assign((size_t)(T_L + 1) * N * T, 0.0f);
  if (d_f > 0) {
    unm.assign((size_t)T_L * N * T * d_u, 0.0f);
    inv_rms_m.assign((size_t)T_L * N * T, 0.0f);
    zf.assign((size_t)T_L * N * T * d_f, 0.0f);
    lam_z.assign((size_t)T_L * N * T * d_f, 0.0f);
    mu_z.assign((size_t)T_L * N * T * d_f, 0.0f);
  }
  A3.assign((size_t)T_L * N * H * T * T, 0.0f);
  y.assign((size_t)(T_L + 1) * N * P * C, 0.0f);
  cs.assign((size_t)T_L * N * H * T * T, 0.0f);
  ds.assign((size_t)T_L * N * H * T * T, 0.0f);
  lam_u0.assign((size_t)N * T * d_u, 0.0f);
  mu_u0.assign((size_t)N * T * d_u, 0.0f);
}

// ─────────────────────────── Tier3Stats ────────────────────────────

void Tier3Stats::Alloc(int H, int T, int d_a, int d_u, int d_f, int C, int T_L) {
  layer.assign(T_L, LayerStats{});
  for (auto& L : layer) {
    L.G_Wq.assign((size_t)H * d_a * d_u, 0.0); L.D_Wq = L.G_Wq;
    L.G_Wk = L.G_Wq; L.D_Wk = L.G_Wq;
    L.G_Wv.assign((size_t)H * d_u * d_u, 0.0); L.D_Wv = L.G_Wv;
    L.G_aq.assign((size_t)H * T * d_a, 0.0); L.D_aq = L.G_aq;
    L.G_ak = L.G_aq; L.D_ak = L.G_aq;
    L.G_bA.assign((size_t)H * T * T, 0.0); L.D_bA = L.G_bA;
    L.G_W1.assign((size_t)std::max(d_f, 0) * d_u, 0.0); L.D_W1 = L.G_W1;
    L.G_W2.assign((size_t)d_u * std::max(d_f, 0), 0.0); L.D_W2 = L.G_W2;
    L.G_th.assign(H, 0.0); L.D_th.assign(H, 0.0);
    L.G_om.assign(H, 0.0); L.D_om.assign(H, 0.0);
  }
  G_WR.assign((size_t)H * d_a * d_u, 0.0); D_WR = G_WR;
  G_WK = G_WR; D_WK = G_WR;
  G_thR.assign(H, 0.0); D_thR.assign(H, 0.0);
  G_V.assign((size_t)C * d_u, 0.0);
  A_V.assign((size_t)C * d_u * d_u, 0.0);
  G_b.assign((size_t)H * (T - 1), 0.0); D_b = G_b;
  G_omR.assign(H, 0.0); D_omR.assign(H, 0.0);
}

void Tier3Stats::Zero() {
  auto z = [](std::vector<double>& v) { std::fill(v.begin(), v.end(), 0.0); };
  for (auto& L : layer) {
    z(L.G_Wq); z(L.D_Wq); z(L.G_Wk); z(L.D_Wk); z(L.G_Wv); z(L.D_Wv);
    z(L.G_aq); z(L.D_aq); z(L.G_ak); z(L.D_ak); z(L.G_bA); z(L.D_bA);
    z(L.G_W1); z(L.D_W1); z(L.G_W2); z(L.D_W2);
    z(L.G_th); z(L.D_th); z(L.G_om); z(L.D_om);
    L.G_gc = 0.0; L.D_gc = 0.0;
  }
  z(G_WR); z(D_WR); z(G_WK); z(D_WK); z(G_thR); z(D_thR);
  z(G_V); z(A_V); z(G_b); z(D_b); z(G_omR); z(D_omR);
}

void Tier3Stats::Add(const Tier3Stats& o) {
  auto a = [](std::vector<double>& v, const std::vector<double>& w) {
    for (size_t i = 0; i < v.size(); ++i) v[i] += w[i];
  };
  for (size_t t = 0; t < layer.size(); ++t) {
    auto& L = layer[t]; const auto& O = o.layer[t];
    a(L.G_Wq, O.G_Wq); a(L.D_Wq, O.D_Wq); a(L.G_Wk, O.G_Wk); a(L.D_Wk, O.D_Wk);
    a(L.G_Wv, O.G_Wv); a(L.D_Wv, O.D_Wv);
    a(L.G_aq, O.G_aq); a(L.D_aq, O.D_aq); a(L.G_ak, O.G_ak); a(L.D_ak, O.D_ak);
    a(L.G_bA, O.G_bA); a(L.D_bA, O.D_bA);
    a(L.G_W1, O.G_W1); a(L.D_W1, O.D_W1); a(L.G_W2, O.G_W2); a(L.D_W2, O.D_W2);
    a(L.G_th, O.G_th); a(L.D_th, O.D_th); a(L.G_om, O.G_om); a(L.D_om, O.D_om);
    L.G_gc += O.G_gc; L.D_gc += O.D_gc;
  }
  a(G_WR, o.G_WR); a(D_WR, o.D_WR); a(G_WK, o.G_WK); a(D_WK, o.D_WK);
  a(G_thR, o.G_thR); a(D_thR, o.D_thR); a(G_V, o.G_V); a(A_V, o.A_V);
  a(G_b, o.G_b); a(D_b, o.D_b); a(G_omR, o.G_omR); a(D_omR, o.D_omR);
}

// ─────────────────────────── Tier3Engine ────────────────────────────

Tier3Engine::Tier3Engine(const Config& cfg, data_size_t N, const TokenPlan& plan)
    : cfg_(cfg), N_(N) {
  const int P = cfg.tier == 0 ? 1 : cfg.num_tokens;
  row_mask_.assign(P + 1, 0u);
  const bool masked = cfg.attn_mask != "full" && (int)plan.mask_bits.size() == P;
  const uint32_t all_tokens = (P >= 32) ? ~0u : ((1u << P) - 1u);
  for (int p = 0; p < P; ++p) {
    uint32_t m = masked ? (plan.mask_bits[p] & all_tokens) : all_tokens;
    if (cfg.use_cls_token) m |= (1u << P);   // CLS column always visible
    row_mask_[p] = m;
  }
  // CLS row: attends to everything (incl. itself) when enabled, else inactive.
  row_mask_[P] = cfg.use_cls_token ? (all_tokens | (1u << P)) : 0u;
}

void Tier3Engine::InitParams(Tier3Params& t3p, const BlockParams& params) const {
  // (E3.21) anchored initialization: all Tier-3 params zero. With e ≡ 0 every
  // ũ is 0, so contextual terms, FFN and CLS head vanish and the first forward
  // is numerically the Tier-2 forward from the same Tier-2 parameters.
  t3p.Init(params.P, params.H, params.C, params.d_a, cfg_.d_hidden,
           cfg_.d_ffn, cfg_.attn_layers, params.gate_L, params.token_L);
}

// ─────────────────────────── Forward (E3.1-E3.20) ────────────────────────────

void Tier3Engine::Forward(const BlockParams& params, const Tier3Params& t3p,
                          const std::vector<score_t>& g,
                          const std::vector<score_t>& h,
                          BlockWork& work, Tier3Work& t3w) const {
  const int P = params.P, H = params.H, C = params.C, d_a = params.d_a;
  const int d_u = t3p.d_u, d_f = t3p.d_f, T_L = t3p.T_L, T = P + 1;
  const double eta_a = cfg_.eta_attn, eta_u = cfg_.eta_u, eta_f = cfg_.eta_ffn;
  const double eta_cls = cfg_.eta_cls;
  const double eps_n = cfg_.norm_eps;

  #pragma omp parallel
  {
    std::vector<leaf_t> lp_loc(T);
    std::vector<float> qe((size_t)H * T * d_a), ke((size_t)H * T * d_a);
    std::vector<float> mv((size_t)H * T * d_u);
    std::vector<float> srow(T), arow(T);
    std::vector<double> uhat(d_u), zbuf(std::max(d_f, 1)), fout(d_u);
    std::vector<double> mcur(P), mnext(P);
    std::vector<float> sread(H * P), aread(H * P), lread(H * P);

    #pragma omp for schedule(static)
    for (int cch = 0; cch < kChunks; ++cch) {
      data_size_t ib = ChunkBegin(N_, cch), ie = ChunkBegin(N_, cch + 1);
      for (data_size_t i = ib; i < ie; ++i) {
        for (int tau = 0; tau <= P; ++tau)
          lp_loc[tau] = work.leaf_idx[(size_t)tau * N_ + i];
        const leaf_t lg = lp_loc[P];

        // ── (E3.1) hidden init + carrier init ──
        for (int tau = 0; tau < T; ++tau) {
          const float* src = (tau < P) ? t3p.Ep(tau, lp_loc[tau]) : t3p.Eg(lg);
          std::memcpy(&t3w.u[t3w.UIdx(0, i, tau)], src, sizeof(float) * d_u);
        }
        for (int p = 0; p < P; ++p) {
          const float* vp = params.Vp(p, lp_loc[p]);
          float* y0 = &t3w.y[t3w.YIdx(0, i, p)];
          for (int k = 0; k < C; ++k) y0[k] = vp[k];
        }

        // ── layers t = 0..T_L-1 (E3.2, E3.10-E3.16) ──
        for (int t = 0; t < T_L; ++t) {
          const auto& L = t3p.layers[t];
          // Gated carrier mix: layer 0 = Tier-2-compatible η, deeper layers
          // start closed (γ = 0 at anchored init → carrier identity).
          const double eta_l = (t == 0)
              ? eta_a
              : eta_a * std::min(1.0f, std::max(0.0f, L.gamma_c));
          // RMSNorm of layer input
          for (int tau = 0; tau < T; ++tau)
            t3w.inv_rms[t3w.RmsIdx(t, i, tau)] =
                RmsNormF(&t3w.u[t3w.UIdx(t, i, tau)],
                         &t3w.un[t3w.UIdx(t, i, tau)], d_u, eps_n);

          // Effective q/k (E3.10) + value messages mv = Wv ũ
          for (int h2 = 0; h2 < H; ++h2) {
            const float* Wq = L.Wq.data() + (size_t)h2 * d_a * d_u;
            const float* Wk = L.Wk.data() + (size_t)h2 * d_a * d_u;
            const float* Wv = L.Wv.data() + (size_t)h2 * d_u * d_u;
            for (int tau = 0; tau < T; ++tau) {
              const float* unp = &t3w.un[t3w.UIdx(t, i, tau)];
              const float* aq = (t == 0 && tau < P)
                  ? params.QAp(tau, lp_loc[tau], h2)
                  : L.a_q.data() + ((size_t)h2 * T + tau) * d_a;
              const float* ak = (t == 0 && tau < P)
                  ? params.KAp(tau, lp_loc[tau], h2)
                  : L.a_k.data() + ((size_t)h2 * T + tau) * d_a;
              float* qeo = &qe[((size_t)h2 * T + tau) * d_a];
              float* keo = &ke[((size_t)h2 * T + tau) * d_a];
              for (int d = 0; d < d_a; ++d) {
                double cq = 0.0, ck = 0.0;
                const float* Wqr = Wq + (size_t)d * d_u;
                const float* Wkr = Wk + (size_t)d * d_u;
                for (int j = 0; j < d_u; ++j) {
                  cq += (double)Wqr[j] * unp[j];
                  ck += (double)Wkr[j] * unp[j];
                }
                qeo[d] = aq[d] + (float)cq;
                keo[d] = ak[d] + (float)ck;
              }
              float* mvo = &mv[((size_t)h2 * T + tau) * d_u];
              for (int j = 0; j < d_u; ++j) {
                double s = 0.0;
                const float* Wvr = Wv + (size_t)j * d_u;
                for (int l = 0; l < d_u; ++l) s += (double)Wvr[l] * unp[l];
                mvo[j] = (float)s;
              }
            }
          }

          // Scores + softmax (E3.11)
          for (int h2 = 0; h2 < H; ++h2) {
            const double invs = 1.0 / (std::exp((double)L.theta[h2]) *
                                       std::sqrt((double)d_a));
            for (int tp = 0; tp < T; ++tp) {
              float* Arow = &t3w.A3[t3w.AIdx(t, i, h2, tp, 0)];
              const uint32_t rm = row_mask_[tp];
              if (rm == 0u) { std::fill(Arow, Arow + T, 0.0f); continue; }
              const float* qp = &qe[((size_t)h2 * T + tp) * d_a];
              for (int tr = 0; tr < T; ++tr) {
                if (!((rm >> tr) & 1u)) { srow[tr] = -1e30f; continue; }
                const float* kr = &ke[((size_t)h2 * T + tr) * d_a];
                srow[tr] = (float)(simd::dot(qp, kr, d_a) * invs +
                                   L.bA3[((size_t)h2 * T + tp) * T + tr]);
              }
              simd::softmax(srow.data(), arow.data(), T);
              for (int tr = 0; tr < T; ++tr)
                Arow[tr] = ((rm >> tr) & 1u) ? arow[tr] : 0.0f;
            }
          }

          // Carrier convex mixing (E3.12) with CLS-excluded renormalization
          for (int p = 0; p < P; ++p) {
            const float* yin_base = &t3w.y[t3w.YIdx(t, i, 0)];
            float* yout = &t3w.y[t3w.YIdx(t + 1, i, p)];
            for (int k = 0; k < C; ++k)
              yout[k] = (float)((1.0 - eta_l) * yin_base[(size_t)p * C + k]);
            for (int h2 = 0; h2 < H; ++h2) {
              const float* Arow = &t3w.A3[t3w.AIdx(t, i, h2, p, 0)];
              double Z = 0.0;
              for (int r = 0; r < P; ++r) Z += Arow[r];
              const double rho = L.rho3[h2];
              if (Z <= 1e-30) {
                // Guarded degenerate row: pass-through (Ābar = δ).
                for (int k = 0; k < C; ++k)
                  yout[k] += (float)(eta_l * rho * yin_base[(size_t)p * C + k]);
                continue;
              }
              const double sc = eta_l * rho / Z;
              for (int r = 0; r < P; ++r) {
                const double w = sc * Arow[r];
                if (w == 0.0) continue;
                const float* yr = yin_base + (size_t)r * C;
                for (int k = 0; k < C; ++k) yout[k] += (float)(w * yr[k]);
              }
            }
          }

          // Hidden residual mixing (E3.14)
          for (int tau = 0; tau < T; ++tau) {
            std::fill(uhat.begin(), uhat.end(), 0.0);
            if (row_mask_[tau] != 0u) {
              for (int h2 = 0; h2 < H; ++h2) {
                const float* Arow = &t3w.A3[t3w.AIdx(t, i, h2, tau, 0)];
                const double rho = L.rho3[h2];
                for (int tr = 0; tr < T; ++tr) {
                  const double w = rho * Arow[tr];
                  if (w == 0.0) continue;
                  const float* m = &mv[((size_t)h2 * T + tr) * d_u];
                  for (int j = 0; j < d_u; ++j) uhat[j] += w * m[j];
                }
              }
            }
            const float* uin = &t3w.u[t3w.UIdx(t, i, tau)];
            float* uout = &t3w.u[t3w.UIdx(t + 1, i, tau)];
            for (int j = 0; j < d_u; ++j)
              uout[j] = (float)(uin[j] + eta_u * uhat[j]);

            // Tree-FFN (E3.15-E3.16)
            if (d_f > 0) {
              float* unm = &t3w.unm[t3w.UIdx(t, i, tau)];
              t3w.inv_rms_m[t3w.RmsIdx(t, i, tau)] =
                  RmsNormF(uout, unm, d_u, eps_n);
              const float* c1 = t3p.C1(t, tau, (tau < P) ? lp_loc[tau] : lg);
              float* z = &t3w.zf[t3w.ZIdx(t, i, tau)];
              for (int a = 0; a < d_f; ++a) {
                double s = c1[a];
                const float* W1r = L.W1.data() + (size_t)a * d_u;
                for (int j = 0; j < d_u; ++j) s += (double)W1r[j] * unm[j];
                z[a] = (float)s;
                zbuf[a] = std::tanh(s);
              }
              for (int j = 0; j < d_u; ++j) {
                double s = 0.0;
                const float* W2r = L.W2.data() + (size_t)j * d_f;
                for (int a = 0; a < d_f; ++a) s += (double)W2r[a] * zbuf[a];
                fout[j] = s;
              }
              for (int j = 0; j < d_u; ++j)
                uout[j] = (float)(uout[j] + eta_f * fout[j]);
            }
          }
        }

        // Readout-time RMSNorm (slot T_L)
        for (int tau = 0; tau < T; ++tau)
          t3w.inv_rms[t3w.RmsIdx(T_L, i, tau)] =
              RmsNormF(&t3w.u[t3w.UIdx(T_L, i, tau)],
                       &t3w.un[t3w.UIdx(T_L, i, tau)], d_u, eps_n);

        // ── Deep readout (E3.17) ──
        const float* un_cls = &t3w.un[t3w.UIdx(T_L, i, P)];
        for (int h2 = 0; h2 < H; ++h2) {
          const double invsR = 1.0 / (std::exp((double)t3p.theta_R[h2]) *
                                      std::sqrt((double)d_a));
          float qeff[kMaxDAttn], keff[kMaxDAttn];
          const float* qg = params.Qg(lg, h2);
          const float* WRh = t3p.WR.data() + (size_t)h2 * d_a * d_u;
          for (int d = 0; d < d_a; ++d) {
            double s = qg[d];
            const float* Wr = WRh + (size_t)d * d_u;
            for (int j = 0; j < d_u; ++j) s += (double)Wr[j] * un_cls[j];
            qeff[d] = (float)s;
          }
          const float* WKh = t3p.WK.data() + (size_t)h2 * d_a * d_u;
          for (int p = 0; p < P; ++p) {
            const float* kp = params.Kp(p, lp_loc[p], h2);
            const float* unp = &t3w.un[t3w.UIdx(T_L, i, p)];
            for (int d = 0; d < d_a; ++d) {
              double s = kp[d];
              const float* Wr = WKh + (size_t)d * d_u;
              for (int j = 0; j < d_u; ++j) s += (double)Wr[j] * unp[j];
              keff[d] = (float)s;
            }
            sread[h2 * P + p] = (float)(simd::dot(qeff, keff, d_a) * invsR +
                                        params.b[h2][p]);
          }
          simd::softmax_log(sread.data() + h2 * P, aread.data() + h2 * P,
                            lread.data() + h2 * P, P);
          for (int p = 0; p < P; ++p) {
            work.Alpha(i, h2, p)    = aread[h2 * P + p];
            work.LogAlpha(i, h2, p) = lread[h2 * P + p];
          }
        }
        for (int p = 0; p < P; ++p) {
          float bp = 0.0f;
          for (int h2 = 0; h2 < H; ++h2)
            bp += params.rho[h2] * work.Alpha(i, h2, p);
          work.Beta(i, p) = bp;
        }

        // ── Block output (E3.18) + residual (E.26) ──
        const float* ytop = &t3w.y[t3w.YIdx(T_L, i, 0)];
        for (int k = 0; k < C; ++k) {
          double phi = 0.0;
          for (int p = 0; p < P; ++p)
            phi += (double)work.Beta(i, p) * ytop[(size_t)p * C + k];
          if (eta_cls > 0.0) {
            double cls = 0.0;
            const float* Vk = t3p.V_cls.data() + (size_t)k * d_u;
            for (int j = 0; j < d_u; ++j) cls += (double)Vk[j] * un_cls[j];
            phi += eta_cls * cls;
          }
          work.Phi(i, k) = phi;
          work.R(i, k) = g[(size_t)i * C + k] + h[(size_t)i * C + k] * phi;
        }

        // ── κ (E3.20): m = β, then m ← Ξ^{(t)T} m backwards ──
        for (int p = 0; p < P; ++p) mcur[p] = work.Beta(i, p);
        for (int t = T_L - 1; t >= 0; --t) {
          const auto& L = t3p.layers[t];
          const double eta_l = (t == 0)
              ? eta_a
              : eta_a * std::min(1.0f, std::max(0.0f, L.gamma_c));
          for (int r = 0; r < P; ++r) mnext[r] = (1.0 - eta_l) * mcur[r];
          for (int h2 = 0; h2 < H; ++h2) {
            const double rho = L.rho3[h2];
            for (int p = 0; p < P; ++p) {
              const float* Arow = &t3w.A3[t3w.AIdx(t, i, h2, p, 0)];
              double Z = 0.0;
              for (int r = 0; r < P; ++r) Z += Arow[r];
              if (Z <= 1e-30) {         // guarded pass-through row
                mnext[p] += eta_l * rho * mcur[p];
                continue;
              }
              const double sc = eta_l * rho * mcur[p] / Z;
              for (int r = 0; r < P; ++r) mnext[r] += sc * Arow[r];
            }
          }
          mcur.swap(mnext);
        }
        for (int p = 0; p < P; ++p)
          work.Kappa(i, p) = (float)mcur[p];
      }
    }
  }
}

// ─────────────────────────── EvaluateQ (E.25 + E3.29) ────────────────────────

double Tier3Engine::EvaluateQ(const BlockParams& params, const Tier3Params& t3p,
                              const std::vector<score_t>& g,
                              const std::vector<score_t>& h,
                              const BlockWork& work) const {
  const int C = params.C, H = params.H, P = params.P;
  double partial[kChunks];
  const bool ent = cfg_.lambda_ent > 0;
  #pragma omp parallel for schedule(static)
  for (int c = 0; c < kChunks; ++c) {
    data_size_t s = ChunkBegin(N_, c), e = ChunkBegin(N_, c + 1);
    double q = 0.0;
    for (data_size_t i = s; i < e; ++i) {
      for (int k = 0; k < C; ++k) {
        double phi = work.phi[(size_t)i * C + k];
        q += g[(size_t)i * C + k] * phi + 0.5 * h[(size_t)i * C + k] * phi * phi;
      }
      if (ent) {
        for (int h2 = 0; h2 < H; ++h2)
          for (int p = 0; p < P; ++p)
            q += cfg_.lambda_ent *
                 work.alpha[(size_t)i * H * P + h2 * P + p] *
                 work.log_alpha[(size_t)i * H * P + h2 * P + p];
      }
    }
    partial[c] = q;
  }
  double Q = 0.0;
  for (int c = 0; c < kChunks; ++c) Q += partial[c];

  auto sq = [](const std::vector<float>& v) {
    double s = 0.0;
    for (float x : v) s += (double)x * x;
    return s;
  };
  // Tier-2 subspace L2 (matches the tier<=2 EvaluateQ so the ladder compares
  // like against like).
  Q += 0.5 * cfg_.lambda_v * sq(params.v);
  Q += 0.5 * cfg_.lambda_q * sq(params.z_or_q);
  Q += 0.5 * cfg_.lambda_k * sq(params.k);
  Q += 0.5 * cfg_.lambda_q * sq(params.qA);
  Q += 0.5 * cfg_.lambda_k * sq(params.kA);
  // Ω₃ (E3.29)
  Q += 0.5 * cfg_.lambda_e * (sq(t3p.e) + sq(t3p.e_gate));
  for (const auto& L : t3p.layers) {
    Q += 0.5 * cfg_.lambda_W * (sq(L.Wq) + sq(L.Wk) + sq(L.Wv) +
                                sq(L.W1) + sq(L.W2));
    Q += 0.5 * cfg_.lambda_q * sq(L.a_q);
    Q += 0.5 * cfg_.lambda_k * sq(L.a_k);
    Q += 0.5 * cfg_.lambda_c * sq(L.c1);
    Q += 0.5 * cfg_.lambda_tau * sq(L.theta);
  }
  Q += 0.5 * cfg_.lambda_W * (sq(t3p.WR) + sq(t3p.WK));
  Q += 0.5 * cfg_.lambda_cls * sq(t3p.V_cls);
  Q += 0.5 * cfg_.lambda_tau * sq(t3p.theta_R);
  return Q;
}

// ─────────────────────────── Backward (E3.23-E3.27) ────────────────────────────

void Tier3Engine::Backward(const BlockParams& params, const Tier3Params& t3p,
                           const std::vector<score_t>& g,
                           const std::vector<score_t>& h,
                           BlockWork& work, Tier3Work& t3w,
                           Tier3Stats& stats) const {
  (void)g;
  const int P = params.P, H = params.H, C = params.C, d_a = params.d_a;
  const int d_u = t3p.d_u, d_f = t3p.d_f, T_L = t3p.T_L, T = P + 1;
  const double eta_a = cfg_.eta_attn, eta_u = cfg_.eta_u, eta_f = cfg_.eta_ffn;
  const double eta_cls = cfg_.eta_cls;
  const bool ent = cfg_.lambda_ent > 0;
  const bool tau_learn = cfg_.tau_learnable;

  std::vector<Tier3Stats> chunk_stats(kChunks);
  for (auto& s : chunk_stats) s.Alloc(H, T, d_a, d_u, d_f, C, T_L);

  #pragma omp parallel
  {
    std::vector<leaf_t> lp_loc(T);
    std::vector<double> Ybar((size_t)H * C);
    std::vector<float> qe((size_t)H * T * d_a), ke((size_t)H * T * d_a);
    std::vector<float> mv((size_t)H * T * d_u);
    std::vector<double> wqk((size_t)H * T * d_u), wkq((size_t)H * T * d_u);
    std::vector<double> lam_u((size_t)T * d_u), mu_u((size_t)T * d_u);
    std::vector<double> lam_un((size_t)T * d_u), mu_un((size_t)T * d_u);
    std::vector<double> M(P), Mnext(P);
    std::vector<double> ydot(P);
    std::vector<double> lamA(T), csrow(T);
    std::vector<double> lam_mv(d_u), mu_mv(d_u);
    std::vector<double> tmp_du(d_u);
    std::vector<double> lz(std::max(d_f, 1)), mz(std::max(d_f, 1));
    std::vector<double> qReff((size_t)H * d_a), kReff((size_t)H * P * d_a);
    std::vector<double> msum((size_t)H * T * d_u);

    #pragma omp for schedule(static)
    for (int cch = 0; cch < kChunks; ++cch) {
      Tier3Stats& S = chunk_stats[cch];
      data_size_t ib = ChunkBegin(N_, cch), ie = ChunkBegin(N_, cch + 1);
      for (data_size_t i = ib; i < ie; ++i) {
        for (int tau = 0; tau <= P; ++tau)
          lp_loc[tau] = work.leaf_idx[(size_t)tau * N_ + i];
        const leaf_t lg = lp_loc[P];
        const float* ytop = &t3w.y[t3w.YIdx(T_L, i, 0)];
        const float* un_cls = &t3w.un[t3w.UIdx(T_L, i, P)];

        // ── readout backward: c_read/d_read (E.32-E.34 with Y = y^{(T_L)}) ──
        std::fill(Ybar.begin(), Ybar.end(), 0.0);
        for (int h2 = 0; h2 < H; ++h2)
          for (int p = 0; p < P; ++p)
            for (int k = 0; k < C; ++k)
              Ybar[h2 * C + k] +=
                  (double)work.Alpha(i, h2, p) * ytop[(size_t)p * C + k];
        for (int h2 = 0; h2 < H; ++h2) {
          double S_ent = 0.0;
          if (ent) {
            for (int p = 0; p < P; ++p)
              S_ent += (double)work.Alpha(i, h2, p) *
                       (work.LogAlpha(i, h2, p) + 1.0);
          }
          for (int p = 0; p < P; ++p) {
            double cip = 0.0, dip = 0.0;
            const double alpha_rho =
                (double)params.rho[h2] * work.Alpha(i, h2, p);
            for (int k = 0; k < C; ++k) {
              double dphids =
                  alpha_rho * (ytop[(size_t)p * C + k] - Ybar[h2 * C + k]);
              cip += work.R(i, k) * dphids;
              dip += h[(size_t)i * C + k] * dphids * dphids;
            }
            if (ent)
              cip += cfg_.lambda_ent * work.Alpha(i, h2, p) *
                     (work.LogAlpha(i, h2, p) + 1.0 - S_ent);
            dip += cfg_.eps_damping;
            work.Cread(i, h2, p) = (float)cip;
            work.Dread(i, h2, p) = (float)dip;
            S.G_b[(size_t)h2 * P + p] += cip;
            S.D_b[(size_t)h2 * P + p] += dip;
          }
        }

        // Effective readout q/k (recomputed; forward parity)
        for (int h2 = 0; h2 < H; ++h2) {
          const float* qg = params.Qg(lg, h2);
          const float* WRh = t3p.WR.data() + (size_t)h2 * d_a * d_u;
          for (int d = 0; d < d_a; ++d) {
            double s = qg[d];
            const float* Wr = WRh + (size_t)d * d_u;
            for (int j = 0; j < d_u; ++j) s += (double)Wr[j] * un_cls[j];
            qReff[(size_t)h2 * d_a + d] = s;
          }
          const float* WKh = t3p.WK.data() + (size_t)h2 * d_a * d_u;
          for (int p = 0; p < P; ++p) {
            const float* kp = params.Kp(p, lp_loc[p], h2);
            const float* unp = &t3w.un[t3w.UIdx(T_L, i, p)];
            for (int d = 0; d < d_a; ++d) {
              double s = kp[d];
              const float* Wr = WKh + (size_t)d * d_u;
              for (int j = 0; j < d_u; ++j) s += (double)Wr[j] * unp[j];
              kReff[((size_t)h2 * P + p) * d_a + d] = s;
            }
          }
        }

        // λũ^{(T_L)} / μũ^{(T_L)} injections (E3.24)
        std::fill(lam_un.begin(), lam_un.end(), 0.0);
        std::fill(mu_un.begin(), mu_un.end(), 0.0);
        // CLS head (E3.18/E3.22)
        if (eta_cls > 0.0) {
          for (int k = 0; k < C; ++k) {
            const float* Vk = t3p.V_cls.data() + (size_t)k * d_u;
            const double rk = work.R(i, k), hk = h[(size_t)i * C + k];
            for (int j = 0; j < d_u; ++j) {
              lam_un[(size_t)P * d_u + j] += eta_cls * (double)Vk[j] * rk;
              mu_un[(size_t)P * d_u + j] +=
                  hk * eta_cls * eta_cls * (double)Vk[j] * Vk[j];
              S.G_V[(size_t)k * d_u + j] += rk * eta_cls * un_cls[j];
            }
            double* Av = S.A_V.data() + (size_t)k * d_u * d_u;
            for (int j = 0; j < d_u; ++j)
              for (int l = 0; l < d_u; ++l)
                Av[(size_t)j * d_u + l] += hk * eta_cls * eta_cls *
                                           (double)un_cls[j] * un_cls[l];
          }
        }
        // Readout score paths → WR/WK/θR stats + λũ/μũ
        for (int h2 = 0; h2 < H; ++h2) {
          const double invsR = 1.0 / (std::exp((double)t3p.theta_R[h2]) *
                                      std::sqrt((double)d_a));
          const double* qEf = &qReff[(size_t)h2 * d_a];
          const float* WRh = t3p.WR.data() + (size_t)h2 * d_a * d_u;
          const float* WKh = t3p.WK.data() + (size_t)h2 * d_a * d_u;
          double kacc[kMaxDAttn] = {0};
          for (int p = 0; p < P; ++p) {
            const double* kEf = &kReff[((size_t)h2 * P + p) * d_a];
            const double c = work.Cread(i, h2, p), dd = work.Dread(i, h2, p);
            double sc = 0.0;
            for (int d = 0; d < d_a; ++d) sc += qEf[d] * kEf[d];
            sc *= invsR;  // score minus bias
            if (tau_learn) {
              S.G_thR[h2] += c * (-sc);
              S.D_thR[h2] += dd * sc * sc;
            }
            for (int d = 0; d < d_a; ++d) kacc[d] += c * kEf[d];
            // WK stats (per p)
            const float* unp = &t3w.un[t3w.UIdx(T_L, i, p)];
            double* GWK = S.G_WK.data() + (size_t)h2 * d_a * d_u;
            double* DWK = S.D_WK.data() + (size_t)h2 * d_a * d_u;
            for (int d = 0; d < d_a; ++d) {
              const double cq = c * qEf[d] * invsR;
              const double dq = dd * qEf[d] * qEf[d] * invsR * invsR;
              for (int j = 0; j < d_u; ++j) {
                GWK[(size_t)d * d_u + j] += cq * unp[j];
                DWK[(size_t)d * d_u + j] += dq * (double)unp[j] * unp[j];
              }
            }
            // λũ[p] += invsR * WK^T (c qEf); μũ[p] += invsR² d ((WK^T qEf))²
            for (int j = 0; j < d_u; ++j) {
              double wq = 0.0;
              for (int d = 0; d < d_a; ++d)
                wq += (double)WKh[(size_t)d * d_u + j] * qEf[d];
              lam_un[(size_t)p * d_u + j] += invsR * c * wq;
              mu_un[(size_t)p * d_u + j] += invsR * invsR * dd * wq * wq;
            }
          }
          // WR stats + λũ[CLS]
          double* GWR = S.G_WR.data() + (size_t)h2 * d_a * d_u;
          double* DWR = S.D_WR.data() + (size_t)h2 * d_a * d_u;
          for (int d = 0; d < d_a; ++d)
            for (int j = 0; j < d_u; ++j)
              GWR[(size_t)d * d_u + j] += invsR * kacc[d] * un_cls[j];
          for (int p = 0; p < P; ++p) {
            const double* kEf = &kReff[((size_t)h2 * P + p) * d_a];
            const double dd = work.Dread(i, h2, p);
            for (int d = 0; d < d_a; ++d) {
              const double dk = dd * kEf[d] * kEf[d] * invsR * invsR;
              for (int j = 0; j < d_u; ++j)
                DWR[(size_t)d * d_u + j] += dk * (double)un_cls[j] * un_cls[j];
            }
            for (int j = 0; j < d_u; ++j) {
              double wk = 0.0;
              for (int d = 0; d < d_a; ++d)
                wk += (double)WRh[(size_t)d * d_u + j] * kEf[d];
              lam_un[(size_t)P * d_u + j] +=
                  invsR * work.Cread(i, h2, p) * wk;
              mu_un[(size_t)P * d_u + j] += invsR * invsR * dd * wk * wk;
            }
          }
        }
        // ω (readout heads): carrier formula, Y1 → y^{(T_L)}
        for (int h2 = 0; h2 < H; ++h2) {
          double gr = 0.0, hs = 0.0;
          for (int p = 0; p < P; ++p) {
            const double dbdo = (double)params.rho[h2] *
                (work.Alpha(i, h2, p) - work.Beta(i, p));
            for (int k = 0; k < C; ++k) {
              const double dpb = ytop[(size_t)p * C + k];
              gr += work.R(i, k) * dpb * dbdo;
              hs += h[(size_t)i * C + k] * dpb * dpb * dbdo * dbdo;
            }
          }
          S.G_omR[h2] += gr;
          S.D_omR[h2] += hs;
        }

        // λu^{(T_L)} = Jᵀ λũ ; μu = μũ · inv_rms² (diag approx)
        std::fill(lam_u.begin(), lam_u.end(), 0.0);
        for (int tau = 0; tau < T; ++tau) {
          const float irms = t3w.inv_rms[t3w.RmsIdx(T_L, i, tau)];
          RmsJacTAcc(&t3w.un[t3w.UIdx(T_L, i, tau)], irms,
                     &lam_un[(size_t)tau * d_u], &lam_u[(size_t)tau * d_u], d_u);
          for (int j = 0; j < d_u; ++j)
            mu_u[(size_t)tau * d_u + j] =
                mu_un[(size_t)tau * d_u + j] * (double)irms * irms;
        }

        for (int p = 0; p < P; ++p) M[p] = work.Beta(i, p);

        // ── layers t = T_L-1 .. 0 ──
        for (int t = T_L - 1; t >= 0; --t) {
          const auto& L = t3p.layers[t];
          auto& LS = chunk_stats[cch].layer[t];
          const double eta_l = (t == 0)
              ? eta_a
              : eta_a * std::min(1.0f, std::max(0.0f, L.gamma_c));

          // 1) Tree-FFN backward (E3.27); after this block lam_u/mu_u hold the
          //    adjoint at u_mid.
          if (d_f > 0) {
            for (int tau = 0; tau < T; ++tau) {
              const float* z = &t3w.zf[t3w.ZIdx(t, i, tau)];
              const float* unm = &t3w.unm[t3w.UIdx(t, i, tau)];
              double* lu = &lam_u[(size_t)tau * d_u];
              double* muu = &mu_u[(size_t)tau * d_u];
              // W2 stats + λ_tanh
              for (int a = 0; a < d_f; ++a) {
                const double tz = std::tanh((double)z[a]);
                double lt = 0.0, mt = 0.0;
                for (int j = 0; j < d_u; ++j) {
                  const double W2ja = L.W2[(size_t)j * d_f + a];
                  LS.G_W2[(size_t)j * d_f + a] += lu[j] * eta_f * tz;
                  LS.D_W2[(size_t)j * d_f + a] += muu[j] * eta_f * eta_f * tz * tz;
                  lt += W2ja * lu[j];
                  mt += W2ja * W2ja * muu[j];
                }
                const double dtz = 1.0 - tz * tz;
                lz[a] = eta_f * lt * dtz;
                mz[a] = eta_f * eta_f * mt * dtz * dtz;
                t3w.lam_z[t3w.ZIdx(t, i, tau) + a] = (float)lz[a];
                t3w.mu_z[t3w.ZIdx(t, i, tau) + a] = (float)mz[a];
                for (int j = 0; j < d_u; ++j) {
                  LS.G_W1[(size_t)a * d_u + j] += lz[a] * unm[j];
                  LS.D_W1[(size_t)a * d_u + j] += mz[a] * (double)unm[j] * unm[j];
                }
              }
              // λũm / μũm → λu_mid via FFN-input RMSNorm
              std::fill(tmp_du.begin(), tmp_du.end(), 0.0);
              for (int a = 0; a < d_f; ++a) {
                const float* W1r = L.W1.data() + (size_t)a * d_u;
                for (int j = 0; j < d_u; ++j) tmp_du[j] += (double)W1r[j] * lz[a];
              }
              const float irm = t3w.inv_rms_m[t3w.RmsIdx(t, i, tau)];
              RmsJacTAcc(unm, irm, tmp_du.data(), lu, d_u);
              std::fill(tmp_du.begin(), tmp_du.end(), 0.0);
              for (int a = 0; a < d_f; ++a) {
                const float* W1r = L.W1.data() + (size_t)a * d_u;
                for (int j = 0; j < d_u; ++j)
                  tmp_du[j] += (double)W1r[j] * W1r[j] * mz[a];
              }
              for (int j = 0; j < d_u; ++j)
                muu[j] += tmp_du[j] * (double)irm * irm;
            }
          }

          // Recompute layer activation inputs (q/k eff, mv) from un[t]
          for (int h2 = 0; h2 < H; ++h2) {
            const float* Wq = L.Wq.data() + (size_t)h2 * d_a * d_u;
            const float* Wk = L.Wk.data() + (size_t)h2 * d_a * d_u;
            const float* Wv = L.Wv.data() + (size_t)h2 * d_u * d_u;
            for (int tau = 0; tau < T; ++tau) {
              const float* unp = &t3w.un[t3w.UIdx(t, i, tau)];
              const float* aq = (t == 0 && tau < P)
                  ? params.QAp(tau, lp_loc[tau], h2)
                  : L.a_q.data() + ((size_t)h2 * T + tau) * d_a;
              const float* ak = (t == 0 && tau < P)
                  ? params.KAp(tau, lp_loc[tau], h2)
                  : L.a_k.data() + ((size_t)h2 * T + tau) * d_a;
              float* qeo = &qe[((size_t)h2 * T + tau) * d_a];
              float* keo = &ke[((size_t)h2 * T + tau) * d_a];
              for (int d = 0; d < d_a; ++d) {
                double cq = 0.0, ck = 0.0;
                const float* Wqr = Wq + (size_t)d * d_u;
                const float* Wkr = Wk + (size_t)d * d_u;
                for (int j = 0; j < d_u; ++j) {
                  cq += (double)Wqr[j] * unp[j];
                  ck += (double)Wkr[j] * unp[j];
                }
                qeo[d] = aq[d] + (float)cq;
                keo[d] = ak[d] + (float)ck;
              }
              float* mvo = &mv[((size_t)h2 * T + tau) * d_u];
              for (int j = 0; j < d_u; ++j) {
                double s = 0.0;
                const float* Wvr = Wv + (size_t)j * d_u;
                for (int l = 0; l < d_u; ++l) s += (double)Wvr[l] * unp[l];
                mvo[j] = (float)s;
              }
              // wqk[h][tau] = Wqᵀ k_eff[tau], wkq[h][tau] = Wkᵀ q_eff[tau]
              double* wq = &wqk[((size_t)h2 * T + tau) * d_u];
              double* wk2 = &wkq[((size_t)h2 * T + tau) * d_u];
              for (int j = 0; j < d_u; ++j) { wq[j] = 0.0; wk2[j] = 0.0; }
              for (int d = 0; d < d_a; ++d) {
                const float* Wqr = Wq + (size_t)d * d_u;
                const float* Wkr = Wk + (size_t)d * d_u;
                const double kd = keo[d], qd = qeo[d];
                for (int j = 0; j < d_u; ++j) {
                  wq[j] += (double)Wqr[j] * kd;
                  wk2[j] += (double)Wkr[j] * qd;
                }
              }
            }
          }

          // Per-token r·y dot for the carrier adjoint
          const float* ylev = &t3w.y[t3w.YIdx(t, i, 0)];
          for (int r = 0; r < P; ++r) {
            double s = 0.0;
            for (int k = 0; k < C; ++k)
              s += work.R(i, k) * ylev[(size_t)r * C + k];
            ydot[r] = s;
          }

          std::fill(lam_un.begin(), lam_un.end(), 0.0);
          std::fill(mu_un.begin(), mu_un.end(), 0.0);
          std::fill(Mnext.begin(), Mnext.end(), 0.0);
          for (int r = 0; r < P; ++r) Mnext[r] = (1.0 - eta_l) * M[r];
          std::fill(msum.begin(), msum.end(), 0.0);

          // ω3 carrier accumulators: mixdot[p] = Σ_h ρ ŷĀdot_h[p] and per-k mix
          std::vector<double> yAdot((size_t)H * P, 0.0);
          std::vector<double> yAk((size_t)H * P * C, 0.0);

          for (int h2 = 0; h2 < H; ++h2) {
            const double rho = L.rho3[h2];
            const double invs = 1.0 / (std::exp((double)L.theta[h2]) *
                                       std::sqrt((double)d_a));
            for (int tp = 0; tp < T; ++tp) {
              const uint32_t rm = row_mask_[tp];
              const float* Arow = &t3w.A3[t3w.AIdx(t, i, h2, tp, 0)];
              float* csrow_o = &t3w.cs[t3w.AIdx(t, i, h2, tp, 0)];
              float* dsrow_o = &t3w.ds[t3w.AIdx(t, i, h2, tp, 0)];
              if (rm == 0u) {
                std::fill(csrow_o, csrow_o + T, 0.0f);
                std::fill(dsrow_o, dsrow_o + T, 0.0f);
                continue;
              }
              // hidden path score adjoint (via A softmax over all cols)
              const double* lumid = &lam_u[(size_t)tp * d_u];
              double gdot = 0.0;
              for (int tr = 0; tr < T; ++tr) {
                if (!((rm >> tr) & 1u)) { lamA[tr] = 0.0; continue; }
                const float* m = &mv[((size_t)h2 * T + tr) * d_u];
                double lm = 0.0;
                for (int j = 0; j < d_u; ++j) lm += lumid[j] * m[j];
                lamA[tr] = eta_u * rho * lm;
                gdot += lamA[tr] * Arow[tr];
              }
              for (int tr = 0; tr < T; ++tr)
                csrow[tr] = ((rm >> tr) & 1u)
                    ? (double)Arow[tr] * (lamA[tr] - gdot) : 0.0;

              // carrier path (rows p<P, cols r<P; CLS drops out exactly —
              // dĀ_r/ds_c = Ā_r(δ_rc − Ā_c·1{c<P}), 数学 §8.4)
              double dcar[kMaxTokens + 1] = {0};
              if (tp < P) {
                double Z = 0.0;
                for (int r = 0; r < P; ++r) Z += Arow[r];
                double* yAkr = &yAk[((size_t)h2 * P + tp) * C];
                if (Z > 1e-30) {
                  double yAd = 0.0;
                  for (int k = 0; k < C; ++k) yAkr[k] = 0.0;
                  for (int r = 0; r < P; ++r) {
                    const double ab = Arow[r] / Z;
                    yAd += ab * ydot[r];
                    for (int k = 0; k < C; ++k)
                      yAkr[k] += ab * ylev[(size_t)r * C + k];
                  }
                  yAdot[(size_t)h2 * P + tp] = yAd;
                  const double coef = M[tp] * eta_l * rho;
                  for (int r = 0; r < P; ++r) {
                    const double ab = Arow[r] / Z;
                    csrow[r] += coef * ab * (ydot[r] - yAd);
                    double dsum = 0.0;
                    for (int k = 0; k < C; ++k) {
                      const double dphids =
                          coef * ab * (ylev[(size_t)r * C + k] - yAkr[k]);
                      dsum += h[(size_t)i * C + k] * dphids * dphids;
                    }
                    dcar[r] = dsum;
                    // carrier transpose: Mnext[r] += η ρ Ābar M[p]
                    Mnext[r] += eta_l * rho * ab * M[tp];
                  }
                } else {
                  Mnext[tp] += eta_l * rho * M[tp];  // guarded pass-through
                  // Pass-through semantics for the ω3/γ accumulators too.
                  yAdot[(size_t)h2 * P + tp] = ydot[tp];
                  for (int k = 0; k < C; ++k)
                    yAkr[k] = ylev[(size_t)tp * C + k];
                }
              }

              // store cs/ds; accumulate bA3 / θ / Wq / Wk / a_q stats + λũ (q side)
              double cvq[kMaxDAttn] = {0};
              double dk2[kMaxDAttn] = {0};
              for (int tr = 0; tr < T; ++tr) {
                const double c = csrow[tr];
                const double dd = ((rm >> tr) & 1u)
                    ? dcar[tr] + cfg_.eps_damping : 0.0;
                csrow_o[tr] = (float)c;
                dsrow_o[tr] = (float)dd;
                if (!((rm >> tr) & 1u)) continue;
                LS.G_bA[((size_t)h2 * T + tp) * T + tr] += c;
                LS.D_bA[((size_t)h2 * T + tp) * T + tr] += dd;
                const float* kEf = &ke[((size_t)h2 * T + tr) * d_a];
                if (tau_learn) {
                  const float* qEf = &qe[((size_t)h2 * T + tp) * d_a];
                  double sc = simd::dot(qEf, kEf, d_a) * invs;
                  LS.G_th[h2] += c * (-sc);
                  LS.D_th[h2] += dd * sc * sc;
                }
                for (int d = 0; d < d_a; ++d) {
                  cvq[d] += c * kEf[d];
                  dk2[d] += dd * (double)kEf[d] * kEf[d];
                }
                // λũ[tp] (q side context) via precomputed wqk[tr]
                const double* wq = &wqk[((size_t)h2 * T + tr) * d_u];
                double* lun = &lam_un[(size_t)tp * d_u];
                double* mun = &mu_un[(size_t)tp * d_u];
                for (int j = 0; j < d_u; ++j) {
                  lun[j] += invs * c * wq[j];
                  mun[j] += invs * invs * dd * wq[j] * wq[j];
                }
                // λũ[tr] (k side context, Σ_p included — M1)
                const double* wk2 = &wkq[((size_t)h2 * T + tp) * d_u];
                double* lunr = &lam_un[(size_t)tr * d_u];
                double* munr = &mu_un[(size_t)tr * d_u];
                for (int j = 0; j < d_u; ++j) {
                  lunr[j] += invs * c * wk2[j];
                  munr[j] += invs * invs * dd * wk2[j] * wk2[j];
                }
                // Wk stats (query side fixed = tp): G += c invs q_eff ⊗ ũ_tr
                const float* qEf2 = &qe[((size_t)h2 * T + tp) * d_a];
                const float* untr = &t3w.un[t3w.UIdx(t, i, tr)];
                double* GWk = LS.G_Wk.data() + (size_t)h2 * d_a * d_u;
                double* DWk = LS.D_Wk.data() + (size_t)h2 * d_a * d_u;
                for (int d = 0; d < d_a; ++d) {
                  const double cq = c * invs * qEf2[d];
                  const double dq = dd * invs * invs * (double)qEf2[d] * qEf2[d];
                  for (int j = 0; j < d_u; ++j) {
                    GWk[(size_t)d * d_u + j] += cq * untr[j];
                    DWk[(size_t)d * d_u + j] += dq * (double)untr[j] * untr[j];
                  }
                }
                // a_k anchor stats (rows where the anchor is the global vector)
                if (!(t == 0 && tr < P)) {
                  for (int d = 0; d < d_a; ++d) {
                    LS.G_ak[((size_t)h2 * T + tr) * d_a + d] +=
                        c * invs * qEf2[d];
                    LS.D_ak[((size_t)h2 * T + tr) * d_a + d] +=
                        dd * invs * invs * (double)qEf2[d] * qEf2[d];
                  }
                }
              }
              // Wq stats: G += invs cvq ⊗ ũ_tp; D += invs² dk2 ⊗ ũ_tp²
              const float* untp = &t3w.un[t3w.UIdx(t, i, tp)];
              double* GWq = LS.G_Wq.data() + (size_t)h2 * d_a * d_u;
              double* DWq = LS.D_Wq.data() + (size_t)h2 * d_a * d_u;
              for (int d = 0; d < d_a; ++d) {
                for (int j = 0; j < d_u; ++j) {
                  GWq[(size_t)d * d_u + j] += invs * cvq[d] * untp[j];
                  DWq[(size_t)d * d_u + j] +=
                      invs * invs * dk2[d] * (double)untp[j] * untp[j];
                }
              }
              // a_q anchor stats
              if (!(t == 0 && tp < P)) {
                for (int d = 0; d < d_a; ++d) {
                  LS.G_aq[((size_t)h2 * T + tp) * d_a + d] += invs * cvq[d];
                  LS.D_aq[((size_t)h2 * T + tp) * d_a + d] += invs * invs * dk2[d];
                }
              }
              // msum for ω3 hidden part: msum_h[tp] = η_u Σ_r A m_hr
              double* mso = &msum[((size_t)h2 * T + tp) * d_u];
              for (int tr = 0; tr < T; ++tr) {
                if (!((rm >> tr) & 1u)) continue;
                const float* m = &mv[((size_t)h2 * T + tr) * d_u];
                const double w = eta_u * Arow[tr];
                for (int j = 0; j < d_u; ++j) mso[j] += w * m[j];
              }
            }
          }

          // Value-path backward: λmv, Wv stats, λũ (v side)
          for (int h2 = 0; h2 < H; ++h2) {
            const double rho = L.rho3[h2];
            const float* Wv = L.Wv.data() + (size_t)h2 * d_u * d_u;
            double* GWv = LS.G_Wv.data() + (size_t)h2 * d_u * d_u;
            double* DWv = LS.D_Wv.data() + (size_t)h2 * d_u * d_u;
            for (int tr = 0; tr < T; ++tr) {
              std::fill(lam_mv.begin(), lam_mv.end(), 0.0);
              std::fill(mu_mv.begin(), mu_mv.end(), 0.0);
              for (int tp = 0; tp < T; ++tp) {
                if (!((row_mask_[tp] >> tr) & 1u)) continue;
                const double a = t3w.A3[t3w.AIdx(t, i, h2, tp, tr)];
                if (a == 0.0) continue;
                const double w = eta_u * rho * a;
                const double* lumid = &lam_u[(size_t)tp * d_u];
                const double* mumid = &mu_u[(size_t)tp * d_u];
                for (int j = 0; j < d_u; ++j) {
                  lam_mv[j] += w * lumid[j];
                  mu_mv[j] += w * w * mumid[j];
                }
              }
              const float* untr = &t3w.un[t3w.UIdx(t, i, tr)];
              for (int j = 0; j < d_u; ++j) {
                if (lam_mv[j] == 0.0 && mu_mv[j] == 0.0) continue;
                for (int l = 0; l < d_u; ++l) {
                  GWv[(size_t)j * d_u + l] += lam_mv[j] * untr[l];
                  DWv[(size_t)j * d_u + l] +=
                      mu_mv[j] * (double)untr[l] * untr[l];
                }
              }
              double* lun = &lam_un[(size_t)tr * d_u];
              double* mun = &mu_un[(size_t)tr * d_u];
              MatTVecAcc(Wv, lam_mv.data(), lun, d_u, d_u);
              MatT2VecAcc(Wv, mu_mv.data(), mun, d_u, d_u);
            }
          }

          // ω3 (per-layer heads): carrier exact + hidden 1st-order
          {
            std::vector<double> mixdot(P, 0.0);
            std::vector<double> mixk((size_t)P * C, 0.0);
            for (int p = 0; p < P; ++p)
              for (int h2 = 0; h2 < H; ++h2) {
                mixdot[p] += L.rho3[h2] * yAdot[(size_t)h2 * P + p];
                for (int k = 0; k < C; ++k)
                  mixk[(size_t)p * C + k] +=
                      L.rho3[h2] * yAk[((size_t)h2 * P + p) * C + k];
              }
            // Carrier gate γ (layers t >= 1): ∂φ/∂γ via ∂η_l/∂γ = eta_a
            if (t >= 1) {
              double gr_gc = 0.0;
              for (int p = 0; p < P; ++p)
                gr_gc += M[p] * (mixdot[p] - ydot[p]);
              LS.G_gc += eta_a * gr_gc;
              for (int k = 0; k < C; ++k) {
                double d1 = 0.0;
                for (int p = 0; p < P; ++p)
                  d1 += M[p] * (mixk[(size_t)p * C + k] -
                                ylev[(size_t)p * C + k]);
                d1 *= eta_a;
                LS.D_gc += h[(size_t)i * C + k] * d1 * d1;
              }
            }
            for (int h2 = 0; h2 < H; ++h2) {
              const double rho = L.rho3[h2];
              double gr = 0.0, hs = 0.0;
              for (int p = 0; p < P; ++p) {
                gr += M[p] * eta_l * rho *
                      (yAdot[(size_t)h2 * P + p] - mixdot[p]);
                for (int k = 0; k < C; ++k) {
                  const double d1 = M[p] * eta_l * rho *
                      (yAk[((size_t)h2 * P + p) * C + k] -
                       mixk[(size_t)p * C + k]);
                  hs += h[(size_t)i * C + k] * d1 * d1;
                }
              }
              // hidden 1st-order: ρ_h Σ_τ λumid·(msum_h − Σρ msum)
              for (int tau = 0; tau < T; ++tau) {
                const double* lumid = &lam_u[(size_t)tau * d_u];
                for (int j = 0; j < d_u; ++j) {
                  double msbar = 0.0;
                  for (int h3 = 0; h3 < H; ++h3)
                    msbar += L.rho3[h3] * msum[((size_t)h3 * T + tau) * d_u + j];
                  gr += rho * lumid[j] *
                        (msum[((size_t)h2 * T + tau) * d_u + j] - msbar);
                }
              }
              LS.G_om[h2] += gr;
              LS.D_om[h2] += hs;
            }
          }

          // λu^{(t)} = λu_mid + Jᵀ λũ^{(t)} ; μu likewise (diag)
          for (int tau = 0; tau < T; ++tau) {
            const float irms = t3w.inv_rms[t3w.RmsIdx(t, i, tau)];
            RmsJacTAcc(&t3w.un[t3w.UIdx(t, i, tau)], irms,
                       &lam_un[(size_t)tau * d_u], &lam_u[(size_t)tau * d_u],
                       d_u);
            for (int j = 0; j < d_u; ++j)
              mu_u[(size_t)tau * d_u + j] +=
                  mu_un[(size_t)tau * d_u + j] * (double)irms * irms;
          }
          M.swap(Mnext);
        }

        // e refit inputs (leaf CSR pass consumes these)
        for (int tau = 0; tau < T; ++tau)
          for (int j = 0; j < d_u; ++j) {
            t3w.lam_u0[((size_t)i * T + tau) * d_u + j] =
                (float)lam_u[(size_t)tau * d_u + j];
            t3w.mu_u0[((size_t)i * T + tau) * d_u + j] =
                (float)mu_u[(size_t)tau * d_u + j];
          }
      }
    }
  }

  // Deterministic reduce (chunk order fixed)
  stats.Zero();
  for (int c = 0; c < kChunks; ++c) stats.Add(chunk_stats[c]);
}

// ─────────────────────────── RefitValues (B1, E.31 with κ=E3.20) ─────────────

void Tier3Engine::RefitValues(
    BlockParams& params, BlockWork& work,
    const std::vector<score_t>& g, const std::vector<score_t>& h,
    const std::vector<std::vector<int>>& leaf_begin,
    const std::vector<std::vector<data_size_t>>& sample_ids) const {
  const int P = params.P, C = params.C;
  for (int p = 0; p < P; ++p) {
    const int L = params.token_L[p];
    #pragma omp parallel for schedule(dynamic)
    for (int l = 0; l < L; ++l) {
      const int beg = leaf_begin[p][l], end = leaf_begin[p][l + 1];
      if (beg == end) continue;
      std::vector<float> dv(C);
      float* vp = params.Vp(p, l);
      for (int k = 0; k < C; ++k) {
        double num = cfg_.lambda_v * vp[k];
        double den = cfg_.lambda_v;
        for (int idx = beg; idx < end; ++idx) {
          const data_size_t i = sample_ids[p][idx];
          const double kap = work.Kappa(i, p);
          num += kap * work.R(i, k);
          den += kap * kap * h[(size_t)i * C + k];
        }
        den = std::max(den, cfg_.eps_damping);
        dv[k] = (float)(-num / den);
        vp[k] += dv[k];
      }
      for (int idx = beg; idx < end; ++idx) {
        const data_size_t i = sample_ids[p][idx];
        const double kap = work.Kappa(i, p);
        for (int k = 0; k < C; ++k) {
          work.Phi(i, k) += kap * dv[k];
          work.R(i, k) =
              g[(size_t)i * C + k] + h[(size_t)i * C + k] * work.Phi(i, k);
        }
      }
    }
  }
}

// ─────────────────────────── ApplyUpdates (B2'-B6) ────────────────────────────

void Tier3Engine::ApplyUpdates(BlockParams& params, Tier3Params& t3p,
                               const std::vector<score_t>& g,
                               const std::vector<score_t>& h,
                               const BlockWork& work,
                               const Tier3Work& t3w) const {
  (void)g; (void)h;
  const int P = params.P, H = params.H, C = params.C, d_a = params.d_a;
  const int d_u = t3p.d_u, d_f = t3p.d_f, T_L = t3p.T_L, T = P + 1;
  const double clip = cfg_.attn_step_clip;
  const Tier3Stats& S = stats_;

  // Rebuild the leaf CSR (leaves fixed within the iteration; cheap vs N loops)
  LeafCSR csr;
  csr.Build(work, P, params.gate_L, params.token_L, N_);

  // Diagonal GN for a flat matrix: θ += −(G + λθ)/(D + λ + ε)
  auto diag_gn = [&](std::vector<float>& W, const std::vector<double>& G,
                     const std::vector<double>& D, double lam) {
    for (size_t x = 0; x < W.size(); ++x) {
      const double den = D[x] + lam + cfg_.eps_damping;
      W[x] += (float)ClipStep(-(G[x] + lam * W[x]) / den, clip);
    }
  };

  // ── B2': readout leaf anchors q (gate leaves) / k (token leaves) ──
  // Same normal equations as (E.36)-(E.38) but with EFFECTIVE opposite vectors
  // (leaf anchor + context) and temperature scaling.
  {
    const int gate_L = params.gate_L;
    #pragma omp parallel for schedule(dynamic)
    for (int hl = 0; hl < H * gate_L; ++hl) {
      const int h2 = hl / gate_L, l = hl % gate_L;
      const int beg = csr.leaf_begin[P][l], end = csr.leaf_begin[P][l + 1];
      if (beg == end) continue;
      const double invsR = 1.0 / (std::exp((double)t3p.theta_R[h2]) *
                                  std::sqrt((double)d_a));
      float G[simd::kMaxDa] = {};
      float A[simd::kMaxDa * simd::kMaxDa];
      std::fill(A, A + d_a * d_a, 0.0f);
      for (int d = 0; d < d_a; ++d) A[d * d_a + d] = (float)cfg_.lambda_q;
      const float* WKh = t3p.WK.data() + (size_t)h2 * d_a * d_u;
      float keff[kMaxDAttn];
      for (int idx = beg; idx < end; ++idx) {
        const data_size_t i = csr.sample_ids[P][idx];
        for (int p = 0; p < P; ++p) {
          const leaf_t lpi = work.leaf_idx[(size_t)p * N_ + i];
          const float* kp = params.Kp(p, lpi, h2);
          const float* unp = &t3w.un[t3w.UIdx(T_L, i, p)];
          for (int d = 0; d < d_a; ++d) {
            double s = kp[d];
            const float* Wr = WKh + (size_t)d * d_u;
            for (int j = 0; j < d_u; ++j) s += (double)Wr[j] * unp[j];
            keff[d] = (float)s;
          }
          const float c = work.c_read[(size_t)i * H * P + h2 * P + p];
          const float dd = work.d_read[(size_t)i * H * P + h2 * P + p];
          for (int d = 0; d < d_a; ++d) G[d] += (float)(c * keff[d] * invsR);
          simd::outer_acc(A, (float)(dd * invsR * invsR), keff, keff, d_a);
        }
      }
      float* qg = params.Qg(l, h2);
      for (int d = 0; d < d_a; ++d) G[d] += (float)(cfg_.lambda_q * qg[d]);
      float delta[simd::kMaxDa];
      if (!simd::cholesky_solve(A, G, delta, d_a)) {
        for (int d = 0; d < d_a; ++d)
          delta[d] = -G[d] / std::max(A[d * d_a + d], (float)cfg_.eps_damping);
      } else {
        for (int d = 0; d < d_a; ++d) delta[d] = -delta[d];
      }
      for (int d = 0; d < d_a; ++d) qg[d] += delta[d];
    }

    std::vector<int> tl_off(P + 1, 0);
    for (int p = 0; p < P; ++p) tl_off[p + 1] = tl_off[p] + params.token_L[p];
    const int total_tl = tl_off[P];
    #pragma omp parallel for schedule(dynamic)
    for (int hpl = 0; hpl < H * total_tl; ++hpl) {
      const int h2 = hpl / total_tl;
      int tt = hpl % total_tl, p = 0;
      while (tt >= tl_off[p + 1]) ++p;
      const int l = tt - tl_off[p];
      const int beg = csr.leaf_begin[p][l], end = csr.leaf_begin[p][l + 1];
      if (beg == end) continue;
      const double invsR = 1.0 / (std::exp((double)t3p.theta_R[h2]) *
                                  std::sqrt((double)d_a));
      float G[simd::kMaxDa] = {};
      float A[simd::kMaxDa * simd::kMaxDa];
      std::fill(A, A + d_a * d_a, 0.0f);
      for (int d = 0; d < d_a; ++d) A[d * d_a + d] = (float)cfg_.lambda_k;
      const float* WRh = t3p.WR.data() + (size_t)h2 * d_a * d_u;
      float qeff[kMaxDAttn];
      for (int idx = beg; idx < end; ++idx) {
        const data_size_t i = csr.sample_ids[p][idx];
        const leaf_t lgi = work.leaf_idx[(size_t)P * N_ + i];
        const float* qg = params.Qg(lgi, h2);
        const float* unc = &t3w.un[t3w.UIdx(T_L, i, P)];
        for (int d = 0; d < d_a; ++d) {
          double s = qg[d];
          const float* Wr = WRh + (size_t)d * d_u;
          for (int j = 0; j < d_u; ++j) s += (double)Wr[j] * unc[j];
          qeff[d] = (float)s;
        }
        const float c = work.c_read[(size_t)i * H * P + h2 * P + p];
        const float dd = work.d_read[(size_t)i * H * P + h2 * P + p];
        for (int d = 0; d < d_a; ++d) G[d] += (float)(c * qeff[d] * invsR);
        simd::outer_acc(A, (float)(dd * invsR * invsR), qeff, qeff, d_a);
      }
      float* kp = params.Kp(p, l, h2);
      for (int d = 0; d < d_a; ++d) G[d] += (float)(cfg_.lambda_k * kp[d]);
      float delta[simd::kMaxDa];
      if (!simd::cholesky_solve(A, G, delta, d_a)) {
        for (int d = 0; d < d_a; ++d)
          delta[d] = -G[d] / std::max(A[d * d_a + d], (float)cfg_.eps_damping);
      } else {
        for (int d = 0; d < d_a; ++d) delta[d] = -delta[d];
      }
      for (int d = 0; d < d_a; ++d) kp[d] += delta[d];
    }
  }

  // Readout bias b + head ω + θR + WR/WK + V_cls
  for (int h2 = 0; h2 < H; ++h2)
    for (int p = 0; p < P; ++p) {
      const double den = S.D_b[(size_t)h2 * P + p] + cfg_.eps_damping;
      params.b[h2][p] +=
          (float)ClipStep(-S.G_b[(size_t)h2 * P + p] / den, clip);
    }
  for (int h2 = 0; h2 < H; ++h2) {
    const double den = std::max(S.D_omR[h2], cfg_.eps_damping);
    params.omega[h2] += (float)ClipStep(-S.G_omR[h2] / den, clip);
  }
  params.UpdateRho();
  if (cfg_.tau_learnable) {
    for (int h2 = 0; h2 < H; ++h2) {
      const double num = S.G_thR[h2] + cfg_.lambda_tau * t3p.theta_R[h2];
      const double den = S.D_thR[h2] + cfg_.lambda_tau + cfg_.eps_damping;
      double th = t3p.theta_R[h2] + ClipStep(-num / den, clip);
      t3p.theta_R[h2] = (float)std::max(-kLnTauMax, std::min(kLnTauMax, th));
    }
  }
  diag_gn(t3p.WR, S.G_WR, S.D_WR, cfg_.lambda_W);
  diag_gn(t3p.WK, S.G_WK, S.D_WK, cfg_.lambda_W);
  if (cfg_.eta_cls > 0.0) {
    // (E3.22) exact ridge Newton per class row (Q is quadratic in V_cls)
    for (int k = 0; k < C; ++k) {
      double Am[kMaxDHidden * kMaxDHidden];
      double bvec[kMaxDHidden], x[kMaxDHidden];
      const double* Av = S.A_V.data() + (size_t)k * d_u * d_u;
      float* Vk = t3p.V_cls.data() + (size_t)k * d_u;
      for (int j = 0; j < d_u; ++j) {
        for (int l = 0; l < d_u; ++l) Am[j * d_u + l] = Av[(size_t)j * d_u + l];
        Am[j * d_u + j] += cfg_.lambda_cls + cfg_.eps_damping;
        bvec[j] = S.G_V[(size_t)k * d_u + j] + cfg_.lambda_cls * Vk[j];
      }
      if (CholSolveD(Am, bvec, x, d_u)) {
        for (int j = 0; j < d_u; ++j) Vk[j] -= (float)x[j];
      } else {
        for (int j = 0; j < d_u; ++j)
          Vk[j] += (float)ClipStep(-bvec[j] / Am[j * d_u + j], clip);
      }
    }
  }

  // ── B3: per-layer parameters ──
  for (int t = 0; t < T_L; ++t) {
    auto& L = t3p.layers[t];
    const auto& LS = S.layer[t];
    diag_gn(L.Wq, LS.G_Wq, LS.D_Wq, cfg_.lambda_W);
    diag_gn(L.Wk, LS.G_Wk, LS.D_Wk, cfg_.lambda_W);
    diag_gn(L.Wv, LS.G_Wv, LS.D_Wv, cfg_.lambda_W);
    if (d_f > 0) {
      diag_gn(L.W1, LS.G_W1, LS.D_W1, cfg_.lambda_W);
      diag_gn(L.W2, LS.G_W2, LS.D_W2, cfg_.lambda_W);
    }
    // a_q / a_k global anchors (diag GN; layer 0 rows p<P use leaf anchors)
    for (int h2 = 0; h2 < H; ++h2)
      for (int tau = 0; tau < T; ++tau) {
        if (t == 0 && tau < P) continue;
        for (int d = 0; d < d_a; ++d) {
          size_t x = ((size_t)h2 * T + tau) * d_a + d;
          double den = LS.D_aq[x] + cfg_.lambda_q + cfg_.eps_damping;
          L.a_q[x] += (float)ClipStep(
              -(LS.G_aq[x] + cfg_.lambda_q * L.a_q[x]) / den, clip);
          den = LS.D_ak[x] + cfg_.lambda_k + cfg_.eps_damping;
          L.a_k[x] += (float)ClipStep(
              -(LS.G_ak[x] + cfg_.lambda_k * L.a_k[x]) / den, clip);
        }
      }
    // bA3 (scalar Newton + clip)
    for (size_t x = 0; x < L.bA3.size(); ++x) {
      const double den = LS.D_bA[x] + cfg_.eps_damping;
      L.bA3[x] += (float)ClipStep(-LS.G_bA[x] / den, clip);
    }
    // θ (temperature)
    if (cfg_.tau_learnable) {
      for (int h2 = 0; h2 < H; ++h2) {
        const double num = LS.G_th[h2] + cfg_.lambda_tau * L.theta[h2];
        const double den = LS.D_th[h2] + cfg_.lambda_tau + cfg_.eps_damping;
        double th = L.theta[h2] + ClipStep(-num / den, clip);
        L.theta[h2] = (float)std::max(-kLnTauMax, std::min(kLnTauMax, th));
      }
    }
    // ω3
    for (int h2 = 0; h2 < H; ++h2) {
      const double den = std::max(LS.D_om[h2], cfg_.eps_damping);
      L.omega3[h2] += (float)ClipStep(-LS.G_om[h2] / den, clip);
    }
    t3p.UpdateRho3(t);
    // Carrier gate γ (t >= 1): scalar Newton, projected to [0, 1]
    if (t >= 1) {
      const double den = std::max(LS.D_gc, cfg_.eps_damping);
      double gc = L.gamma_c + ClipStep(-LS.G_gc / den, clip);
      L.gamma_c = (float)std::max(0.0, std::min(1.0, gc));
    }

    // Layer-0 leaf anchors qA/kA: full GN per (h, token, leaf) via CSR, using
    // cs/ds and effective opposite vectors (Tier-3 版 (E.36)-(E.38); the Σ_p
    // for the key side is inherent in cs[p][r]).
    if (t == 0) {
      std::vector<int> tl_off(P + 1, 0);
      for (int p = 0; p < P; ++p) tl_off[p + 1] = tl_off[p] + params.token_L[p];
      const int total_tl = tl_off[P];
      #pragma omp parallel for schedule(dynamic)
      for (int hpl = 0; hpl < 2 * H * total_tl; ++hpl) {
        const bool is_q = hpl < H * total_tl;
        int rem = is_q ? hpl : hpl - H * total_tl;
        const int h2 = rem / total_tl;
        int tt = rem % total_tl, p = 0;
        while (tt >= tl_off[p + 1]) ++p;
        const int l = tt - tl_off[p];
        const int beg = csr.leaf_begin[p][l], end = csr.leaf_begin[p][l + 1];
        if (beg == end) continue;
        const double invs = 1.0 / (std::exp((double)L.theta[h2]) *
                                   std::sqrt((double)d_a));
        const double lam = is_q ? cfg_.lambda_q : cfg_.lambda_k;
        float G[simd::kMaxDa] = {};
        float A[simd::kMaxDa * simd::kMaxDa];
        std::fill(A, A + d_a * d_a, 0.0f);
        for (int d = 0; d < d_a; ++d) A[d * d_a + d] = (float)lam;
        const float* Wq = L.Wq.data() + (size_t)h2 * d_a * d_u;
        const float* Wk = L.Wk.data() + (size_t)h2 * d_a * d_u;
        float opp[kMaxDAttn];
        for (int idx = beg; idx < end; ++idx) {
          const data_size_t i = csr.sample_ids[p][idx];
          for (int tc = 0; tc < T; ++tc) {
            // is_q: row = p, columns tc; else column = p, rows tc
            const int tp2 = is_q ? p : tc;
            const int tr2 = is_q ? tc : p;
            if (!((row_mask_[tp2] >> tr2) & 1u)) continue;
            const float c = t3w.cs[t3w.AIdx(0, i, h2, tp2, tr2)];
            const float dd = t3w.ds[t3w.AIdx(0, i, h2, tp2, tr2)];
            if (c == 0.0f && dd == 0.0f) continue;
            // opposite effective vector
            const int to = tc;
            const leaf_t lo = work.leaf_idx[(size_t)std::min(to, P) * N_ + i];
            const float* anch = (to < P)
                ? (is_q ? params.KAp(to, lo, h2) : params.QAp(to, lo, h2))
                : (is_q ? L.a_k.data() + ((size_t)h2 * T + to) * d_a
                        : L.a_q.data() + ((size_t)h2 * T + to) * d_a);
            const float* uno = &t3w.un[t3w.UIdx(0, i, to)];
            const float* Wo = is_q ? Wk : Wq;
            for (int d = 0; d < d_a; ++d) {
              double s = anch[d];
              const float* Wr = Wo + (size_t)d * d_u;
              for (int j = 0; j < d_u; ++j) s += (double)Wr[j] * uno[j];
              opp[d] = (float)s;
            }
            for (int d = 0; d < d_a; ++d) G[d] += (float)(c * opp[d] * invs);
            simd::outer_acc(A, (float)(dd * invs * invs), opp, opp, d_a);
          }
        }
        float* prm = is_q ? params.QAp(p, l, h2) : params.KAp(p, l, h2);
        for (int d = 0; d < d_a; ++d) G[d] += (float)(lam * prm[d]);
        float delta[simd::kMaxDa];
        if (!simd::cholesky_solve(A, G, delta, d_a)) {
          for (int d = 0; d < d_a; ++d)
            delta[d] = -G[d] / std::max(A[d * d_a + d], (float)cfg_.eps_damping);
        } else {
          for (int d = 0; d < d_a; ++d) delta[d] = -delta[d];
        }
        for (int d = 0; d < d_a; ++d) prm[d] += delta[d];
      }
    }

    // Tree-FFN leaf bias c1 (diag GN per leaf via CSR over lam_z/mu_z)
    if (d_f > 0) {
      #pragma omp parallel for schedule(dynamic)
      for (int tau = 0; tau <= P; ++tau) {
        const int Lc = (tau < P) ? params.token_L[tau] : params.gate_L;
        for (int l = 0; l < Lc; ++l) {
          const int beg = csr.leaf_begin[tau][l], end = csr.leaf_begin[tau][l + 1];
          if (beg == end) continue;
          float* c1 = t3p.C1(t, tau, l);
          for (int a = 0; a < d_f; ++a) {
            double num = cfg_.lambda_c * c1[a], den = cfg_.lambda_c;
            for (int idx = beg; idx < end; ++idx) {
              const data_size_t i = csr.sample_ids[tau][idx];
              num += t3w.lam_z[t3w.ZIdx(t, i, tau) + a];
              den += t3w.mu_z[t3w.ZIdx(t, i, tau) + a];
            }
            den = std::max(den, cfg_.eps_damping);
            c1[a] += (float)ClipStep(-num / den, clip);
          }
        }
      }
    }
  }

  // ── B3d: leaf hidden embeddings e (diag GN per leaf via CSR) ──
  #pragma omp parallel for schedule(dynamic)
  for (int tau = 0; tau <= P; ++tau) {
    const int Lc = (tau < P) ? params.token_L[tau] : params.gate_L;
    for (int l = 0; l < Lc; ++l) {
      const int beg = csr.leaf_begin[tau][l], end = csr.leaf_begin[tau][l + 1];
      if (beg == end) continue;
      float* ep = (tau < P) ? t3p.Ep(tau, l) : t3p.Eg(l);
      for (int j = 0; j < d_u; ++j) {
        double num = cfg_.lambda_e * ep[j], den = cfg_.lambda_e;
        for (int idx = beg; idx < end; ++idx) {
          const data_size_t i = csr.sample_ids[tau][idx];
          num += t3w.lam_u0[((size_t)i * T + tau) * d_u + j];
          den += t3w.mu_u0[((size_t)i * T + tau) * d_u + j];
        }
        den = std::max(den, cfg_.eps_damping);
        ep[j] += (float)ClipStep(-num / den, clip);
      }
    }
  }

  SpectralProject(t3p);
}

// ─────────────────────────── SpectralProject (E3.29 安全弁) ──────────────────

static double SpecNorm(const float* W, int rows, int cols) {
  // 3 rounds of power iteration on WᵀW; enough for a projection cap.
  std::vector<double> v(cols, 1.0 / std::sqrt((double)cols)), wv(rows), u(cols);
  double sigma = 0.0;
  for (int it = 0; it < 3; ++it) {
    for (int r = 0; r < rows; ++r) {
      double s = 0.0;
      for (int c = 0; c < cols; ++c) s += (double)W[(size_t)r * cols + c] * v[c];
      wv[r] = s;
    }
    std::fill(u.begin(), u.end(), 0.0);
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < cols; ++c)
        u[c] += (double)W[(size_t)r * cols + c] * wv[r];
    double n = 0.0;
    for (int c = 0; c < cols; ++c) n += u[c] * u[c];
    n = std::sqrt(n);
    if (n < 1e-30) return 0.0;
    for (int c = 0; c < cols; ++c) v[c] = u[c] / n;
    double wn = 0.0;
    for (int r = 0; r < rows; ++r) wn += wv[r] * wv[r];
    sigma = std::sqrt(wn);
  }
  return sigma;
}

void Tier3Engine::SpectralProject(Tier3Params& t3p) const {
  if (cfg_.spectral_max <= 0.0) return;
  const double smax = cfg_.spectral_max;
  auto cap = [&](std::vector<float>& W, int rows, int cols) {
    if (W.empty()) return;
    const double s = SpecNorm(W.data(), rows, cols);
    if (s > smax) {
      const float sc = (float)(smax / s);
      for (auto& x : W) x *= sc;
    }
  };
  for (auto& L : t3p.layers) {
    for (int h2 = 0; h2 < t3p.H; ++h2) {
      // per-head Wv slab
      if (!L.Wv.empty()) {
        float* Wh = L.Wv.data() + (size_t)h2 * t3p.d_u * t3p.d_u;
        const double s = SpecNorm(Wh, t3p.d_u, t3p.d_u);
        if (s > smax) {
          const float sc = (float)(smax / s);
          for (int x = 0; x < t3p.d_u * t3p.d_u; ++x) Wh[x] *= sc;
        }
      }
    }
    if (t3p.d_f > 0) cap(L.W2, t3p.d_u, t3p.d_f);
  }
}

// ─────────────────────────── PhaseB (§8.2 + ladder E3.28) ────────────────────

// Uniform lerp of the Tier-2 parameter fields: out = saved*(1-a) + cur*a.
static void LerpParams(BlockParams& out, const BlockParams& saved,
                       const BlockParams& cur, double a) {
  auto lf = [&](float s, float c) { return (float)(s * (1.0 - a) + c * a); };
  auto lv = [&](std::vector<float>& o, const std::vector<float>& s,
                const std::vector<float>& c) {
    for (size_t i = 0; i < c.size(); ++i) o[i] = lf(s[i], c[i]);
  };
  lv(out.v, saved.v, cur.v);
  lv(out.z_or_q, saved.z_or_q, cur.z_or_q);
  lv(out.k, saved.k, cur.k);
  lv(out.qA, saved.qA, cur.qA);
  lv(out.kA, saved.kA, cur.kA);
  for (size_t h = 0; h < cur.b.size(); ++h)
    for (size_t p = 0; p < cur.b[h].size(); ++p)
      out.b[h][p] = lf(saved.b[h][p], cur.b[h][p]);
  for (size_t h = 0; h < cur.omega.size(); ++h)
    out.omega[h] = lf(saved.omega[h], cur.omega[h]);
  out.UpdateRho();
}

static void LerpT3(Tier3Params& out, const Tier3Params& saved,
                   const Tier3Params& cur, double a) {
  auto lf = [&](float s, float c) { return (float)(s * (1.0 - a) + c * a); };
  auto lv = [&](std::vector<float>& o, const std::vector<float>& s,
                const std::vector<float>& c) {
    for (size_t i = 0; i < c.size(); ++i) o[i] = lf(s[i], c[i]);
  };
  lv(out.e, saved.e, cur.e);
  lv(out.e_gate, saved.e_gate, cur.e_gate);
  for (size_t t = 0; t < cur.layers.size(); ++t) {
    auto& O = out.layers[t];
    const auto& Sv = saved.layers[t];
    const auto& Cu = cur.layers[t];
    lv(O.Wq, Sv.Wq, Cu.Wq); lv(O.Wk, Sv.Wk, Cu.Wk); lv(O.Wv, Sv.Wv, Cu.Wv);
    lv(O.a_q, Sv.a_q, Cu.a_q); lv(O.a_k, Sv.a_k, Cu.a_k);
    lv(O.bA3, Sv.bA3, Cu.bA3);
    lv(O.W1, Sv.W1, Cu.W1); lv(O.W2, Sv.W2, Cu.W2); lv(O.c1, Sv.c1, Cu.c1);
    lv(O.omega3, Sv.omega3, Cu.omega3);
    lv(O.theta, Sv.theta, Cu.theta);
    O.gamma_c = lf(Sv.gamma_c, Cu.gamma_c);
    out.UpdateRho3((int)t);
  }
  lv(out.WR, saved.WR, cur.WR);
  lv(out.WK, saved.WK, cur.WK);
  lv(out.V_cls, saved.V_cls, cur.V_cls);
  lv(out.theta_R, saved.theta_R, cur.theta_R);
}

double Tier3Engine::PhaseB(BlockParams& params, Tier3Params& t3p,
                           const std::vector<score_t>& g,
                           const std::vector<score_t>& h,
                           BlockWork& work, Tier3Work& t3w) {
  const int P = params.P;
  stats_.Alloc(params.H, P + 1, params.d_a, t3p.d_u, t3p.d_f, params.C,
               t3p.T_L);

  LeafCSR csr;
  csr.Build(work, P, params.gate_L, params.token_L, N_);

  Forward(params, t3p, g, h, work, t3w);
  double Q = EvaluateQ(params, t3p, g, h, work);

  for (int sweep = 0; sweep < cfg_.inner_refit_steps; ++sweep) {
    const double Q_before = Q;

    // (B1) exact — monotone on its own; ladder rollback target is post-B1.
    RefitValues(params, work, g, h, csr.leaf_begin, csr.sample_ids);
    BlockParams snap2 = params;
    Tier3Params snap3 = t3p;

    // Refresh activations (carrier levels changed), backward, apply, refresh.
    Forward(params, t3p, g, h, work, t3w);
    Backward(params, t3p, g, h, work, t3w, stats_);
    ApplyUpdates(params, t3p, g, h, work, t3w);
    Forward(params, t3p, g, h, work, t3w);
    Q = EvaluateQ(params, t3p, g, h, work);
    if (Q <= Q_before + 1e-12) continue;

    // ── Ladder (E3.28) ──
    SHIMAENAGA_LOG_DEBUG("Tier3 ladder: Q %.6g -> %.6g, backtracking",
                         Q_before, Q);
    BlockParams cur2 = params;
    Tier3Params cur3 = t3p;
    bool accepted = false;
    for (int tb = 0; tb < cfg_.max_backtrack; ++tb) {           // 段 1
      const double a = std::pow(cfg_.beta_ls, tb + 1);
      LerpParams(params, snap2, cur2, a);
      LerpT3(t3p, snap3, cur3, a);
      Forward(params, t3p, g, h, work, t3w);
      Q = EvaluateQ(params, t3p, g, h, work);
      if (Q <= Q_before + 1e-12) { accepted = true; break; }
    }
    if (!accepted) {                                            // 段 2
      params = cur2;
      t3p = snap3;
      Forward(params, t3p, g, h, work, t3w);
      Q = EvaluateQ(params, t3p, g, h, work);
      if (Q <= Q_before + 1e-12) {
        accepted = true;
        SHIMAENAGA_LOG_DEBUG("Tier3 ladder: tier-2 subspace step accepted");
      }
    }
    if (!accepted) {                                            // 段 3
      params = snap2;
      t3p = snap3;
      Forward(params, t3p, g, h, work, t3w);
      Q = EvaluateQ(params, t3p, g, h, work);
      SHIMAENAGA_LOG_DEBUG("Tier3 ladder: rollback to post-B1 state");
    }
  }
  return Q;
}

} // namespace shimaenaga
