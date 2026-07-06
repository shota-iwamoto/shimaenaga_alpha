// Tier-3 verification tests (Tier-3 基本設計書 §11 の T12/T14 + 整合性検査)
//   1. anchored parity: Tier3Engine::Forward (e≡0) ≡ tier-2 AttentionEngine
//   2. T14: analytic gradients (Backward stats) vs central differences of Q
//   3. T12: tier=3 with infinite ctx_warmup ≡ tier=2 (bit-identical predict)
//   4. end-to-end: tier=3 trains, improves, batch/single + save/load bit-exact
// Standalone (no GTest). Returns non-zero on failure.
#include "shimaenaga/booster.h"
#include "shimaenaga/config.h"
#include "shimaenaga/dataset.h"
#include "shimaenaga/model.h"
#include "attention/attention_engine.h"
#include "attention/layer_stack.h"
#include "data/token_planner.h"
#include <vector>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <algorithm>

namespace tf = shimaenaga;

static int g_fail = 0;
#define CHECK(cond, msg) do { if(!(cond)){ printf("  [FAIL] %s\n", msg); g_fail++; } else { printf("  [ok]   %s\n", msg);} } while(0)

static void MakeReg(int n, int f, uint64_t seed, std::vector<float>& X, std::vector<float>& y) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  X.resize((size_t)n * f); y.resize(n);
  for (int i = 0; i < n; ++i) {
    float s = 0;
    for (int j = 0; j < f; ++j) {
      X[(size_t)i * f + j] = nd(rng);
      s += X[(size_t)i * f + j] * ((j % 2) ? -0.7f : 1.0f);
    }
    // token-interaction term so the deep path has signal to find
    y[i] = s + 0.5f * X[(size_t)i * f + 0] * X[(size_t)i * f + (f > 1 ? 1 : 0)]
             + 0.05f * nd(rng);
  }
}

// ─── Synthetic engine-level fixture ───
struct EngineFixture {
  tf::Config cfg;
  tf::TokenPlan plan;
  int N, P, H, C, d_a;
  std::vector<int> token_L;
  int gate_L;
  tf::BlockWork work;
  tf::BlockParams params;
  std::vector<tf::score_t> g, h;
  std::mt19937 rng;

  EngineFixture(int n, int p, int hh, int c, int da, int TL, int du, int df,
                uint64_t seed)
      : N(n), P(p), H(hh), C(c), d_a(da), rng(seed) {
    cfg.objective = "regression";
    cfg.tier = 3;
    cfg.num_tokens = P; cfg.num_heads = H; cfg.d_attn = d_a;
    cfg.attn_layers = TL; cfg.d_hidden = du; cfg.d_ffn = df;
    cfg.eta_attn = 0.5; cfg.eta_u = 0.5; cfg.eta_ffn = 0.5; cfg.eta_cls = 0.2;
    cfg.lambda_ent = 1e-3;
    plan.P = P;                       // full mask
    token_L.assign(P, 5);
    gate_L = 4;
    work.Alloc(N, P, H, C, /*tier=*/2);
    // random leaf assignments
    std::uniform_int_distribution<int> uleaf(0, 4), ugate(0, gate_L - 1);
    for (int p2 = 0; p2 < P; ++p2)
      for (int i = 0; i < N; ++i)
        work.LeafIdx(p2, i) = (tf::leaf_t)uleaf(rng);
    for (int i = 0; i < N; ++i) work.LeafIdx(P, i) = (tf::leaf_t)ugate(rng);
    params.Init(P, H, C, d_a, "qk_leaf", gate_L, token_L, /*tier=*/2);
    // random tier-2 params (v, q, k, qA, kA, b); bA stays 0 (= bA3 anchored)
    auto rnd = [&](std::vector<float>& v, float s) {
      std::uniform_real_distribution<float> u(-s, s);
      for (auto& x : v) x = u(rng);
    };
    rnd(params.v, 0.8f); rnd(params.z_or_q, 0.4f); rnd(params.k, 0.4f);
    rnd(params.qA, 0.4f); rnd(params.kA, 0.4f);
    std::uniform_real_distribution<float> ub(-0.2f, 0.2f);
    for (auto& row : params.b) for (auto& x : row) x = ub(rng);
    g.resize((size_t)N * C); h.resize((size_t)N * C);
    std::uniform_real_distribution<double> ug(-1.0, 1.0);
    for (auto& x : g) x = ug(rng);
    for (auto& x : h) x = 0.5 + 0.5 * std::abs(ug(rng));
  }
};

static void test_anchored_parity() {
  printf("Test: anchored parity — Tier-3 (e=0) vs tier-2 forward (数学 §8.1)\n");
  for (int TL : {1, 2}) {
    for (int C : {1, 3}) {
      EngineFixture fx(257, 4, 2, C, 4, TL, 8, 8, 1234 + TL * 10 + C);

      // tier-2 reference forward
      tf::AttentionEngine e2(fx.cfg, fx.N, fx.plan);
      tf::ReadoutLUT rlut; tf::SelfAttnLUT slut;
      e2.BuildReadoutLUT(fx.params, rlut);
      e2.BuildSelfAttnLUT(fx.params, slut);
      tf::BlockWork w2 = fx.work;
      e2.Forward(fx.params, rlut, slut, fx.g, fx.h, w2, false);

      // Tier-3 anchored forward
      tf::Tier3Engine e3(fx.cfg, fx.N, fx.plan);
      tf::Tier3Params t3p;
      e3.InitParams(t3p, fx.params);
      tf::Tier3Work t3w;
      t3w.Alloc(fx.N, fx.P, fx.H, C, fx.cfg.d_hidden, fx.cfg.d_ffn, TL);
      tf::BlockWork w3 = fx.work;
      e3.Forward(fx.params, t3p, fx.g, fx.h, w3, t3w);

      double dphi = 0, dbeta = 0, dkap = 0;
      for (int i = 0; i < fx.N; ++i) {
        for (int k = 0; k < C; ++k)
          dphi = std::max(dphi, std::fabs(w2.Phi(i, k) - w3.Phi(i, k)));
        for (int p = 0; p < fx.P; ++p) {
          dbeta = std::max(dbeta, (double)std::fabs(w2.Beta(i, p) - w3.Beta(i, p)));
          dkap  = std::max(dkap,  (double)std::fabs(w2.Kappa(i, p) - w3.Kappa(i, p)));
        }
      }
      printf("  T_L=%d C=%d: max|Δφ|=%.3g  max|Δβ|=%.3g  max|Δκ|=%.3g\n",
             TL, C, dphi, dbeta, dkap);
      char msg[128];
      snprintf(msg, sizeof(msg), "anchored parity T_L=%d C=%d (φ/β/κ < 1e-4)", TL, C);
      CHECK(dphi < 1e-4 && dbeta < 1e-4 && dkap < 1e-4, msg);
    }
  }
}

// ─── T14: gradient verification ───
struct Probe {
  const char* name;
  float* param;                 // pointer to the perturbed scalar
  double analytic;              // G_data + reg gradient
  bool update_rho = false;      // recompute ρ after perturbation
  tf::Tier3Params* t3p = nullptr;
  tf::BlockParams* bp = nullptr;
  int rho_layer = -1;           // >=0: t3p->UpdateRho3(layer); -2: bp->UpdateRho()
};

static void test_gradients() {
  printf("Test: T14 gradient verification (analytic vs central diff, rel<2e-2)\n");
  const int TL = 2, du = 6, df = 5, C = 1;
  EngineFixture fx(193, 3, 2, C, 3, TL, du, df, 777);
  const tf::Config& cfg = fx.cfg;

  tf::Tier3Engine e3(cfg, fx.N, fx.plan);
  tf::Tier3Params t3p;
  e3.InitParams(t3p, fx.params);
  // Randomize ALL tier-3 params to a generic interior point
  std::mt19937 rng(4242);
  auto rnd = [&](std::vector<float>& v, float s) {
    std::uniform_real_distribution<float> u(-s, s);
    for (auto& x : v) x = u(rng);
  };
  rnd(t3p.e, 0.5f); rnd(t3p.e_gate, 0.5f);
  for (auto& L : t3p.layers) {
    rnd(L.Wq, 0.3f); rnd(L.Wk, 0.3f); rnd(L.Wv, 0.3f);
    rnd(L.a_q, 0.3f); rnd(L.a_k, 0.3f); rnd(L.bA3, 0.2f);
    rnd(L.W1, 0.3f); rnd(L.W2, 0.3f); rnd(L.c1, 0.3f);
    rnd(L.omega3, 0.2f); rnd(L.theta, 0.15f);
    L.gamma_c = 0.35f;
  }
  for (int t = 0; t < TL; ++t) t3p.UpdateRho3(t);
  rnd(t3p.WR, 0.3f); rnd(t3p.WK, 0.3f); rnd(t3p.V_cls, 0.4f);
  rnd(t3p.theta_R, 0.15f);

  tf::Tier3Work t3w;
  t3w.Alloc(fx.N, fx.P, fx.H, C, du, df, TL);
  tf::BlockWork& work = fx.work;
  e3.Forward(fx.params, t3p, fx.g, fx.h, work, t3w);
  tf::Tier3Stats stats;
  stats.Alloc(fx.H, fx.P + 1, fx.d_a, du, df, C, TL);
  e3.Backward(fx.params, t3p, fx.g, fx.h, work, t3w, stats);

  const int T = fx.P + 1;
  const int dadu = fx.d_a * du;

  // Analytic leaf-param gradients from the per-sample adjoints
  auto e_grad = [&](int tau, int leaf, int j) {
    double s = (tau < fx.P) ? cfg.lambda_e * t3p.Ep(tau, leaf)[j]
                            : cfg.lambda_e * t3p.Eg(leaf)[j];
    for (int i = 0; i < fx.N; ++i)
      if (work.LeafIdx(tau, i) == leaf)
        s += t3w.lam_u0[((size_t)i * T + tau) * du + j];
    return s;
  };
  auto c1_grad = [&](int t, int tau, int leaf, int a) {
    double s = cfg.lambda_c * t3p.C1(t, tau, leaf)[a];
    for (int i = 0; i < fx.N; ++i)
      if (work.LeafIdx(tau, i) == leaf)
        s += t3w.lam_z[t3w.ZIdx(t, i, tau) + a];
    return s;
  };

  std::vector<Probe> probes;
  auto& L0 = t3p.layers[0];
  auto& L1 = t3p.layers[1];
  const auto& S0 = stats.layer[0];
  const auto& S1 = stats.layer[1];
  auto idx3 = [&](int h, int d, int j) { return (size_t)h * dadu + d * du + j; };

  probes.push_back({"Wq[l1][h0][1][2]", &L1.Wq[idx3(0, 1, 2)],
                    S1.G_Wq[idx3(0, 1, 2)] + cfg.lambda_W * L1.Wq[idx3(0, 1, 2)]});
  probes.push_back({"Wk[l0][h1][0][3]", &L0.Wk[idx3(1, 0, 3)],
                    S0.G_Wk[idx3(1, 0, 3)] + cfg.lambda_W * L0.Wk[idx3(1, 0, 3)]});
  probes.push_back({"Wv[l0][h0][1][1]", &L0.Wv[(size_t)0 * du * du + 1 * du + 1],
                    S0.G_Wv[(size_t)0 * du * du + 1 * du + 1] +
                        cfg.lambda_W * L0.Wv[(size_t)0 * du * du + 1 * du + 1]});
  probes.push_back({"W1[l1][2][1]", &L1.W1[(size_t)2 * du + 1],
                    S1.G_W1[(size_t)2 * du + 1] + cfg.lambda_W * L1.W1[(size_t)2 * du + 1]});
  probes.push_back({"W2[l0][1][3]", &L0.W2[(size_t)1 * df + 3],
                    S0.G_W2[(size_t)1 * df + 3] + cfg.lambda_W * L0.W2[(size_t)1 * df + 3]});
  probes.push_back({"WR[h0][0][1]", &t3p.WR[idx3(0, 0, 1)],
                    stats.G_WR[idx3(0, 0, 1)] + cfg.lambda_W * t3p.WR[idx3(0, 0, 1)]});
  probes.push_back({"WK[h1][2][0]", &t3p.WK[idx3(1, 2, 0)],
                    stats.G_WK[idx3(1, 2, 0)] + cfg.lambda_W * t3p.WK[idx3(1, 2, 0)]});
  probes.push_back({"V_cls[0][2]", &t3p.V_cls[2],
                    stats.G_V[2] + cfg.lambda_cls * t3p.V_cls[2]});
  probes.push_back({"theta[l0][h1]", &L0.theta[1],
                    S0.G_th[1] + cfg.lambda_tau * L0.theta[1]});
  probes.push_back({"theta_R[h0]", &t3p.theta_R[0],
                    stats.G_thR[0] + cfg.lambda_tau * t3p.theta_R[0]});
  probes.push_back({"gamma_c[l1]", &L1.gamma_c, S1.G_gc});
  probes.push_back({"bA3[l0][h0][1][2]", &L0.bA3[(size_t)0 * T * T + 1 * T + 2],
                    S0.G_bA[(size_t)0 * T * T + 1 * T + 2]});
  probes.push_back({"bA3[l0][h1][cls][0]", &L0.bA3[(size_t)1 * T * T + fx.P * T + 0],
                    S0.G_bA[(size_t)1 * T * T + fx.P * T + 0]});
  probes.push_back({"a_q[l1][h0][tau1][0]", &L1.a_q[((size_t)0 * T + 1) * fx.d_a + 0],
                    S1.G_aq[((size_t)0 * T + 1) * fx.d_a + 0] +
                        cfg.lambda_q * L1.a_q[((size_t)0 * T + 1) * fx.d_a + 0]});
  probes.push_back({"a_k[l0][h0][cls][1]", &L0.a_k[((size_t)0 * T + fx.P) * fx.d_a + 1],
                    S0.G_ak[((size_t)0 * T + fx.P) * fx.d_a + 1] +
                        cfg.lambda_k * L0.a_k[((size_t)0 * T + fx.P) * fx.d_a + 1]});
  probes.push_back({"e[tok1][leaf2][1]", &t3p.Ep(1, 2)[1], e_grad(1, 2, 1)});
  probes.push_back({"e_gate[leaf0][3]", &t3p.Eg(0)[3], e_grad(fx.P, 0, 3)});
  probes.push_back({"c1[l0][tok0][leaf1][2]", &t3p.C1(0, 0, 1)[2], c1_grad(0, 0, 1, 2)});
  probes.push_back({"c1[l1][gate][leaf2][0]", &t3p.C1(1, fx.P, 2)[0],
                    c1_grad(1, fx.P, 2, 0)});
  probes.push_back({"b[h0][p1]", &fx.params.b[0][1], stats.G_b[(size_t)0 * fx.P + 1]});
  {
    Probe pr{"omega3[l0][h1]", &L0.omega3[1], S0.G_om[1]};
    pr.update_rho = true; pr.t3p = &t3p; pr.rho_layer = 0;
    probes.push_back(pr);
  }
  {
    Probe pr{"omega[h0] (readout)", &fx.params.omega[0], stats.G_omR[0]};
    pr.update_rho = true; pr.bp = &fx.params; pr.rho_layer = -2;
    probes.push_back(pr);
  }

  const double eps = 1e-3;
  for (auto& pr : probes) {
    const float orig = *pr.param;
    auto refresh = [&]() {
      if (pr.update_rho) {
        if (pr.rho_layer >= 0) pr.t3p->UpdateRho3(pr.rho_layer);
        else pr.bp->UpdateRho();
      }
    };
    *pr.param = orig + (float)eps; refresh();
    e3.Forward(fx.params, t3p, fx.g, fx.h, work, t3w);
    const double Qp = e3.EvaluateQ(fx.params, t3p, fx.g, fx.h, work);
    *pr.param = orig - (float)eps; refresh();
    e3.Forward(fx.params, t3p, fx.g, fx.h, work, t3w);
    const double Qm = e3.EvaluateQ(fx.params, t3p, fx.g, fx.h, work);
    *pr.param = orig; refresh();
    const double fd = (Qp - Qm) / (2 * eps);
    const double scale = std::max({std::fabs(fd), std::fabs(pr.analytic), 1e-3});
    const double rel = std::fabs(fd - pr.analytic) / scale;
    char msg[160];
    snprintf(msg, sizeof(msg), "%-26s analytic=% .5g  fd=% .5g  rel=%.3g",
             pr.name, pr.analytic, fd, rel);
    CHECK(rel < 2e-2, msg);
  }
  // restore state for hygiene
  e3.Forward(fx.params, t3p, fx.g, fx.h, work, t3w);
}

static void test_t12_degenerate_ctx() {
  printf("Test: T12 — tier=3 with infinite ctx_warmup ≡ tier=2 (bit-identical)\n");
  int N = 300, F = 8;
  std::vector<float> X, y; MakeReg(N, F, 21, X, y);
  auto run = [&](int tier, int ctx) {
    tf::Config cfg;
    cfg.objective = "regression"; cfg.tier = tier; cfg.num_tokens = 4;
    cfg.num_heads = 2; cfg.d_attn = 4; cfg.num_iterations = 14;
    cfg.learning_rate = 0.1; cfg.token_num_leaves = 8; cfg.gate_num_leaves = 8;
    cfg.min_data_in_leaf = 10; cfg.attn_warmup = 3; cfg.seed = 5;
    cfg.ctx_warmup = ctx;
    auto ds = tf::Dataset::Build(X.data(), N, F, y.data(), nullptr, nullptr, 0, cfg);
    tf::Booster b(cfg, std::shared_ptr<tf::Dataset>(std::move(ds)));
    b.Train();
    return b.Predict(X.data(), N, F);
  };
  auto p2 = run(2, -1);
  auto p3 = run(3, 1000000000);
  double md = 0;
  for (int i = 0; i < N; ++i) md = std::max(md, std::fabs(p2[i] - p3[i]));
  printf("  max|tier2 - tier3(ctx=inf)| = %.6g\n", md);
  CHECK(md == 0.0, "T12: degenerate tier-3 bit-identical to tier-2");
}

static void test_end_to_end() {
  printf("Test: tier=3 end-to-end (train/improve/consistency/save-load)\n");
  int N = 500, F = 8;
  std::vector<float> X, y; MakeReg(N, F, 33, X, y);
  tf::Config cfg;
  cfg.objective = "regression"; cfg.tier = 3; cfg.num_tokens = 4;
  cfg.num_heads = 2; cfg.d_attn = 4; cfg.num_iterations = 25;
  cfg.learning_rate = 0.1; cfg.token_num_leaves = 8; cfg.gate_num_leaves = 8;
  cfg.min_data_in_leaf = 10; cfg.attn_warmup = 3; cfg.seed = 7;
  cfg.attn_layers = 2; cfg.d_hidden = 6; cfg.d_ffn = 6; cfg.eta_cls = 0.1;
  auto ds = tf::Dataset::Build(X.data(), N, F, y.data(), nullptr, nullptr, 0, cfg);
  tf::Booster b(cfg, std::shared_ptr<tf::Dataset>(std::move(ds)));
  b.Train();

  // Tier-3 blocks were actually produced (post ctx-warmup)
  int n_t3 = 0;
  for (const auto& blk : b.GetModel().blocks) if (blk.T_L > 0) n_t3++;
  printf("  tier-3 blocks: %d / %zu\n", n_t3, b.GetModel().blocks.size());
  CHECK(n_t3 > 0, "deep blocks present after ctx warm-up");

  auto pred = b.Predict(X.data(), N, F);
  double rmse = 0;
  for (int i = 0; i < N; ++i) rmse += (pred[i] - y[i]) * (pred[i] - y[i]);
  rmse = std::sqrt(rmse / N);
  double yvar = 0, ym = 0;
  for (int i = 0; i < N; ++i) ym += y[i];
  ym /= N;
  for (int i = 0; i < N; ++i) yvar += (y[i] - ym) * (y[i] - ym);
  double ysd = std::sqrt(yvar / N);
  printf("  train RMSE = %.4f (label sd = %.4f)\n", rmse, ysd);
  CHECK(std::isfinite(rmse) && rmse < 0.5 * ysd, "tier-3 fits the training data");

  // batch vs single-row bit equality (I2/T3 for the deep path)
  double md = 0;
  for (int i = 0; i < N; i += 17) {
    auto one = b.Predict(X.data() + (size_t)i * F, 1, F);
    md = std::max(md, std::fabs(one[0] - pred[i]));
  }
  CHECK(md == 0.0, "tier-3 per-row prediction bit-identical to batch");

  // T17: save → load → predict bit equality (format v4 roundtrip)
  b.SaveModel("/tmp/t03_tier3.sbb");
  b.LoadModel("/tmp/t03_tier3.sbb");
  auto pred2 = b.Predict(X.data(), N, F);
  double md2 = 0;
  for (int i = 0; i < N; ++i) md2 = std::max(md2, std::fabs(pred2[i] - pred[i]));
  printf("  max|pred(save/load) - pred| = %.6g\n", md2);
  CHECK(md2 == 0.0, "T17: v4 serialization roundtrip bit-identical");

  // attention diagnostics still a probability vector (T6 continuity)
  std::vector<float> beta;
  b.PredictContrib(X.data(), 32, F, &beta);
  double worst = 0;
  for (int i = 0; i < 32; ++i) {
    double s = 0;
    for (int p = 0; p < cfg.num_tokens; ++p) {
      double v = beta[(size_t)i * cfg.num_tokens + p];
      if (v < -1e-6) worst = 1e9;
      s += v;
    }
    worst = std::max(worst, std::fabs(s - 1.0));
  }
  CHECK(worst < 1e-4, "β diagnostics rows sum to 1 (convex)");
}

static void test_multiclass_and_binary() {
  printf("Test: tier=3 binary / multiclass smoke\n");
  int N = 400, F = 6;
  std::vector<float> X, y; MakeReg(N, F, 55, X, y);
  {
    std::vector<float> yb(N);
    for (int i = 0; i < N; ++i) yb[i] = y[i] > 0 ? 1.0f : 0.0f;
    tf::Config cfg;
    cfg.objective = "binary"; cfg.tier = 3; cfg.num_tokens = 3;
    cfg.num_heads = 2; cfg.num_iterations = 15; cfg.learning_rate = 0.1;
    cfg.token_num_leaves = 8; cfg.gate_num_leaves = 8; cfg.min_data_in_leaf = 10;
    cfg.attn_warmup = 3; cfg.attn_layers = 1; cfg.d_hidden = 4; cfg.d_ffn = 4;
    auto ds = tf::Dataset::Build(X.data(), N, F, yb.data(), nullptr, nullptr, 0, cfg);
    tf::Booster b(cfg, std::shared_ptr<tf::Dataset>(std::move(ds)));
    b.Train();
    auto pred = b.Predict(X.data(), N, F);
    int correct = 0;
    for (int i = 0; i < N; ++i)
      if ((pred[i] > 0) == (yb[i] > 0.5f)) correct++;
    printf("  binary train acc = %.3f\n", (double)correct / N);
    CHECK(correct > N * 0.8, "tier-3 binary learns");
  }
  {
    std::vector<float> ym(N);
    for (int i = 0; i < N; ++i) ym[i] = (float)(y[i] < -1 ? 0 : (y[i] < 1 ? 1 : 2));
    tf::Config cfg;
    cfg.objective = "multiclass"; cfg.num_class = 3; cfg.tier = 3;
    cfg.num_tokens = 3; cfg.num_heads = 2; cfg.num_iterations = 15;
    cfg.learning_rate = 0.1; cfg.token_num_leaves = 8; cfg.gate_num_leaves = 8;
    cfg.min_data_in_leaf = 10; cfg.attn_warmup = 3;
    cfg.attn_layers = 2; cfg.d_hidden = 4; cfg.d_ffn = 0;   // FFN off branch
    auto ds = tf::Dataset::Build(X.data(), N, F, ym.data(), nullptr, nullptr, 0, cfg);
    tf::Booster b(cfg, std::shared_ptr<tf::Dataset>(std::move(ds)));
    b.Train();
    auto pred = b.Predict(X.data(), N, F);
    int correct = 0;
    for (int i = 0; i < N; ++i) {
      int am = 0;
      for (int k = 1; k < 3; ++k)
        if (pred[(size_t)i * 3 + k] > pred[(size_t)i * 3 + am]) am = k;
      if (am == (int)ym[i]) correct++;
    }
    printf("  multiclass train acc = %.3f\n", (double)correct / N);
    CHECK(correct > N * 0.7, "tier-3 multiclass (d_ffn=0) learns");
  }
}

int main() {
  printf("=== Tier-3 tests (t03) ===\n");
  test_anchored_parity();
  test_gradients();
  test_t12_degenerate_ctx();
  test_end_to_end();
  test_multiclass_and_binary();
  if (g_fail == 0) printf("ALL PASS\n");
  else printf("%d FAILURE(S)\n", g_fail);
  return g_fail == 0 ? 0 : 1;
}
