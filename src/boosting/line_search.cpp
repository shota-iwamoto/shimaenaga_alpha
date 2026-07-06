#include "line_search.h"
#include "../util/log.h"
#include <cmath>

namespace shimaenaga {

LineSearch::LineSearch(const Config& cfg, Refitter& refitter, BlockParams& params,
                       BlockWork& work, AttentionEngine& engine,
                       ReadoutLUT& rlut, SelfAttnLUT& slut,
                       const std::vector<score_t>& g, const std::vector<score_t>& h)
    : cfg_(cfg), refitter_(refitter), params_(params), work_(work),
      engine_(engine), rlut_(rlut), slut_(slut), g_(g), h_(h) {}

void LineSearch::SaveState() {
  saved_params_ = params_;
}

// Interpolate every parameter field: out = saved*(1-a) + cur*a. Scaling the
// whole sweep's Δθ uniformly (§7.6) requires touching ALL updated parameters —
// v, z_or_q, k, qA, kA, b, bA, omega — not just v and q.
static void LerpInto(BlockParams& out, const BlockParams& saved,
                     const BlockParams& cur, double a) {
  auto lf = [&](float s, float c) { return static_cast<float>(s * (1.0 - a) + c * a); };
  auto lerp_vec = [&](std::vector<float>& o, const std::vector<float>& s,
                      const std::vector<float>& c) {
    for (size_t i = 0; i < c.size(); ++i) o[i] = lf(s[i], c[i]);
  };
  // Flat leaf-indexed tensors.
  lerp_vec(out.v,      saved.v,      cur.v);
  lerp_vec(out.z_or_q, saved.z_or_q, cur.z_or_q);
  lerp_vec(out.k,      saved.k,      cur.k);
  lerp_vec(out.qA,     saved.qA,     cur.qA);   // empty for tier<2 (no-op)
  lerp_vec(out.kA,     saved.kA,     cur.kA);
  // Small dense params.
  for (size_t h = 0; h < cur.b.size(); ++h)
    for (size_t p = 0; p < cur.b[h].size(); ++p)
      out.b[h][p] = lf(saved.b[h][p], cur.b[h][p]);
  for (size_t h = 0; h < cur.omega.size(); ++h)
    out.omega[h] = lf(saved.omega[h], cur.omega[h]);
  for (size_t h = 0; h < cur.bA.size(); ++h)
    for (size_t p = 0; p < cur.bA[h].size(); ++p)
      for (size_t r = 0; r < cur.bA[h][p].size(); ++r)
        out.bA[h][p][r] = lf(saved.bA[h][p][r], cur.bA[h][p][r]);
  for (size_t h = 0; h < cur.omegaA.size(); ++h)
    out.omegaA[h] = lf(saved.omegaA[h], cur.omegaA[h]);
  out.UpdateRho();
  out.UpdateRhoA();
}

bool LineSearch::Check(double Q_prev) {
  double Q_now = refitter_.EvaluateQ();
  if (Q_now <= Q_prev + 1e-12) return true;

  SHIMAENAGA_LOG_DEBUG("LineSearch: Q increased %.6g -> %.6g, backtracking", Q_prev, Q_now);

  BlockParams cur = params_;  // full post-sweep parameters
  // Trial evaluations only need what EvaluateQ reads (α for the entropy term,
  // A/y1, φ). The backward tensors and κ are recomputed once on acceptance —
  // computing them per rejected trial was pure waste.
  ForwardOpts lean;
  lean.alpha = true; lean.selfattn = true;
  lean.kappa = false; lean.back_read = false; lean.back_self = false;
  for (int t = 0; t < cfg_.max_backtrack; ++t) {
    double alpha = std::pow(cfg_.beta_ls, t + 1);
    LerpInto(params_, saved_params_, cur, alpha);

    // Apply and re-forward (lean: enough for EvaluateQ)
    engine_.BuildReadoutLUT(params_, rlut_);
    if (cfg_.tier >= 2) engine_.BuildSelfAttnLUT(params_, slut_);
    engine_.Forward(params_, rlut_, slut_, g_, h_, work_, false, lean);

    double Q_new = refitter_.EvaluateQ();
    if (Q_new <= Q_prev + 1e-12) {
      SHIMAENAGA_LOG_DEBUG("LineSearch: backtrack %d steps, Q=%.6g", t + 1, Q_new);
      // Full forward so κ and the backward tensors match the accepted params.
      engine_.Forward(params_, rlut_, slut_, g_, h_, work_);
      return true;
    }
  }

  // Rollback to the saved post-(B1) state (§7.6): B1's monotone progress is
  // kept, only the B2-B4 updates of this sweep are discarded.
  SHIMAENAGA_LOG_DEBUG("LineSearch: rollback to post-B1 state");
  params_ = saved_params_;
  engine_.BuildReadoutLUT(params_, rlut_);
  if (cfg_.tier >= 2) engine_.BuildSelfAttnLUT(params_, slut_);
  // Full re-forward so every derived quantity (α, β, κ, c/d, φ, r) is consistent
  // with the restored parameters before the next sweep begins.
  engine_.Forward(params_, rlut_, slut_, g_, h_, work_);
  return false;
}

} // namespace shimaenaga
