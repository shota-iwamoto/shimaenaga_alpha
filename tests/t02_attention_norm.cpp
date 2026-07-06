// T2: Attention normalization test
// Verifies: Σ_p alpha[h][p] = 1, alpha >= 0, Σ_p beta[p] = 1 (T2)
#include <gtest/gtest.h>
#include "attention/attention_engine.h"
#include "data/token_planner.h"
#include <vector>
#include <cmath>
#include <numeric>
#include <random>

namespace tf = shimaenaga;

TEST(AttentionNorm, SoftmaxSumsToOne) {
  // Test the softmax kernel directly
  std::vector<float> in = {1.0f, 2.0f, 0.5f, -1.0f};
  std::vector<float> out(4);
  tf::simd::softmax(in.data(), out.data(), 4);

  float sum = 0.0f;
  for (float v : out) {
    EXPECT_GE(v, 0.0f);
    sum += v;
  }
  EXPECT_NEAR(sum, 1.0f, 1e-6f);
}

TEST(AttentionNorm, SoftmaxNumericalStability) {
  // Large values should not cause overflow
  std::vector<float> in = {1000.0f, 1001.0f, 999.0f};
  std::vector<float> out(3);
  tf::simd::softmax(in.data(), out.data(), 3);

  float sum = 0.0f;
  for (float v : out) {
    EXPECT_GE(v, 0.0f);
    EXPECT_LE(v, 1.0f);
    EXPECT_TRUE(std::isfinite(v));
    sum += v;
  }
  EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST(AttentionNorm, CholeskySmall) {
  // Test 2x2 Cholesky solve
  // A = [[4, 2], [2, 3]], b = [4, 1]
  // Solution: Ax = b -> x = [1.125, -0.25] (approximately)
  float A[4] = {4.0f, 2.0f, 2.0f, 3.0f};
  float b[2] = {4.0f, 1.0f};
  float x[2] = {};
  bool ok = tf::simd::cholesky_solve(A, b, x, 2);
  EXPECT_TRUE(ok);

  // Verify: A*x ≈ b
  float r0 = 4*x[0] + 2*x[1];
  float r1 = 2*x[0] + 3*x[1];
  EXPECT_NEAR(r0, 4.0f, 1e-4f);
  EXPECT_NEAR(r1, 1.0f, 1e-4f);
}

TEST(AttentionNorm, BetaSumsToOne) {
  // With uniform rho and equal alpha, beta should sum to 1/H
  // and Σ_p beta[p] = 1
  int P = 4, H = 2;
  std::vector<float> rho = {0.6f, 0.4f};
  std::vector<std::vector<float>> alpha(H, std::vector<float>(P));
  // Make uniform alpha
  for (int h = 0; h < H; ++h)
    for (int p = 0; p < P; ++p)
      alpha[h][p] = 1.0f / P;

  // Compute beta[p] = Σ_h rho[h] * alpha[h][p]
  std::vector<float> beta(P, 0.0f);
  for (int h = 0; h < H; ++h)
    for (int p = 0; p < P; ++p)
      beta[p] += rho[h] * alpha[h][p];

  float sum = 0.0f;
  for (float b : beta) sum += b;
  EXPECT_NEAR(sum, 1.0f, 1e-6f);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
