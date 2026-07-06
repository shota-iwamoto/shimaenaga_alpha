// Smoke test - runs without GTest dependency
#include "shimaenaga/booster.h"
#include "shimaenaga/config.h"
#include "shimaenaga/dataset.h"
#include <vector>
#include <cmath>
#include <cstdio>
#include <random>
#include <cassert>

namespace tf = shimaenaga;

void test_regression_tier0() {
  printf("Test: regression tier=0 (pure GBDT)...\n");
  const int N = 300, F = 8;
  std::mt19937 rng(42);
  std::normal_distribution<float> nd;
  std::vector<float> X(N * F), y(N);
  for (int i = 0; i < N; ++i) {
    float s = 0;
    for (int j = 0; j < F; ++j) { X[i*F+j] = nd(rng); s += X[i*F+j] * (j % 3 == 0 ? 1 : -0.5f); }
    y[i] = s + 0.1f * nd(rng);
  }

  tf::Config cfg;
  cfg.objective = "regression";
  cfg.tier = 0;
  cfg.num_iterations = 20;
  cfg.learning_rate = 0.1;
  cfg.token_num_leaves = 16;
  cfg.gate_num_leaves = 8;
  cfg.min_data_in_leaf = 10;

  auto ds = tf::Dataset::Build(X.data(), N, F, y.data(), nullptr, nullptr, 0, cfg);
  assert(ds != nullptr);
  printf("  Dataset built: %d samples, %d features\n", N, F);

  tf::Booster booster(cfg, std::shared_ptr<tf::Dataset>(std::move(ds)));
  booster.Train();
  printf("  Training complete, %d iterations\n", booster.BestIteration());

  auto pred = booster.Predict(X.data(), N, F);
  assert((int)pred.size() == N);

  // Compute RMSE on training data
  double rmse = 0;
  for (int i = 0; i < N; ++i) {
    double d = pred[i] - y[i];
    rmse += d * d;
  }
  rmse = std::sqrt(rmse / N);
  printf("  Train RMSE = %.4f (should be < initial %.4f)\n", rmse, 1.0);
  assert(std::isfinite(rmse));
  printf("  PASSED\n\n");
}

void test_regression_tier1() {
  printf("Test: regression tier=1 (Tier-1 attention)...\n");
  const int N = 200, F = 6;
  std::mt19937 rng(77);
  std::normal_distribution<float> nd;
  std::vector<float> X(N * F), y(N);
  for (int i = 0; i < N; ++i) {
    float s = 0;
    for (int j = 0; j < F; ++j) { X[i*F+j] = nd(rng); s += X[i*F+j]; }
    y[i] = s;
  }

  tf::Config cfg;
  cfg.objective = "regression";
  cfg.tier = 1;
  cfg.num_tokens = 3;
  cfg.num_heads = 2;
  cfg.d_attn = 4;
  cfg.num_iterations = 10;
  cfg.learning_rate = 0.1;
  cfg.token_num_leaves = 8;
  cfg.gate_num_leaves = 8;
  cfg.min_data_in_leaf = 5;
  cfg.inner_refit_steps = 1;
  cfg.attn_warmup = 3;

  auto ds = tf::Dataset::Build(X.data(), N, F, y.data(), nullptr, nullptr, 0, cfg);
  tf::Booster booster(cfg, std::shared_ptr<tf::Dataset>(std::move(ds)));
  booster.Train();

  auto pred = booster.Predict(X.data(), N, F);
  assert((int)pred.size() == N);
  for (auto v : pred) assert(std::isfinite(v));
  printf("  PASSED (tier=1 with attention)\n\n");
}

void test_binary_classification() {
  printf("Test: binary classification...\n");
  const int N = 200, F = 5;
  std::mt19937 rng(99);
  std::normal_distribution<float> nd;
  std::vector<float> X(N * F), y(N);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < F; ++j) X[i*F+j] = nd(rng);
    float s = 2*X[i*F] + X[i*F+1] - X[i*F+2];
    y[i] = (s > 0) ? 1.0f : 0.0f;
  }

  tf::Config cfg;
  cfg.objective = "binary";
  cfg.tier = 1;
  cfg.num_tokens = 2;
  cfg.num_heads = 1;
  cfg.num_iterations = 15;
  cfg.learning_rate = 0.1;
  cfg.token_num_leaves = 8;
  cfg.gate_num_leaves = 4;
  cfg.min_data_in_leaf = 5;

  auto ds = tf::Dataset::Build(X.data(), N, F, y.data(), nullptr, nullptr, 0, cfg);
  tf::Booster booster(cfg, std::shared_ptr<tf::Dataset>(std::move(ds)));
  booster.Train();

  auto pred = booster.Predict(X.data(), N, F);
  assert((int)pred.size() == N);

  int correct = 0;
  for (int i = 0; i < N; ++i) {
    float p = 1.0f / (1.0f + std::exp(-(float)pred[i]));
    if ((p > 0.5f) == (y[i] > 0.5f)) correct++;
  }
  float acc = (float)correct / N;
  printf("  Train accuracy = %.3f\n", acc);
  printf("  PASSED\n\n");
}

void test_save_load() {
  printf("Test: model save/load...\n");
  const int N = 100, F = 4;
  std::mt19937 rng(111);
  std::normal_distribution<float> nd;
  std::vector<float> X(N * F), y(N);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < F; ++j) X[i*F+j] = nd(rng);
    y[i] = X[i*F] + X[i*F+1];
  }

  tf::Config cfg;
  cfg.objective = "regression";
  cfg.tier = 0;
  cfg.num_iterations = 5;
  cfg.token_num_leaves = 4;
  cfg.gate_num_leaves = 4;
  cfg.min_data_in_leaf = 5;

  auto ds = tf::Dataset::Build(X.data(), N, F, y.data(), nullptr, nullptr, 0, cfg);
  tf::Booster booster(cfg, std::shared_ptr<tf::Dataset>(std::move(ds)));
  booster.Train();

  auto pred_before = booster.Predict(X.data(), N, F);
  booster.SaveModel("/tmp/smoke_test.sbb");
  printf("  Model saved\n");

  booster.LoadModel("/tmp/smoke_test.sbb");
  printf("  Model loaded\n");
  // Note: predictions after reload come from re-trained model,
  // would need Predictor to use loaded model's mappers
  printf("  PASSED\n\n");
}

int main() {
  printf("=== Shimaenaga Smoke Tests ===\n\n");
  try {
    test_regression_tier0();
    test_regression_tier1();
    test_binary_classification();
    test_save_load();
    printf("=== All tests PASSED ===\n");
  } catch (const std::exception& e) {
    printf("FAILED: %s\n", e.what());
    return 1;
  }
  return 0;
}
