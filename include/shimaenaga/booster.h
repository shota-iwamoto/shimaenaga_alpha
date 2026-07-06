#pragma once
#include <memory>
#include <vector>
#include <string>
#include "config.h"
#include "dataset.h"
#include "model.h"
#include "export.h"

namespace shimaenaga {

class SHIMAENAGA_EXPORT Booster {
 public:
  explicit Booster(const Config& cfg, std::shared_ptr<Dataset> train);
  ~Booster();

  void AddValidData(std::shared_ptr<Dataset> valid);

  // Train M iterations (with early stopping if configured)
  void Train();

  // Predict on new data. Returns [n * C] scores.
  std::vector<score_t> Predict(const float* X, data_size_t n, int num_features) const;

  // Predict with attention diagnostics (T6)
  std::vector<score_t> PredictContrib(const float* X, data_size_t n, int num_features,
                                        std::vector<float>* beta_out = nullptr) const;

  void SaveModel(const std::string& path) const;
  void LoadModel(const std::string& path);

  const Model& GetModel() const { return model_; }
  int BestIteration() const { return best_iter_; }

 private:
  Config cfg_;
  std::shared_ptr<Dataset> train_;
  std::shared_ptr<Dataset> valid_;
  Model model_;
  int best_iter_ = 0;

  // Current F scores [N][C]
  std::vector<score_t> F_train_;
  std::vector<score_t> F_valid_;

  // Previous iteration beta [N][P] for Phase A weighting
  std::vector<float> beta_prev_;

  void LogProgress(int iter, double train_metric, double valid_metric) const;

  // Per-block score LUTs for inference (葉ペア分解 M9 applied to Predict).
  // Built lazily on first Predict, invalidated by Train/LoadModel.
  struct BlockLUT {
    std::vector<float> readout;   // [H][P][L_g][L_p_max]
    std::vector<float> selfattn;  // [H][P][P][L_p_max][L_p_max] (may be empty)
    int lp_max = 0;
  };
  struct PredictCache { std::vector<BlockLUT> luts; };
  mutable std::unique_ptr<PredictCache> pred_cache_;

  static void BuildBlockLUT(const AttentiveBlock& blk, BlockLUT& out,
                            bool build_selfattn);
  void EnsurePredictCache() const;

  // Accumulate one block's contribution (learning_rate * phi) into out[0..C-1]
  // given precomputed token leaf indices lp[P] and gate leaf lg. Shared by
  // Predict (raw features) and incremental valid evaluation (binned features).
  // beta_acc (optional, size P): += this block's β_{ip} (attention diagnostics).
  // lut (optional): precomputed score tables; nullptr falls back to QK dots.
  void ApplyBlock(const AttentiveBlock& blk, const leaf_t* lp,
                  leaf_t lg, score_t* out,
                  float* beta_acc = nullptr,
                  const BlockLUT* lut = nullptr) const;

  // Tier-3 block evaluation (T_L > 0): deep layer forward mirroring
  // Tier3Engine::Forward for a single row (E3.1-E3.18). No LUT — contextual
  // scores are row-dependent (Tier-3 数学設計書 §12.1).
  void ApplyBlockT3(const AttentiveBlock& blk, const leaf_t* lp,
                    leaf_t lg, score_t* out, float* beta_acc) const;

  // Incrementally apply newest block to F_valid_ using valid's bin matrix.
  void UpdateValidScores(const AttentiveBlock& blk);

  // Apply configured thread count (num_threads > 0) to OpenMP.
  void ApplyThreadConfig() const;
};

} // namespace shimaenaga
