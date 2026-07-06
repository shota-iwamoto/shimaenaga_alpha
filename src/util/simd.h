#pragma once
#include <cmath>
#include <algorithm>

// SIMD abstraction. AVX2 on x86-64, NEON on ARM64, scalar otherwise.
// For correctness, scalar path is always valid.
#ifdef SHIMAENAGA_AVX2
  #include <immintrin.h>
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  #define SHIMAENAGA_NEON 1
  #include <arm_neon.h>
#endif

namespace shimaenaga {
namespace simd {

// Numerically stable softmax for P <= 16 values
// in[0..P-1] -> out[0..P-1] (each row)
inline void softmax(const float* in, float* out, int P) {
  float mx = in[0];
  for (int p = 1; p < P; ++p) mx = std::max(mx, in[p]);
  float sum = 0.0f;
  for (int p = 0; p < P; ++p) {
    out[p] = std::exp(in[p] - mx);
    sum += out[p];
  }
  float inv = 1.0f / (sum + 1e-15f);
  for (int p = 0; p < P; ++p) out[p] *= inv;
}

// Softmax that also emits log-probabilities: logα_p = (s_p − mx) − log(Σ exp).
// One log() per row instead of P per consumer — the entropy regularizer
// (E.35/E.41) reads cached logα so it never calls log itself.
inline void softmax_log(const float* in, float* out, float* log_out, int P) {
  float mx = in[0];
  for (int p = 1; p < P; ++p) mx = std::max(mx, in[p]);
  float sum = 0.0f;
  for (int p = 0; p < P; ++p) {
    out[p] = std::exp(in[p] - mx);
    sum += out[p];
  }
  float ls  = std::log(sum + 1e-15f);
  float inv = 1.0f / (sum + 1e-15f);
  for (int p = 0; p < P; ++p) {
    log_out[p] = in[p] - mx - ls;
    out[p] *= inv;
  }
}

// dot product of two float vectors of length d
inline float dot(const float* a, const float* b, int d) {
#ifdef SHIMAENAGA_AVX2
  if (d >= 8) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= d; i += 8)
      acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
    // Horizontal sum of acc.
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float s = _mm_cvtss_f32(lo);
    for (; i < d; ++i) s += a[i] * b[i];  // scalar tail
    return s;
  }
#endif
#ifdef SHIMAENAGA_NEON
  if (d >= 4) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= d; i += 4)
      acc = vfmaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
    float s = vaddvq_f32(acc);
    for (; i < d; ++i) s += a[i] * b[i];  // scalar tail
    return s;
  }
#endif
  float s = 0.0f;
  for (int i = 0; i < d; ++i) s += a[i] * b[i];
  return s;
}

// outer product accumulation: A += scale * a * b^T  (a,b in R^d)
// Each row i becomes A[i,:] += (scale*a[i]) * b[:]; the per-row scalar is
// broadcast and FMA'd across the row. This is the innermost op of the
// Gauss-Newton A-matrix build (RefitReadout/RefitSelfAttn), so it is the
// dominant cost of Phase B for qk_leaf / tier-2.
inline void outer_acc(float* A, float scale, const float* a, const float* b, int d) {
#ifdef SHIMAENAGA_AVX2
  if (d == 8) {
    __m256 bv = _mm256_loadu_ps(b);
    for (int i = 0; i < 8; ++i) {
      __m256 s = _mm256_set1_ps(scale * a[i]);
      float* row = A + i * 8;
      _mm256_storeu_ps(row, _mm256_fmadd_ps(s, bv, _mm256_loadu_ps(row)));
    }
    return;
  }
  if (d >= 8) {
    for (int i = 0; i < d; ++i) {
      float si = scale * a[i];
      __m256 s = _mm256_set1_ps(si);
      float* row = A + i * d;
      int j = 0;
      for (; j + 8 <= d; j += 8)
        _mm256_storeu_ps(row + j,
            _mm256_fmadd_ps(s, _mm256_loadu_ps(b + j), _mm256_loadu_ps(row + j)));
      for (; j < d; ++j) row[j] += si * b[j];
    }
    return;
  }
#endif
#ifdef SHIMAENAGA_NEON
  if (d >= 4) {
    for (int i = 0; i < d; ++i) {
      float si = scale * a[i];
      float32x4_t s = vdupq_n_f32(si);
      float* row = A + i * d;
      int j = 0;
      for (; j + 4 <= d; j += 4)
        vst1q_f32(row + j, vfmaq_f32(vld1q_f32(row + j), s, vld1q_f32(b + j)));
      for (; j < d; ++j) row[j] += si * b[j];
    }
    return;
  }
#endif
  for (int i = 0; i < d; ++i) {
    float si = scale * a[i];
    for (int j = 0; j < d; ++j)
      A[i * d + j] += si * b[j];
  }
}

// Max attention dimension d_a supported by the fixed-size solver buffers.
// Shared with the Phase-B refit kernels so their stack scratch matches.
static constexpr int kMaxDa = 32;

// Solve d×d symmetric positive definite system Ax=b via Cholesky (d<=kMaxDa)
// Returns false if factorization fails (falls back to diagonal Newton).
// No pre-zeroing: the factorization only ever reads lower-triangular entries
// it has already written, so leaving L uninitialised is safe and avoids a
// per-call memset in the hottest Phase-B inner loop.
inline bool cholesky_solve(const float* A, const float* b, float* x, int d) {
  float L[kMaxDa * kMaxDa];  // L[i*d+j], lower triangular
  for (int i = 0; i < d; ++i) {
    for (int j = 0; j <= i; ++j) {
      float s = A[i * d + j];
      for (int k = 0; k < j; ++k) s -= L[i * d + k] * L[j * d + k];
      if (i == j) {
        if (s <= 0.0f) return false;
        L[i * d + i] = std::sqrt(s);
      } else {
        L[i * d + j] = s / L[j * d + j];
      }
    }
  }
  // Forward substitution: L y = b
  float y[kMaxDa];
  for (int i = 0; i < d; ++i) {
    float s = b[i];
    for (int k = 0; k < i; ++k) s -= L[i * d + k] * y[k];
    y[i] = s / L[i * d + i];
  }
  // Backward substitution: L^T x = y
  for (int i = d - 1; i >= 0; --i) {
    float s = y[i];
    for (int k = i + 1; k < d; ++k) s -= L[k * d + i] * x[k];
    x[i] = s / L[i * d + i];
  }
  return true;
}

} // namespace simd
} // namespace shimaenaga
