#pragma once
#include <cstdlib>
#include <cstddef>
#include <new>
#include <vector>

namespace shimaenaga {

// 64-byte aligned allocation for SIMD and cache efficiency
inline void* AlignedMalloc(size_t bytes, size_t align = 64) {
  if (bytes == 0) return nullptr;
#ifdef _WIN32
  void* p = _aligned_malloc(bytes, align);
#else
  void* p = nullptr;
  if (posix_memalign(&p, align, bytes) != 0) p = nullptr;
#endif
  if (!p) throw std::bad_alloc();
  return p;
}

inline void AlignedFree(void* p) {
#ifdef _WIN32
  _aligned_free(p);
#else
  free(p);
#endif
}

// Aligned vector using std::vector with aligned allocator
template <typename T, size_t Align = 64>
class AlignedVec {
 public:
  AlignedVec() = default;
  explicit AlignedVec(size_t n, T val = T{}) { resize(n, val); }

  void resize(size_t n, T val = T{}) {
    data_.resize(n, val);
  }
  void assign(size_t n, T val) { data_.assign(n, val); }
  size_t size() const { return data_.size(); }
  bool   empty() const { return data_.empty(); }
  void   clear() { data_.clear(); }

  T*       data()       { return data_.data(); }
  const T* data() const { return data_.data(); }
  T&       operator[](size_t i)       { return data_[i]; }
  const T& operator[](size_t i) const { return data_[i]; }
  T* begin() { return data_.data(); }
  T* end()   { return data_.data() + data_.size(); }

 private:
  std::vector<T> data_;
};

} // namespace shimaenaga
