#pragma once
#include <cstdint>
#include <vector>
#include <algorithm>

namespace shimaenaga {

// xorshift128+ for deterministic, fast random numbers (T10)
class Random {
 public:
  explicit Random(uint64_t seed = 0) {
    s_[0] = splitmix64(seed + 1);
    s_[1] = splitmix64(s_[0] + 1);
  }

  uint64_t NextUInt64() {
    uint64_t s1 = s_[0], s0 = s_[1];
    s_[0] = s0;
    s1 ^= s1 << 23;
    s_[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
    return s_[1] + s0;
  }

  double NextDouble() {
    return (NextUInt64() >> 11) * (1.0 / (1ull << 53));
  }

  int NextInt(int lo, int hi) {
    return lo + static_cast<int>(NextUInt64() % static_cast<uint64_t>(hi - lo));
  }

  // Fisher-Yates shuffle
  void Shuffle(std::vector<int>& v) {
    for (int i = static_cast<int>(v.size()) - 1; i > 0; --i) {
      int j = static_cast<int>(NextUInt64() % static_cast<uint64_t>(i + 1));
      std::swap(v[i], v[j]);
    }
  }

  // Sample k items from [0..n) without replacement
  std::vector<int> SampleK(int n, int k) {
    std::vector<int> pool(n);
    for (int i = 0; i < n; ++i) pool[i] = i;
    Shuffle(pool);
    pool.resize(k);
    std::sort(pool.begin(), pool.end());
    return pool;
  }

 private:
  uint64_t s_[2];
  static uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
  }
};

} // namespace shimaenaga
