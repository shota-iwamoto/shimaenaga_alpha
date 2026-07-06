#include "serializer.h"
#include "../util/log.h"
#include <fstream>
#include <cstring>
#include <stdexcept>
#include <sstream>

namespace shimaenaga {

static const char kMagic[4] = {'S','B','G','B'};
// v2: qA/kA/bA are always written (empty allowed). v1 wrote them only when
// non-empty and the loader sniffed the next byte for tags 5/6/7 — but raw rho
// float bytes follow, so a mantissa LSB of 5/6/7 desynced the whole stream.
// v3: + per-block attention mask (feature_local), tree nodes drop a dead pad
// u32. v2 files remain loadable (version-dispatched reader, 詳細設計書 §11).
// v4: + Tier-3 fields (T_L/d_u/d_f, e, layers, WR/WK, V_cls — Tier-3 詳細設計書
// §10). Files are written as v4 only when a Tier-3 block is present, so
// tier<=2 models remain readable by older releases.
static const uint32_t kVersion = 4;
static const uint32_t kMinVersion = 2;

// ─── Primitive I/O ───

void Serializer::WriteU32(std::vector<uint8_t>& buf, uint32_t v) {
  buf.push_back((v >> 0) & 0xFF); buf.push_back((v >> 8) & 0xFF);
  buf.push_back((v >> 16) & 0xFF); buf.push_back((v >> 24) & 0xFF);
}
void Serializer::WriteI32(std::vector<uint8_t>& buf, int32_t v) {
  WriteU32(buf, static_cast<uint32_t>(v));
}
void Serializer::WriteF32(std::vector<uint8_t>& buf, float v) {
  uint32_t u; std::memcpy(&u, &v, 4); WriteU32(buf, u);
}
void Serializer::WriteF64(std::vector<uint8_t>& buf, double v) {
  uint64_t u; std::memcpy(&u, &v, 8);
  WriteU32(buf, static_cast<uint32_t>(u));
  WriteU32(buf, static_cast<uint32_t>(u >> 32));
}
void Serializer::WriteStr(std::vector<uint8_t>& buf, const std::string& s) {
  WriteU32(buf, static_cast<uint32_t>(s.size()));
  for (char c : s) buf.push_back(static_cast<uint8_t>(c));
}

uint32_t Serializer::ReadU32(const uint8_t* data, size_t& pos) {
  uint32_t v = data[pos] | (data[pos+1]<<8) | (data[pos+2]<<16) | (data[pos+3]<<24);
  pos += 4; return v;
}
int32_t Serializer::ReadI32(const uint8_t* data, size_t& pos) {
  return static_cast<int32_t>(ReadU32(data, pos));
}
float Serializer::ReadF32(const uint8_t* data, size_t& pos) {
  uint32_t u = ReadU32(data, pos); float v; std::memcpy(&v, &u, 4); return v;
}
double Serializer::ReadF64(const uint8_t* data, size_t& pos) {
  uint64_t lo = ReadU32(data, pos), hi = ReadU32(data, pos);
  uint64_t u = lo | (hi << 32); double v; std::memcpy(&v, &u, 8); return v;
}
std::string Serializer::ReadStr(const uint8_t* data, size_t& pos, size_t size) {
  uint32_t len = ReadU32(data, pos);
  if (pos + len > size) throw IOError("String read out of bounds");
  std::string s(reinterpret_cast<const char*>(data + pos), len); pos += len; return s;
}

// ─── CRC32 ───

uint32_t Serializer::CRC32(const uint8_t* data, size_t len) {
  // Standard IEEE 802.3 CRC-32 (polynomial 0xEDB88420), table built on first use.
  static uint32_t tbl[256];
  static bool init = false;
  if (!init) {
    for (uint32_t n = 0; n < 256; ++n) {
      uint32_t c = n;
      for (int k = 0; k < 8; ++k)
        c = (c & 1) ? (0xEDB88420u ^ (c >> 1)) : (c >> 1);
      tbl[n] = c;
    }
    init = true;
  }
  uint32_t crc = 0xFFFFFFFFu;
  for (size_t i = 0; i < len; ++i)
    crc = (crc >> 8) ^ tbl[(crc ^ data[i]) & 0xFF];
  return crc ^ 0xFFFFFFFFu;
}

// ─── BinMapper I/O ───

void Serializer::WriteBinMapper(std::vector<uint8_t>& buf, const BinMapper& m) {
  buf.push_back(static_cast<uint8_t>(m.Type()));  // 0=Numerical, 1=Categorical
  WriteI32(buf, m.MaxCatId());
  const auto& thr = m.Thresholds();
  WriteU32(buf, static_cast<uint32_t>(thr.size()));
  for (float v : thr) WriteF32(buf, v);
  const auto& cat = m.CatMap();
  WriteU32(buf, static_cast<uint32_t>(cat.size()));
  for (uint32_t v : cat) WriteU32(buf, v);
}

BinMapper Serializer::ReadBinMapper(const uint8_t* data, size_t& pos, size_t size) {
  BinMapper m;
  uint8_t ftype_byte = data[pos++];
  FeatureType ft = (ftype_byte == 1) ? FeatureType::Categorical : FeatureType::Numerical;
  int max_cat = ReadI32(data, pos);

  uint32_t n_thr = ReadU32(data, pos);
  std::vector<float> thr(n_thr);
  for (uint32_t i = 0; i < n_thr; ++i) thr[i] = ReadF32(data, pos);

  uint32_t n_cat = ReadU32(data, pos);
  std::vector<uint32_t> cat(n_cat);
  for (uint32_t i = 0; i < n_cat; ++i) cat[i] = ReadU32(data, pos);

  // Reconstruct by calling Fit on a synthetic single value if needed,
  // but it's cleaner to directly set internal state.
  // We use friend access via a helper lambda in bin_mapper.cpp.
  // Since we can't directly set private fields, we reconstruct via Fit.
  // Instead, use the SetState helper (added to BinMapper public API):
  m.SetState(ft, std::move(thr), std::move(cat), max_cat);
  (void)size;
  return m;
}

// ─── Tree I/O ───

void Serializer::WriteTree(std::vector<uint8_t>& buf, const Tree& t) {
  WriteI32(buf, t.num_leaves);
  int nn = t.num_leaves - 1;
  for (int i = 0; i < nn; ++i) {
    WriteI32(buf, t.split_feature[i]);
    buf.push_back(t.threshold_bin[i]);
    buf.push_back(t.default_left[i]);
    buf.push_back(t.is_categorical[i]);
    uint64_t bs = t.cat_bitset.empty() ? 0 : t.cat_bitset[i];
    WriteU32(buf, static_cast<uint32_t>(bs));
    WriteU32(buf, static_cast<uint32_t>(bs >> 32));
    WriteI32(buf, t.left_child[i]);
    WriteI32(buf, t.right_child[i]);
  }
}

Tree Serializer::ReadTree(const uint8_t* data, size_t& pos, size_t size,
                          uint32_t version) {
  Tree t;
  t.num_leaves = ReadI32(data, pos);
  int nn = t.num_leaves - 1;
  t.split_feature.resize(nn); t.threshold_bin.resize(nn);
  t.default_left.resize(nn); t.is_categorical.resize(nn);
  t.cat_bitset.resize(nn); t.left_child.resize(nn); t.right_child.resize(nn);
  for (int i = 0; i < nn; ++i) {
    t.split_feature[i] = ReadI32(data, pos);
    t.threshold_bin[i] = data[pos++];
    t.default_left[i]  = data[pos++];
    t.is_categorical[i] = data[pos++];
    if (version <= 2) ReadU32(data, pos);  // v2 wrote a dead pad u32 per node
    uint32_t bslo = ReadU32(data, pos), bshi = ReadU32(data, pos);
    t.cat_bitset[i] = bslo | (static_cast<uint64_t>(bshi) << 32);
    t.left_child[i]  = ReadI32(data, pos);
    t.right_child[i] = ReadI32(data, pos);
  }
  (void)size;
  return t;
}

// ─── Float array I/O ───

void Serializer::WriteFloat32Array(std::vector<uint8_t>& buf,
                                    const std::vector<param_t>& arr, uint8_t tag) {
  buf.push_back(tag);
  WriteU32(buf, static_cast<uint32_t>(arr.size()));
  for (float v : arr) WriteF32(buf, v);
}

std::vector<param_t> Serializer::ReadFloat32Array(const uint8_t* data, size_t& pos, size_t size) {
  uint8_t tag = data[pos++]; (void)tag;
  uint32_t n = ReadU32(data, pos);
  std::vector<param_t> arr(n);
  for (uint32_t i = 0; i < n; ++i) arr[i] = ReadF32(data, pos);
  (void)size;
  return arr;
}

// ─── Save/Load ───

void Serializer::Save(const Model& model, const std::string& path) {
  std::vector<uint8_t> buf;

  // v4 only when a Tier-3 block exists (older readers keep loading tier<=2).
  bool any_t3 = false;
  for (const auto& blk : model.blocks)
    if (blk.T_L > 0) { any_t3 = true; break; }
  const uint32_t file_ver = any_t3 ? 4u : 3u;

  // Magic + version
  buf.insert(buf.end(), kMagic, kMagic + 4);
  WriteU32(buf, file_ver);

  // Config (minimal JSON). The Tier-3 keys are required at inference
  // (ApplyBlockT3 mixes with η's from train_cfg); harmless for old readers.
  std::ostringstream cfg_ss;
  cfg_ss << "{\"objective\":\"" << model.train_cfg.objective << "\""
         << ",\"num_class\":" << model.train_cfg.num_class
         << ",\"tier\":" << model.train_cfg.tier
         << ",\"num_tokens\":" << model.train_cfg.num_tokens
         << ",\"num_heads\":" << model.train_cfg.num_heads
         << ",\"d_attn\":" << model.train_cfg.d_attn
         << ",\"attention_mode\":\"" << model.train_cfg.attention_mode << "\""
         << ",\"eta_attn\":" << model.train_cfg.eta_attn
         << ",\"learning_rate\":" << model.train_cfg.learning_rate
         << ",\"attn_layers\":" << model.train_cfg.attn_layers
         << ",\"d_hidden\":" << model.train_cfg.d_hidden
         << ",\"d_ffn\":" << model.train_cfg.d_ffn
         << ",\"eta_u\":" << model.train_cfg.eta_u
         << ",\"eta_ffn\":" << model.train_cfg.eta_ffn
         << ",\"eta_cls\":" << model.train_cfg.eta_cls
         << ",\"use_cls_token\":" << (model.train_cfg.use_cls_token ? 1 : 0)
         << ",\"norm_eps\":" << model.train_cfg.norm_eps
         << "}";
  WriteStr(buf, cfg_ss.str());

  // C and F0
  WriteI32(buf, model.C);
  WriteU32(buf, static_cast<uint32_t>(model.F0.size()));
  for (double v : model.F0) WriteF64(buf, v);

  // Blocks
  WriteU32(buf, static_cast<uint32_t>(model.blocks.size()));
  for (const auto& blk : model.blocks) {
    WriteI32(buf, blk.P); WriteI32(buf, blk.H);
    WriteI32(buf, blk.C); WriteI32(buf, blk.d_a);
    WriteI32(buf, blk.tier); WriteI32(buf, blk.gate_num_leaves);
    WriteStr(buf, blk.attention_mode);
    // token L sizes
    for (int p = 0; p < blk.P; ++p) WriteI32(buf, blk.v_lsize[p]);
    // Token trees
    for (const auto& t : blk.token_trees) WriteTree(buf, t);
    // Gate tree
    WriteTree(buf, blk.gate_tree);
    // Params
    WriteFloat32Array(buf, blk.v,      1);
    WriteFloat32Array(buf, blk.z_or_q, 2);
    WriteFloat32Array(buf, blk.k,      3);
    WriteFloat32Array(buf, blk.b,      4);
    WriteFloat32Array(buf, blk.qA, 5);  // empty for tier<2
    WriteFloat32Array(buf, blk.kA, 6);
    WriteFloat32Array(buf, blk.bA, 7);
    // Head weights
    for (int h = 0; h < blk.H; ++h) WriteF32(buf, blk.rho[h]);
    for (int h = 0; h < blk.H; ++h) WriteF32(buf, blk.rhoA[h]);
    // v3: attention mask rows (empty = full attention)
    WriteU32(buf, static_cast<uint32_t>(blk.attn_mask.size()));
    for (uint32_t m : blk.attn_mask) WriteU32(buf, m);
    // v4: Tier-3 fields (Tier-3 詳細設計書 §10). T_L == 0 blocks (tier<=2 or
    // warm-up) write only the three dims — no arrays.
    if (file_ver >= 4) {
      WriteI32(buf, blk.T_L); WriteI32(buf, blk.d_u); WriteI32(buf, blk.d_f);
      if (blk.T_L > 0) {
        WriteFloat32Array(buf, blk.e,      8);
        WriteFloat32Array(buf, blk.e_gate, 9);
        WriteFloat32Array(buf, blk.WR,     10);
        WriteFloat32Array(buf, blk.WK,     11);
        WriteFloat32Array(buf, blk.V_cls,  12);
        for (int h = 0; h < blk.H; ++h) WriteF32(buf, blk.theta_R[h]);
        for (const auto& L : blk.layers) {
          WriteFloat32Array(buf, L.Wq,  13);
          WriteFloat32Array(buf, L.Wk,  14);
          WriteFloat32Array(buf, L.Wv,  15);
          WriteFloat32Array(buf, L.a_q, 16);
          WriteFloat32Array(buf, L.a_k, 17);
          WriteFloat32Array(buf, L.bA3, 18);
          WriteFloat32Array(buf, L.W1,  19);
          WriteFloat32Array(buf, L.W2,  20);
          WriteFloat32Array(buf, L.c1,  21);
          for (int h = 0; h < blk.H; ++h) WriteF32(buf, L.rho3[h]);
          for (int h = 0; h < blk.H; ++h) WriteF32(buf, L.theta[h]);
          WriteF32(buf, L.gamma_c);
        }
      }
    }
  }

  // BinMappers section (tag 0x42 'B')
  buf.push_back(0x42);
  WriteU32(buf, static_cast<uint32_t>(model.bin_mappers.size()));
  for (const auto& m : model.bin_mappers) WriteBinMapper(buf, m);

  // CRC32 over everything written so far (integrity check on load).
  uint32_t crc = CRC32(buf.data(), buf.size());
  WriteU32(buf, crc);

  // Write to file
  std::ofstream f(path, std::ios::binary);
  if (!f) throw IOError("Cannot open file for writing: " + path);
  f.write(reinterpret_cast<const char*>(buf.data()), buf.size());
  SHIMAENAGA_LOG_INFO("Model saved to %s (%zu bytes)", path.c_str(), buf.size());
}

Model Serializer::Load(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw IOError("Cannot open model file: " + path);

  f.seekg(0, std::ios::end);
  size_t sz = f.tellg(); f.seekg(0);
  std::vector<uint8_t> buf(sz);
  f.read(reinterpret_cast<char*>(buf.data()), sz);

  size_t pos = 0;
  // Check magic
  if (sz < 8 || buf[0]!='S'||buf[1]!='B'||buf[2]!='G'||buf[3]!='B')
    throw IOError("Invalid model file (bad magic)");

  // Verify CRC32 (last 4 bytes) over the preceding payload.
  uint32_t stored_crc = buf[sz-4] | (buf[sz-3]<<8) | (buf[sz-2]<<16) |
                        (static_cast<uint32_t>(buf[sz-1])<<24);
  uint32_t actual_crc = CRC32(buf.data(), sz - 4);
  if (stored_crc != actual_crc)
    throw IOError("Model file CRC mismatch (corrupted or truncated)");
  pos = 4;
  uint32_t ver = ReadU32(buf.data(), pos);
  if (ver < kMinVersion || ver > kVersion)
    throw IOError("Model version mismatch: got " + std::to_string(ver) +
                  " (supported: " + std::to_string(kMinVersion) + ".." +
                  std::to_string(kVersion) + ")");

  // Config JSON (skip parsing for now)
  std::string cfg_json = ReadStr(buf.data(), pos, sz);

  Model model;
  // Parse minimal config
  auto find = [&](const std::string& key) -> std::string {
    auto idx = cfg_json.find("\"" + key + "\":");
    if (idx == std::string::npos) return "";
    idx += key.size() + 3;
    auto end = cfg_json.find_first_of(",}", idx);
    return cfg_json.substr(idx, end - idx);
  };
  model.train_cfg.objective = find("objective");
  // Remove quotes
  if (!model.train_cfg.objective.empty() && model.train_cfg.objective.front() == '"')
    model.train_cfg.objective = model.train_cfg.objective.substr(1, model.train_cfg.objective.size()-2);
  auto to_int = [](const std::string& s) { return s.empty() ? 0 : std::stoi(s); };
  auto to_dbl = [](const std::string& s) { return s.empty() ? 0.0 : std::stod(s); };
  model.train_cfg.num_class  = to_int(find("num_class"));
  model.train_cfg.tier       = to_int(find("tier"));
  model.train_cfg.num_tokens = to_int(find("num_tokens"));
  model.train_cfg.num_heads  = to_int(find("num_heads"));
  model.train_cfg.d_attn     = to_int(find("d_attn"));
  model.train_cfg.attention_mode = find("attention_mode");
  if (!model.train_cfg.attention_mode.empty() && model.train_cfg.attention_mode.front() == '"')
    model.train_cfg.attention_mode = model.train_cfg.attention_mode.substr(1, model.train_cfg.attention_mode.size()-2);
  model.train_cfg.eta_attn     = to_dbl(find("eta_attn"));
  model.train_cfg.learning_rate = to_dbl(find("learning_rate"));
  // Tier-3 keys (absent in files written before v4 → keep the defaults)
  if (!find("attn_layers").empty()) {
    model.train_cfg.attn_layers = to_int(find("attn_layers"));
    model.train_cfg.d_hidden    = to_int(find("d_hidden"));
    model.train_cfg.d_ffn       = to_int(find("d_ffn"));
    model.train_cfg.eta_u       = to_dbl(find("eta_u"));
    model.train_cfg.eta_ffn     = to_dbl(find("eta_ffn"));
    model.train_cfg.eta_cls     = to_dbl(find("eta_cls"));
    model.train_cfg.use_cls_token = to_int(find("use_cls_token")) != 0;
    model.train_cfg.norm_eps    = to_dbl(find("norm_eps"));
  }

  model.C = ReadI32(buf.data(), pos);
  uint32_t f0_size = ReadU32(buf.data(), pos);
  model.F0.resize(f0_size);
  for (uint32_t i = 0; i < f0_size; ++i) model.F0[i] = ReadF64(buf.data(), pos);

  uint32_t n_blocks = ReadU32(buf.data(), pos);
  model.blocks.resize(n_blocks);
  for (uint32_t bi = 0; bi < n_blocks; ++bi) {
    auto& blk = model.blocks[bi];
    blk.P = ReadI32(buf.data(), pos); blk.H = ReadI32(buf.data(), pos);
    blk.C = ReadI32(buf.data(), pos); blk.d_a = ReadI32(buf.data(), pos);
    blk.tier = ReadI32(buf.data(), pos); blk.gate_num_leaves = ReadI32(buf.data(), pos);
    blk.attention_mode = ReadStr(buf.data(), pos, sz);
    blk.v_lsize.resize(blk.P);
    for (int p = 0; p < blk.P; ++p) blk.v_lsize[p] = ReadI32(buf.data(), pos);
    blk.token_trees.resize(blk.P);
    for (int p = 0; p < blk.P; ++p) blk.token_trees[p] = ReadTree(buf.data(), pos, sz, ver);
    blk.gate_tree = ReadTree(buf.data(), pos, sz, ver);
    blk.v      = ReadFloat32Array(buf.data(), pos, sz);
    blk.z_or_q = ReadFloat32Array(buf.data(), pos, sz);
    blk.k      = ReadFloat32Array(buf.data(), pos, sz);
    blk.b      = ReadFloat32Array(buf.data(), pos, sz);
    blk.qA = ReadFloat32Array(buf.data(), pos, sz);  // always present since v2
    blk.kA = ReadFloat32Array(buf.data(), pos, sz);
    blk.bA = ReadFloat32Array(buf.data(), pos, sz);
    for (int h = 0; h < blk.H; ++h) blk.rho[h]  = ReadF32(buf.data(), pos);
    for (int h = 0; h < blk.H; ++h) blk.rhoA[h] = ReadF32(buf.data(), pos);
    if (ver >= 3) {
      uint32_t nm = ReadU32(buf.data(), pos);
      blk.attn_mask.resize(nm);
      for (uint32_t t = 0; t < nm; ++t) blk.attn_mask[t] = ReadU32(buf.data(), pos);
    }  // v2: no mask → full attention (empty)
    if (ver >= 4) {
      blk.T_L = ReadI32(buf.data(), pos);
      blk.d_u = ReadI32(buf.data(), pos);
      blk.d_f = ReadI32(buf.data(), pos);
      if (blk.T_L > 0) {
        blk.e      = ReadFloat32Array(buf.data(), pos, sz);
        blk.e_gate = ReadFloat32Array(buf.data(), pos, sz);
        blk.WR     = ReadFloat32Array(buf.data(), pos, sz);
        blk.WK     = ReadFloat32Array(buf.data(), pos, sz);
        blk.V_cls  = ReadFloat32Array(buf.data(), pos, sz);
        for (int h = 0; h < blk.H; ++h) blk.theta_R[h] = ReadF32(buf.data(), pos);
        blk.layers.resize(blk.T_L);
        for (auto& L : blk.layers) {
          L.Wq  = ReadFloat32Array(buf.data(), pos, sz);
          L.Wk  = ReadFloat32Array(buf.data(), pos, sz);
          L.Wv  = ReadFloat32Array(buf.data(), pos, sz);
          L.a_q = ReadFloat32Array(buf.data(), pos, sz);
          L.a_k = ReadFloat32Array(buf.data(), pos, sz);
          L.bA3 = ReadFloat32Array(buf.data(), pos, sz);
          L.W1  = ReadFloat32Array(buf.data(), pos, sz);
          L.W2  = ReadFloat32Array(buf.data(), pos, sz);
          L.c1  = ReadFloat32Array(buf.data(), pos, sz);
          for (int h = 0; h < blk.H; ++h) L.rho3[h] = ReadF32(buf.data(), pos);
          for (int h = 0; h < blk.H; ++h) L.theta[h] = ReadF32(buf.data(), pos);
          L.gamma_c = ReadF32(buf.data(), pos);
        }
      }
    }  // v2/v3: no Tier-3 fields → T_L = 0 (tier<=2 evaluation)
  }

  // BinMappers section (optional, tag 0x42 'B')
  if (pos < sz && buf[pos] == 0x42) {
    pos++;
    uint32_t n_mappers = ReadU32(buf.data(), pos);
    model.bin_mappers.resize(n_mappers);
    for (uint32_t mi = 0; mi < n_mappers; ++mi)
      model.bin_mappers[mi] = ReadBinMapper(buf.data(), pos, sz);
  }

  SHIMAENAGA_LOG_INFO("Model loaded from %s, %u blocks", path.c_str(), n_blocks);
  return model;
}

void Serializer::SaveText(const Model& model, const std::string& path) {
  std::ofstream f(path);
  if (!f) throw IOError("Cannot open file: " + path);
  f << "Shimaenaga model dump\n";
  f << "Blocks: " << model.blocks.size() << "\n";
  f << "C: " << model.C << "\n";
  for (const auto& blk : model.blocks) {
    f << "Block P=" << blk.P << " H=" << blk.H
      << " gate_leaves=" << blk.gate_num_leaves << "\n";
  }
}

} // namespace shimaenaga
