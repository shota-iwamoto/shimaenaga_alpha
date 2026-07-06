#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "../../include/shimaenaga/model.h"

namespace shimaenaga {

// Binary model serializer (詳細設計書 §11)
// Format: [magic "SBGB"][version u32][Config JSON][F0][num_blocks][blocks...][CRC32]
class Serializer {
 public:
  static void Save(const Model& model, const std::string& path);
  static Model Load(const std::string& path);

  // Text dump (human readable, v1.1)
  static void SaveText(const Model& model, const std::string& path);

 private:
  static void WriteTree(std::vector<uint8_t>& buf, const Tree& tree);
  static Tree ReadTree(const uint8_t* data, size_t& pos, size_t size,
                       uint32_t version);

  static void WriteBinMapper(std::vector<uint8_t>& buf, const BinMapper& m);
  static BinMapper ReadBinMapper(const uint8_t* data, size_t& pos, size_t size);

  static void WriteFloat32Array(std::vector<uint8_t>& buf,
                                 const std::vector<param_t>& arr, uint8_t tag);
  static std::vector<param_t> ReadFloat32Array(const uint8_t* data, size_t& pos, size_t size);

  static uint32_t CRC32(const uint8_t* data, size_t len);

  // Write primitives
  static void WriteU32(std::vector<uint8_t>& buf, uint32_t v);
  static void WriteI32(std::vector<uint8_t>& buf, int32_t v);
  static void WriteF32(std::vector<uint8_t>& buf, float v);
  static void WriteF64(std::vector<uint8_t>& buf, double v);
  static void WriteStr(std::vector<uint8_t>& buf, const std::string& s);

  static uint32_t ReadU32(const uint8_t* data, size_t& pos);
  static int32_t  ReadI32(const uint8_t* data, size_t& pos);
  static float    ReadF32(const uint8_t* data, size_t& pos);
  static double   ReadF64(const uint8_t* data, size_t& pos);
  static std::string ReadStr(const uint8_t* data, size_t& pos, size_t size);
};

} // namespace shimaenaga
