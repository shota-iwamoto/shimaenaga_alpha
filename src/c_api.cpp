#include "../include/shimaenaga/c_api.h"
#include "../include/shimaenaga/booster.h"
#include "../include/shimaenaga/dataset.h"
#include "util/log.h"
#include <stdexcept>
#include <sstream>
#include <map>
#include <string>
#include <cstring>

namespace {

thread_local std::string g_last_error;

void SetError(const char* msg) { g_last_error = msg; }

// Parse minimal key=value params (supports JSON-like format)
std::map<std::string, std::string> ParseParams(const char* json_or_kv) {
  std::map<std::string, std::string> out;
  if (!json_or_kv || !json_or_kv[0]) return out;
  std::string s(json_or_kv);

  // Try JSON: {"key":"value","key2":value}
  if (!s.empty() && s[0] == '{') {
    // Simple JSON parser
    size_t i = 1;
    while (i < s.size() && s[i] != '}') {
      // Skip whitespace and commas
      while (i < s.size() && (s[i]==' '||s[i]==','||s[i]=='\n'||s[i]=='\r')) i++;
      if (s[i] == '}') break;
      // Read key (quoted)
      std::string key, val;
      if (s[i] == '"') {
        i++;
        while (i < s.size() && s[i] != '"') key += s[i++];
        i++;  // skip closing "
      } else {
        while (i < s.size() && s[i] != ':') key += s[i++];
      }
      while (i < s.size() && s[i]==' ') i++;
      if (i < s.size() && s[i]==':') i++;
      while (i < s.size() && s[i]==' ') i++;
      // Read value
      if (i < s.size() && s[i] == '"') {
        i++;
        while (i < s.size() && s[i] != '"') val += s[i++];
        i++;
      } else {
        while (i < s.size() && s[i]!=',' && s[i]!='}') val += s[i++];
        // Trim
        while (!val.empty() && val.back()==' ') val.pop_back();
      }
      if (!key.empty()) out[key] = val;
    }
  } else {
    // key=value pairs, space/newline separated
    std::istringstream ss(s);
    std::string token;
    while (ss >> token) {
      auto eq = token.find('=');
      if (eq != std::string::npos)
        out[token.substr(0,eq)] = token.substr(eq+1);
    }
  }
  return out;
}

#define SHIMAENAGA_CATCH_RETURN(code) \
  catch (const shimaenaga::ConfigError& e) { SetError(e.what()); return SHIMAENAGA_ERROR_CONFIG; } \
  catch (const shimaenaga::DataError&   e) { SetError(e.what()); return SHIMAENAGA_ERROR_DATA;   } \
  catch (const shimaenaga::TrainError&  e) { SetError(e.what()); return SHIMAENAGA_ERROR_TRAIN;  } \
  catch (const shimaenaga::IOError&     e) { SetError(e.what()); return SHIMAENAGA_ERROR_IO;     } \
  catch (const std::exception&          e) { SetError(e.what()); return SHIMAENAGA_ERROR_OTHER;  }

} // anonymous namespace

extern "C" {

const char* SHIMAENAGA_GetLastError(void) { return g_last_error.c_str(); }

int SHIMAENAGA_DatasetCreate(
    const float* X, int64_t n, int32_t nf,
    const float* y, const float* w,
    const int32_t* grp, int32_t ng,
    const char* params_json, SHIMAENAGA_DatasetHandle* out) {
  try {
    auto cfg = shimaenaga::Config::FromMap(ParseParams(params_json));
    auto ds = shimaenaga::Dataset::Build(X, (shimaenaga::data_size_t)n, nf,
                                          y, w, grp, ng, cfg);
    *out = new std::shared_ptr<shimaenaga::Dataset>(std::move(ds));
    return SHIMAENAGA_SUCCESS;
  } SHIMAENAGA_CATCH_RETURN(code)
}

int SHIMAENAGA_DatasetCreateLike(
    SHIMAENAGA_DatasetHandle train, const float* X, int64_t n,
    const float* y, const float* w,
    const int32_t* grp, int32_t ng, SHIMAENAGA_DatasetHandle* out) {
  try {
    auto& train_sp = *static_cast<std::shared_ptr<shimaenaga::Dataset>*>(train);
    auto ds = shimaenaga::Dataset::BuildLike(*train_sp, X, (shimaenaga::data_size_t)n, y, w, grp, ng);
    *out = new std::shared_ptr<shimaenaga::Dataset>(std::move(ds));
    return SHIMAENAGA_SUCCESS;
  } SHIMAENAGA_CATCH_RETURN(code)
}

int SHIMAENAGA_DatasetFree(SHIMAENAGA_DatasetHandle h) {
  delete static_cast<std::shared_ptr<shimaenaga::Dataset>*>(h);
  return SHIMAENAGA_SUCCESS;
}

int SHIMAENAGA_BoosterCreate(SHIMAENAGA_DatasetHandle train, const char* params_json, SHIMAENAGA_BoosterHandle* out) {
  try {
    auto cfg = shimaenaga::Config::FromMap(ParseParams(params_json));
    auto& train_sp = *static_cast<std::shared_ptr<shimaenaga::Dataset>*>(train);
    *out = new shimaenaga::Booster(cfg, train_sp);
    return SHIMAENAGA_SUCCESS;
  } SHIMAENAGA_CATCH_RETURN(code)
}

int SHIMAENAGA_BoosterAddValid(SHIMAENAGA_BoosterHandle booster, SHIMAENAGA_DatasetHandle valid) {
  try {
    auto* b = static_cast<shimaenaga::Booster*>(booster);
    auto& valid_sp = *static_cast<std::shared_ptr<shimaenaga::Dataset>*>(valid);
    b->AddValidData(valid_sp);
    return SHIMAENAGA_SUCCESS;
  } SHIMAENAGA_CATCH_RETURN(code)
}

int SHIMAENAGA_BoosterTrain(SHIMAENAGA_BoosterHandle booster) {
  try {
    static_cast<shimaenaga::Booster*>(booster)->Train();
    return SHIMAENAGA_SUCCESS;
  } SHIMAENAGA_CATCH_RETURN(code)
}

int SHIMAENAGA_BoosterFree(SHIMAENAGA_BoosterHandle booster) {
  delete static_cast<shimaenaga::Booster*>(booster);
  return SHIMAENAGA_SUCCESS;
}

int SHIMAENAGA_BoosterPredict(SHIMAENAGA_BoosterHandle booster,
                       const float* X, int64_t n, int32_t nf, double* out) {
  try {
    auto* b = static_cast<shimaenaga::Booster*>(booster);
    auto pred = b->Predict(X, (shimaenaga::data_size_t)n, nf);
    std::memcpy(out, pred.data(), pred.size() * sizeof(double));
    return SHIMAENAGA_SUCCESS;
  } SHIMAENAGA_CATCH_RETURN(code)
}

int SHIMAENAGA_BoosterPredictContrib(SHIMAENAGA_BoosterHandle booster,
                              const float* X, int64_t n, int32_t nf,
                              double* score_out, float* beta_out) {
  try {
    auto* b = static_cast<shimaenaga::Booster*>(booster);
    std::vector<float> beta;
    auto pred = b->PredictContrib(X, (shimaenaga::data_size_t)n, nf,
                                   beta_out ? &beta : nullptr);
    if (!pred.empty()) std::memcpy(score_out, pred.data(), pred.size() * sizeof(double));
    if (beta_out && !beta.empty()) std::memcpy(beta_out, beta.data(), beta.size()*sizeof(float));
    return SHIMAENAGA_SUCCESS;
  } SHIMAENAGA_CATCH_RETURN(code)
}

int SHIMAENAGA_BoosterSave(SHIMAENAGA_BoosterHandle booster, const char* path) {
  try {
    static_cast<shimaenaga::Booster*>(booster)->SaveModel(path);
    return SHIMAENAGA_SUCCESS;
  } SHIMAENAGA_CATCH_RETURN(code)
}

int SHIMAENAGA_BoosterLoad(SHIMAENAGA_BoosterHandle booster, const char* path) {
  try {
    static_cast<shimaenaga::Booster*>(booster)->LoadModel(path);
    return SHIMAENAGA_SUCCESS;
  } SHIMAENAGA_CATCH_RETURN(code)
}

int SHIMAENAGA_BoosterGetNumIterations(SHIMAENAGA_BoosterHandle booster, int* out) {
  *out = (int)static_cast<shimaenaga::Booster*>(booster)->GetModel().blocks.size();
  return SHIMAENAGA_SUCCESS;
}

int SHIMAENAGA_BoosterGetNumClasses(SHIMAENAGA_BoosterHandle booster, int* out) {
  *out = static_cast<shimaenaga::Booster*>(booster)->GetModel().C;
  return SHIMAENAGA_SUCCESS;
}

int SHIMAENAGA_BoosterGetBestIteration(SHIMAENAGA_BoosterHandle booster, int* out) {
  *out = static_cast<shimaenaga::Booster*>(booster)->BestIteration();
  return SHIMAENAGA_SUCCESS;
}

int SHIMAENAGA_BoosterGetNumTokens(SHIMAENAGA_BoosterHandle booster, int* out) {
  const auto& blocks = static_cast<shimaenaga::Booster*>(booster)->GetModel().blocks;
  *out = blocks.empty() ? 0 : blocks[0].P;
  return SHIMAENAGA_SUCCESS;
}

int SHIMAENAGA_DatasetGetNumFeatures(SHIMAENAGA_DatasetHandle h, int* out) {
  *out = (*static_cast<std::shared_ptr<shimaenaga::Dataset>*>(h))->NumFeatures();
  return SHIMAENAGA_SUCCESS;
}

} // extern "C"
