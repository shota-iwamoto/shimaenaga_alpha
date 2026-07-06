#pragma once
#include <stdint.h>
#include "export.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void* SHIMAENAGA_DatasetHandle;
typedef void* SHIMAENAGA_BoosterHandle;

// Error handling (thread-local)
SHIMAENAGA_EXPORT const char* SHIMAENAGA_GetLastError(void);

// Dataset
SHIMAENAGA_EXPORT int SHIMAENAGA_DatasetCreate(
    const float* X, int64_t n, int32_t num_features,
    const float* y, const float* weights,
    const int32_t* group_sizes, int32_t num_groups,
    const char* params_json,
    SHIMAENAGA_DatasetHandle* out);

SHIMAENAGA_EXPORT int SHIMAENAGA_DatasetCreateLike(
    SHIMAENAGA_DatasetHandle train,
    const float* X, int64_t n,
    const float* y, const float* weights,
    const int32_t* group_sizes, int32_t num_groups,
    SHIMAENAGA_DatasetHandle* out);

SHIMAENAGA_EXPORT int SHIMAENAGA_DatasetFree(SHIMAENAGA_DatasetHandle h);

// Booster
SHIMAENAGA_EXPORT int SHIMAENAGA_BoosterCreate(
    SHIMAENAGA_DatasetHandle train,
    const char* params_json,
    SHIMAENAGA_BoosterHandle* out);

SHIMAENAGA_EXPORT int SHIMAENAGA_BoosterAddValid(SHIMAENAGA_BoosterHandle booster, SHIMAENAGA_DatasetHandle valid);
SHIMAENAGA_EXPORT int SHIMAENAGA_BoosterTrain(SHIMAENAGA_BoosterHandle booster);
SHIMAENAGA_EXPORT int SHIMAENAGA_BoosterFree(SHIMAENAGA_BoosterHandle booster);

SHIMAENAGA_EXPORT int SHIMAENAGA_BoosterPredict(
    SHIMAENAGA_BoosterHandle booster,
    const float* X, int64_t n, int32_t num_features,
    double* out);

SHIMAENAGA_EXPORT int SHIMAENAGA_BoosterPredictContrib(
    SHIMAENAGA_BoosterHandle booster,
    const float* X, int64_t n, int32_t num_features,
    double* score_out, float* beta_out);

SHIMAENAGA_EXPORT int SHIMAENAGA_BoosterSave(SHIMAENAGA_BoosterHandle booster, const char* path);
SHIMAENAGA_EXPORT int SHIMAENAGA_BoosterLoad(SHIMAENAGA_BoosterHandle booster, const char* path);

SHIMAENAGA_EXPORT int SHIMAENAGA_BoosterGetNumIterations(SHIMAENAGA_BoosterHandle booster, int* out);
SHIMAENAGA_EXPORT int SHIMAENAGA_BoosterGetNumClasses(SHIMAENAGA_BoosterHandle booster, int* out);
SHIMAENAGA_EXPORT int SHIMAENAGA_BoosterGetBestIteration(SHIMAENAGA_BoosterHandle booster, int* out);
SHIMAENAGA_EXPORT int SHIMAENAGA_BoosterGetNumTokens(SHIMAENAGA_BoosterHandle booster, int* out);
SHIMAENAGA_EXPORT int SHIMAENAGA_DatasetGetNumFeatures(SHIMAENAGA_DatasetHandle h, int* out);

#define SHIMAENAGA_SUCCESS 0
#define SHIMAENAGA_ERROR_CONFIG  1
#define SHIMAENAGA_ERROR_DATA    2
#define SHIMAENAGA_ERROR_TRAIN   3
#define SHIMAENAGA_ERROR_IO      4
#define SHIMAENAGA_ERROR_OTHER   5

#ifdef __cplusplus
}
#endif
