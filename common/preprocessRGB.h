#ifndef __PREPROCESS_RGB_H
#define __PREPROCESS_RGB_H

#include "cudaUtility.h"

// from imageNet.cu
cudaError_t cudaPreImageNetMean( float4* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, const float3& mean_value, cudaStream_t stream );

#endif
