#ifndef __CUDA_PROCESS_DEPTH_H
#define __CUDA_PROCESS_DEPTH_H


#include "cudaUtility.h"
#include <stdint.h>

/**
 * Convert 32-bit float grayscale image to 32-bit RGB colormap
 * @ingroup util
 */
cudaError_t cudaGrayToRGB8(float* input, uchar3* output, size_t width, size_t height, float max_value);


/**
 * Divide all pixels of 32-bit grayscale image by the same value
 * @ingroup util
 */
cudaError_t cudaDivideByMaxValue(float* input, float* output, size_t width, size_t height, float max_value);

#endif
