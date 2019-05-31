#include "processDepth.h"

__global__ void GrayToRGB8(float* srcImage, uchar3* dstImage, int width, int height, float max_value)
{
	const int x_ = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	const int pixel = y * width + x_;

	if( x_ >= width )
		return; 

	if( y >= height )
		return;

	float x = srcImage[pixel] / max_value;
	float r, g, b;
	x = x * 6;
	r = 0.0f; g = 0.0f; b = 0.0f;
	if ((0 <= x && x <= 1) || (5 <= x && x <= 6)) r = 1.0f;
	else if (4 <= x && x <= 5) r = x - 4;
	else if (1 <= x && x <= 2) r = 1.0f - (x - 1);
	if (1 <= x && x <= 3) g = 1.0f;
	else if (0 <= x && x <= 1) g = x - 0;
	else if (3 <= x && x <= 4) g = 1.0f - (x - 3);
	if (3 <= x && x <= 5) b = 1.0f;
	else if (2 <= x && x <= 3) b = x - 2;
	else if (5 <= x && x <= 6) b = 1.0f - (x - 5);
	dstImage[pixel] = make_uchar3(b * 255, g * 255, r * 255);
}

cudaError_t cudaGrayToRGB8(float* srcDev, uchar3* destDev, size_t width, size_t height, float max_value )
{
	if( !srcDev || !destDev )
		return cudaErrorInvalidDevicePointer;

	const dim3 blockDim(8,8,1);
	const dim3 gridDim(iDivUp(width, blockDim.x), iDivUp(height, blockDim.y), 1);

	GrayToRGB8<<<gridDim, blockDim>>>( srcDev, destDev, width, height, max_value );
	
	return CUDA(cudaGetLastError());
}


__global__ void DivideByMaxValue(float* srcImage, float* dstImage, int width, int height, float max_value)
{
	const int x_ = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	const int pixel = y * width + x_;

	if( x_ >= width )
		return; 

	if( y >= height )
		return;

	float x = srcImage[pixel] / max_value;
	dstImage[pixel] = x;
}

cudaError_t cudaDivideByMaxValue(float* srcDev, float* destDev, size_t width, size_t height, float max_value )
{
	if( !srcDev || !destDev )
		return cudaErrorInvalidDevicePointer;

	const dim3 blockDim(8,8,1);
	const dim3 gridDim(iDivUp(width, blockDim.x), iDivUp(height, blockDim.y), 1);

	DivideByMaxValue<<<gridDim, blockDim>>>( srcDev, destDev, width, height, max_value );
	
	return CUDA(cudaGetLastError());
}
