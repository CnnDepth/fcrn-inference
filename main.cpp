#include "gstCamera.h"

#include "glDisplay.h"
#include "glTexture.h"

#include "NvInfer.h"

#include <signal.h>
#include <unistd.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>
#include <chrono>
#include <cublas_v2.h>
#include <cudnn.h>

#include "cudaResize.h"
#include "cudaMappedMemory.h"
#include "cudaNormalize.h"
#include "cudaFont.h"
#include "cudaRGB.h"
#include "cudaResize.h"
#include "cudaOverlay.h"
#include "cudaUtility.h"
#include "preprocessRGB.h"
#include "processDepth.h"
#include "upsamplingPlugin.h"
#include "fp16.h"
#include "slicePlugin.h"
#include "interleavingPlugin.h"
#include "argsParser.h"
#include "preprocessRGB.h"

#define DIMS_C(x) x.d[0]
#define DIMS_H(x) x.d[1]
#define DIMS_W(x) x.d[2]

static samplesCommon::Args args;

const int MAX_BATCH_SIZE = 1;
int DEFAULT_CAMERA = 0;
const char* MODEL_NAME = "";
const char* INPUT_BLOB = "";
const char* OUTPUT_BLOB = "";
int IMG_WIDTH = 320;
int IMG_HEIGHT = 240;

bool signalRecieved = false;

void signalHandler( int sigNo )
{
  if( sigNo == SIGINT )
  {
    signalRecieved = true;
  }
}

Logger gLogger;

void getColor(float x, float&b, float&g, float&r) 
{
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
}


int main( int argc, char** argv )
{
  samplesCommon::parseArgs(args, argc, argv);
  IMG_WIDTH = args.width;
  IMG_HEIGHT = args.height;
  MODEL_NAME = args.engineFile.c_str();
  INPUT_BLOB = args.uffInputBlob.c_str();
  OUTPUT_BLOB = args.outputBlob.c_str();
  if( signal(SIGINT, signalHandler) == SIG_ERR )
  { 
    std::cout << "\ncan't catch SIGINT" <<std::endl;
  }
 
  gstCamera* camera = gstCamera::Create(IMG_WIDTH, IMG_HEIGHT, DEFAULT_CAMERA);
	
  if( !camera )
  {
    std::cout << "\nsegnet-camera:  failed to initialize video device" << std::endl;
    return 0;
  }
	
  std::cout << "\nsegnet-camera:  successfully initialized video device" << std::endl;
  std::cout << "    width:  " << camera->GetWidth() << std::endl;
  std::cout << "   height:  " << camera->GetHeight() << std::endl;
  std::cout << "    depth:  "<< camera->GetPixelDepth() << std::endl;
  
//Read model from file, deserialize it and create runtime, engine and exec context
  std::ifstream model( MODEL_NAME, std::ios::binary );

  std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(model), {});
  std::size_t modelSize = buffer.size() * sizeof( unsigned char );

  nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
  nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine( buffer.data(), modelSize, nullptr );
  nvinfer1::IExecutionContext* context = engine->createExecutionContext(); 

  if( !context )
  {
    std::cout << "Failed to create execution context" << std::endl;
    return 0;
  }


//Set input info + alloc memory
  const int inputIndex = engine->getBindingIndex( INPUT_BLOB );
  nvinfer1::Dims inputDims = engine->getBindingDimensions( inputIndex );

  std::size_t inputSize = MAX_BATCH_SIZE * DIMS_C(inputDims) * DIMS_H(inputDims) * DIMS_W(inputDims) * sizeof(float);
  std::size_t inputSizeUint8 = MAX_BATCH_SIZE * DIMS_C(inputDims) * DIMS_H(inputDims) * DIMS_W(inputDims) * sizeof(uint8_t);

  void* inputCPU  = nullptr;
  void* inputCUDA = nullptr;

  if( !cudaAllocMapped( (void**)&inputCPU, (void**)&inputCUDA, inputSize) )
  {
    std::cout << "Failed to alloc CUDA memory for input" << std::endl;
    return 0;
  }

//Set output info + alloc memory 
  void* outputCPU  = nullptr;
  void* outputCUDA = nullptr;


  const int outputIndex = engine->getBindingIndex( OUTPUT_BLOB );
  nvinfer1::Dims outputDims = engine->getBindingDimensions( outputIndex );
  std::size_t outputSize = MAX_BATCH_SIZE * DIMS_C(outputDims) * DIMS_H(outputDims) * 								DIMS_W(outputDims) * sizeof(float);
  std::size_t engineOutputSize = outputSize;

  if( !cudaAllocMapped( (void**)&outputCPU, (void**)&outputCUDA, outputSize) )
  {
    std::cout << "Failed to alloc CUDA memory for output" << std::endl;
    return 0;
  }

//Create CUDA stream for GPU inner data sync
  cudaStream_t stream = nullptr;
  CUDA_FAILED( cudaStreamCreateWithFlags(&stream, cudaStreamDefault ) );

  void* imgRGBCPU = NULL;
  void* imgRGB = NULL;
  if( !cudaAllocMapped((void**)&imgRGBCPU, (void**)&imgRGB, inputSize) )
  {
    printf("failed to alloc CUDA mapped memory for RGB image, %zu bytes\n", inputSize);
    return false;
  }

  void* imgRGB8CPU = NULL;
  void* imgRGB8 = NULL;
  if( !cudaAllocMapped((void**)&imgRGB8CPU, (void**)&imgRGB8, inputSizeUint8) )
  {
    printf("failed to alloc CUDA mapped memory for RGB8 image, %zu bytes\n", inputSizeUint8);
    return false;
  }


  if( !camera->Open() )
  {
    std::cout << "Failed to open camera" << std::endl;
  }

  std::cout << "Camera is open for streaming" << std::endl;

  glDisplay* display = glDisplay::Create();
  glTexture* texture = NULL;
  glTexture* texture2 = NULL;
  glTexture* texture3 = NULL;
  
  if( !display ) {
    printf("\nsegnet-camera:  failed to create openGL display\n");
  }
  else
  {
    texture = glTexture::Create(camera->GetWidth(), camera->GetHeight(), GL_LUMINANCE32F_ARB);

    if( !texture )
    {
      printf("segnet-camera:  failed to create first openGL texture\n");
    }

    texture2 = glTexture::Create(camera->GetWidth(), camera->GetHeight(), GL_RGB8);

    if( !texture2 )
    {
      printf("segnet-camera:  failed to create second openGL texture\n");
    }

    texture3 = glTexture::Create(camera->GetWidth(), camera->GetHeight(), GL_RGB8);

    if( !texture3 )
    {
      printf("segnet-camera:  failed to create second openGL texture\n");
    }

  }

  void* coloredDepth = NULL;
  void* coloredDepthCPU = NULL;
  if( !cudaAllocMapped((void**)&coloredDepthCPU, (void**)&coloredDepth, inputSizeUint8) )
  {
    printf("failed to alloc CUDA mapped memory for RGB image, %zu bytes\n", inputSizeUint8);
    return false;
  }

  void* dividedDepth = NULL;
  void* dividedDepthCPU = NULL;
  if( !cudaAllocMapped((void**)&dividedDepthCPU, (void**)&dividedDepth, outputSize) )
  {
    printf("failed to alloc CUDA mapped memory for RGB image, %zu bytes\n", outputSize);
    return false;
  }

// Main loop
  while( !signalRecieved )
  {
    auto start = std::chrono::high_resolution_clock::now();

    void* imgCPU = NULL;
    void* imgCUDA = NULL;
    void* imgRGBA = NULL;

    if( !camera->Capture(&imgCPU, &imgCUDA, 1000) )
    {
      std::cout << "Failed to capture frame" << std::endl;
    }
    auto moment_after_capture = std::chrono::high_resolution_clock::now();

    if( !camera->ConvertRGBA(imgCUDA, &imgRGBA, true) )
    {
      std::cout << "Failed to convert from NV12 to RGBA" << std::endl;
    }

    const float3 mean_value = make_float3(123.0, 115.0, 101.0);
    if( CUDA_FAILED(cudaPreImageNetMean((float4*)imgRGBA, IMG_WIDTH, IMG_HEIGHT, (float*)imgRGB, IMG_WIDTH, IMG_HEIGHT, mean_value, stream)) )
    {
      printf("cudaPreImageNet failed\n");
    }
    auto moment_after_preprocess = std::chrono::high_resolution_clock::now();
		
		void* bindings[] = {imgRGB, outputCUDA};
  	context->execute( MAX_BATCH_SIZE, bindings );
 
    auto finish = std::chrono::high_resolution_clock::now();

    //measure execution time
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Time to process single frame: " << elapsed.count() << std::endl;

    // update display
    if( display != NULL )
    {
      char str[256];
      display->SetTitle(str); 
  
      // next frame in the window
      display->UserEvents();
      display->BeginRender();

      if( texture2 != NULL )
      {
        // rescale image pixel intensities for display
        CUDA(cudaRGBA32ToRGB8((float4*)imgRGBA, (uchar3*)imgRGB8, camera->GetWidth(), camera->GetHeight()));

        // map from CUDA to openGL using GL interop
        void* tex_map = texture2->MapCUDA();

        if( tex_map != NULL )
        {
          cudaMemcpy(tex_map, imgRGB8, texture2->GetSize(), cudaMemcpyDeviceToDevice);
          texture2->Unmap();
        }
        else
        {
          std::cout << "tex_map2 = NULL" << std::endl;
        }

        // draw the texture
        texture2->Render(10 + IMG_WIDTH, 10);
      }
      else
      {
        std::cout << "Texture2 is NULL" << std::endl;
      }

      if( texture3 != NULL )
      {

        float* depth = (float*)outputCUDA;

        if (CUDA_FAILED(cudaGrayToRGB8(depth, (uchar3*)coloredDepth, camera->GetWidth(), camera->GetHeight(), 10)))
        {
          std::cout << "Failed to convert gray depth to RGB colormap" << std::endl;
        }

        // map from CUDA to openGL using GL interop
        void* tex_map = texture3->MapCUDA();

        if( tex_map != NULL )
        {
          cudaMemcpy(tex_map, coloredDepth, texture3->GetSize(), cudaMemcpyDeviceToDevice);
          texture3->Unmap();
        }
        else
        {
          std::cout << "tex_map = NULL" << std::endl;
        }

        // draw the texture
        texture3->Render(10, 10 + IMG_HEIGHT);
      }

      if( texture != NULL )
      {

        float* depth = (float*)outputCUDA;

        //for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++)
        //  depth[i] /= 10;

        if (CUDA_FAILED(cudaDivideByMaxValue(depth, (float*)dividedDepth, camera->GetWidth(), camera->GetHeight(), 10)))
        {
          std::cout << "Failed to convert gray depth to RGB colormap" << std::endl;
        }

        // map from CUDA to openGL using GL interop
        void* tex_map = texture->MapCUDA();

        if( tex_map != NULL )
        {
          cudaMemcpy(tex_map, dividedDepth, texture->GetSize(), cudaMemcpyDeviceToDevice);
          texture->Unmap();
        }
        else
        {
          std::cout << "tex_map = NULL" << std::endl;
        }

        // draw the texture
        texture->Render(10, 10);
      }

      display->EndRender();
    }
  }

  if( camera != nullptr )
  {
    camera->Close();
    delete camera;
  }

  if( display != NULL )
  {
    delete display;
    display = NULL;
  }

  //delete imgRGBAResized;
  delete imgRGB;
  return 0;
}
