#include "plugin.h"
#include <cublas_v2.h>
#include <cudnn.h>
#include "NvInferPlugin.h"
#include "common.h"
#include "stridedSlice.h"

int StridedSlicePlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    std::cout << "enquque plugin " << this << std::endl;
    // perform nearest neighbor upsampling using cuda
    CHECK(cublasSetStream(mCublas, stream));
    CHECK(cudnnSetStream(mCudnn, stream));
    if (mDataType == DataType::kFLOAT)
        CHECK(cudaSlice<float>((float*)inputs[0], SLICE_INPUT_C, SLICE_INPUT_H, SLICE_INPUT_W, (float*)outputs[0], stream));
    else
        CHECK(cudaSlice<__half>((__half*)inputs[0], SLICE_INPUT_C, SLICE_INPUT_H, SLICE_INPUT_W, (__half*)outputs[0], stream));
    return 0;
}

IPluginV2* StridedSlicePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    std::cout << "create plugin using the creator" << std::endl;
    std::cout << "name is " << name << std::endl;
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        std::cout << "attrname: " << attrName << std::endl;
        if (!strcmp(attrName, "Index"))
        {
            //assert(fields[i].type == PluginFieldType::kINT32);
            mIndex = *(static_cast<const int*>(fields[i].data));
            std::cout << "index is: " << mIndex << std::endl;
        }
        if (!strcmp(attrName, "T"))
        {
            //assert(fields[i].type == PluginFieldType::kINT32);
            mT = *(static_cast<const int*>(fields[i].data));
            std::cout << "T is: " << mT << std::endl;
        }
    }
    return new StridedSlicePlugin(0, 0, 0);
}
