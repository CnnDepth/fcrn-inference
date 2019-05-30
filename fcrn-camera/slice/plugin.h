#ifndef __STRIDED_SLICE_PLUGIN_H__
#define __STRIDED_SLICE_PLUGIN_H__

#include <cublas_v2.h>
#include <cudnn.h>
#include <cstdio>
#include "NvInferPlugin.h"
#include "common.h"
#include "stridedSlice.h"

const int SLICE_INPUT_C = 1024;
const int SLICE_INPUT_H = 16;
const int SLICE_INPUT_W = 20;

class StridedSlicePlugin : public IPluginV2
{
public:
    StridedSlicePlugin(int a, int b, int c)
    {
        std::cout << "Init " << this << " from dims" << std::endl;  
        /*mNbInputChannels = nbInputChannels;
        mInputWidth = inputWidth;
        mInputHeight = inputHeight;
        std::cout << "input dims: " << mNbInputChannels << ' ' << mInputHeight << ' ' << mInputWidth << std::endl;*/
        //printf("inputs: %d %d %d %d %d %d %d\n", a, b, c, d, e, f, g);
    }

    StridedSlicePlugin(const Weights *weights, size_t nbWeights)
    {
        std::cout << "Init " << this << " from weights" << std::endl;
        std::cout << "I have " << nbWeights << " weights" << endl;
        std::cout << "weights ptr is " << weights << std::endl;
    }

    StridedSlicePlugin(const void* data, size_t length)
    {
        std::cout << "Init " << this << " from data and length" << std::endl;
        const char* d = static_cast<const char*>(data), *a = d;
        read(d, mNbInputChannels);
        read(d, mInputWidth);
        read(d, mInputHeight);
        read(d, mDataType);
        std::cout << "Readed input width " << mInputWidth << std::endl;
        assert(d == a + length);
    }

    ~StridedSlicePlugin()
    {
        std::cout << "delete plugin " << this << std::endl;
    }

    int getNbOutputs() const override
    {
        std::cout << "get number of outputs of " << this << std::endl;
        return 1;
    }

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        std::cout << "Slice: Get output dimensions of " << this << std::endl;
        std::cout << "input dims are: " << inputs[0].d[0] << ' ' << inputs[0].d[1] << ' ' << inputs[0].d[2] << std::endl;
        assert(index == 0 && nbInputDims == 3 && inputs[0].nbDims == 3);
        std::cout << "Dims of first  input: " << inputs[0].d[0] << ' ' << inputs[0].d[1] << ' ' << inputs[0].d[2] << std::endl;
        std::cout << "Dims of second  input: " << inputs[1].d[0] << ' ' << inputs[1].d[1] << ' ' << inputs[1].d[2] << std::endl;
        std::cout << "Dims of third  input: " << inputs[2].d[0] << ' ' << inputs[2].d[1] << ' ' << inputs[2].d[2] << std::endl;
        mNbInputChannels = inputs[0].d[0];
        mInputHeight = inputs[0].d[1];
        mInputWidth = inputs[0].d[2];
        std::cout << "set input width " << mInputWidth << std::endl;
        return nvinfer1::Dims3(inputs[0].d[0], inputs[0].d[1] - 1, inputs[0].d[2]);
    }

    bool supportsFormat(DataType type, PluginFormat format) const override 
    { 
        std::cout << "supports format? " << this << std::endl;
        return (type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kNCHW; 
    }

    void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override
    {
        std::cout << "configure " << this << " with format" << std::endl;
        assert(supportsFormat(type, format));
        mDataType = type;
    }

    int initialize() override
    {
        std::cout << "Initialize plugin " << this << std::endl;
        CHECK(cudnnCreate(&mCudnn));// initialize cudnn and cublas
        CHECK(cublasCreate(&mCublas));
        return 0;
    }

    virtual void terminate() override
    {
        std::cout << "terminate plugin " << this << std::endl;
        CHECK(cublasDestroy(mCublas));
        CHECK(cudnnDestroy(mCudnn));
        // write below code for custom variables
    }

    virtual size_t getWorkspaceSize(int maxBatchSize) const override
    {
        std::cout << "get workspace size of " << this << std::endl;
        return 0;
    }

    virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    virtual size_t getSerializationSize() const override
    {
        // 3 size_t values: input width, input height, and number of channels
        // and one more value for data type
        return sizeof(DataType) + 3 * sizeof(int);
    }

    virtual void serialize(void* buffer) const override
    {
        std::cout << "serialize " << this << std::endl;
        char* d = static_cast<char*>(buffer), *a = d;

        std::cout << "this is " << this << std::endl;
        std::cout << "Write input width " << mInputWidth << std::endl;
        write(d, mNbInputChannels);
        write(d, mInputWidth);
        write(d, mInputHeight);
        write(d, mDataType);
        assert(d == a + getSerializationSize());
    }

    const char* getPluginType() const override 
    { 
        std::cout << "get type of " << this << std::endl;
        return "Slice";
    }

    const char* getPluginVersion() const override 
    { 
        std::cout << "get version of " << this << std::endl;
        return "1";
    }

    void destroy() override { delete this; }

    IPluginV2* clone() const override
    {
        return new StridedSlicePlugin(0, 0, 0);
    }

    void setPluginNamespace(const char* libNamespace) override { mNamespace = libNamespace; }

    const char* getPluginNamespace() const override { return mNamespace.c_str(); }

private:
    size_t type2size(DataType type) { return type == DataType::kFLOAT ? sizeof(float) : sizeof(__half); }

    template<typename T> void write(char*& buffer, const T& val) const
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template<typename T> T read(const char*& buffer, T& val) const
    {
        val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
        return val;
    }

    void* copyToDevice(const void* data, size_t count)
    {
        void* deviceData;
        CHECK(cudaMalloc(&deviceData, count));
        CHECK(cudaMemcpy(deviceData, data, count, cudaMemcpyHostToDevice));
        return deviceData;
    }

    void deserializeToDevice(const char*& hostBuffer, void*& deviceWeights, size_t size)
    {
        deviceWeights = copyToDevice(hostBuffer, size);
        hostBuffer += size;
    }

    DataType mDataType{DataType::kFLOAT};
    cudnnHandle_t mCudnn;
    cublasHandle_t mCublas;
    int mNbInputChannels=0, mInputWidth=0, mInputHeight=0;
    std::string mNamespace = "";
};


class StridedSlicePluginCreator: public IPluginCreator
{
public:
    StridedSlicePluginCreator()
    {
        std::cout << "Create plugin creator" << std::endl;
        mPluginAttributes.emplace_back(PluginField("nbInputChannels", nullptr, PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(PluginField("inputHeight", nullptr, PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(PluginField("inputWidth", nullptr, PluginFieldType::kINT32, 1));

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    ~StridedSlicePluginCreator() {}

    const char* getPluginName() const override 
    {
        std::cout << "get plugin name" << std::endl;
        return "Slice"; 
    }

    const char* getPluginVersion() const override
    {
        std::cout << "get plugin version" << std::endl;
        return "1";
    }

    const PluginFieldCollection* getFieldNames() override { return &mFC; }

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override
    {
        std::cout << "deserialize plugin using the creator" << std::endl;
        //This object will be deleted when the network is destroyed, which will
        //call Concat::destroy()
        return new StridedSlicePlugin(serialData, serialLength);
    }

    void setPluginNamespace(const char* libNamespace) override { mNamespace = libNamespace; }

    const char* getPluginNamespace() const override { return mNamespace.c_str(); }
private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace = "";
    int mIndex, mT;
};
#endif