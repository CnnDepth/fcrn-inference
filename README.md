# FCRN inference

## General information

This is an utility for performing inference of neural networks on raw camera input using NVIDIA TensorRT library. It supports standard TensorRT layers and also Upsampling and Interleaving which speeds up deconvolutional blocks in fully-convolutional neural networks (see [this paper](https://arxiv.org/pdf/1606.00373.pdf) ).  It was tested with TensorRT version 5.0.2.6.

## System requirements

This utility needs:

* Linux-based system with aarch64 or x86_64 architecture
* NVIDIA graphic card

## Installation

To make this package work, you need to:  

1) Install NVIDIA graphic drivers, CUDA and CUDNN  
2) Install gcc, g++, cmake, build-essential, glib and gstreamer libraries  
3) Install TensorRT  
4) Clone this repository  
5) Create `build` subdirectory and enter it:  
`mkidr build`  
`cd build`  
6) Build the package using CMake:
`cmake .. -DPATH_TO_TENSORRT_LIB=_path_to_your_tensorrt_libraries -DPATH_TO_TENSORRT_INCLUDE=_path_to_your_tensorrt_include_files -DPATH_TO_CUDNN=_path_to_your_cudnn_library -DPATH_TO_CUBLAS=_path_to_your_cublas_library`  
Default values for paths: `PATH_TO_TENSORRT_LIB=/usr/lib/aarch64-linux-gnu, PATH_TO_TENSORRT_INCLUDE=/usr/include/aarch64-linux-gnu, PATH_TO_CUDNN=/usr/local/cuda/lib64, PATH_TO_CUBLAS=/usr/local/cuda/lib64`  
If you want to build executable code examples `main.cpp` or `fcrnEngineBuilder.cpp`, add CMake flags `-DBUILD_INFERENCE_EXAMPLE` or `-DBUILD_ENGINE_BUILDER` respectively.  
7) Run `make -j`

## Usage

### Inference

The example of usage of this utility is presented in `main.cpp` file. This example demonstrates real-time depth reconstruction on images from video camera. To run it, execute `fcrn-inference` binary.

### TensorRT engine from UFF model

The example of creating TensorRT engine from UFF model is presented in `fcrnEngineBuilder.cpp`. This example supports all native TensorRT layers and also Upsampling and Interleaving. It can be run as follows:  
`fcrn-engine-builder [-h] params`

Params:
* `--uff` - path to the UFF model you want to convert
* `--uffInput` - name of the input layer in the UFF model
* `--output` - name of the output layer in the UFF model
* `--height` - height of input and output of the UFF model
* `--width` - width of input and output of the UFF model
* `--engine` - desired path to target TensorRT engine. If not set, engine will be created and tested, but not saved on disk
* `--fp16` - whether to use FP16 mode or not
* `--processImage` - path to PPM image file to run test inference on. The result of inference will be saved into `depth.ppm` file. If not set, no test inference will be run

Execution example:
`./sample_uff_fcrn --uff=./model.uff --uffInput=Placeholder --output=MarkOutput0 --height=256 --width=256 --engine=./engine.trt --fp16 --processImage=./image.ppm`

## Notes

If you run this code on NVIDIA Jetson TX2 with TensorRT 5, you may face inversion of images from camera. You can fix it changing constant `flipMethod` to `0` in file `jetson-utils/camera/gstCamera.cpp`.
