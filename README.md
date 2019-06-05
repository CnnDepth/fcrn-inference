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
5) Compile this package by following commands:  
`mkidr build`  
`cd build`  
`cmake .. -DPATH_TO_TENSORRT_LIB=_path_to_your_tensorrt_libraries -DPATH_TO_TENSORRT_INCLUDE=_path_to_your_tensorrt_include_files -DPATH_TO_CUDNN=_path_to_your_cudnn_library -DPATH_TO_CUBLAS=_path_to_your_cublas_library`  
Default values for paths: `PATH_TO_TENSORRT_LIB=/usr/lib/aarch64-linux-gnu, PATH_TO_TENSORRT_INCLUDE=/usr/include/aarch64-linux-gnu, PATH_TO_CUDNN=/usr/local/cuda/lib64, PATH_TO_CUBLAS=/usr/local/cuda/lib64`  
`make -j`

## Usage

The example of usage of this utility is presented in `main.cpp` file. This example demonstrates real-time depth reconstruction on images from video camera. To run it, execute `fcrn-inference` binary.

## Notes

If you run this code on NVIDIA Jetson TX2 with TensorRT 5, you may face inversion of images from camera. You can fix it changing constant `flipMethod` to `0` in file `jetson-utils/camera/gstCamera.cpp`.
