
file(GLOB fcrnCameraSources *.cpp upsampling/*cu upsampling/*.cpp interleaving/*.cu interleaving/*.cpp slice/*.cu slice/*.cpp)
file(GLOB fcrnCameraIncludes *.h upsampling/*.h ../*.h interleaving/*.h slice/*.h)

include_directories( ../ )

cuda_add_executable(fcrn-camera ${fcrnCameraSources})

target_link_libraries( fcrn-camera jetson-utils nvinfer jetson-inference /usr/local/cuda-9.0/targets/x86_64-linux/lib/libcudnn.so /usr/local/cuda-9.0/targets/x86_64-linux/lib/libcublas.so )

install(TARGETS fcrn-camera DESTINATION bin)
