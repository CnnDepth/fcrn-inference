
file(GLOB fcrnCameraSources *.cpp upsampling/*cu upsampling/*.cpp)
file(GLOB fcrnCameraIncludes *.h upsampling/*.h ../*.h)
file(GLOB fcrnCameraCudas upsampling/*.cu)

include_directories( ../ )

#add_library(fcrnCuda ${fcrnCameraCudas})
#cuda_add_executable(fcrnCuda ${fcrnCameraCudas})
#cuda_add_library(fcrn-camera-cuda SHARED ${fcrnCameraCudas})
cuda_add_executable(fcrn-camera ${fcrnCameraSources})

target_link_libraries( fcrn-camera jetson-utils nvinfer jetson-inference cudnn /usr/local/cuda-10.0/targets/aarch64-linux/lib/libcublas.so )
#target_link_libraries(fcrn-camera jetson-inference)
#target_link_libraries(fcrn-camera fcrn-camera-cuda)

install(TARGETS fcrn-camera DESTINATION bin)
