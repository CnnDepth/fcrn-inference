
file(GLOB fcrnCameraSources *.cpp upsampling/*cu upsampling/*.cpp interleaving/*.cu interleaving/*.cpp slice/*.cu slice/*.cpp)
file(GLOB fcrnCameraIncludes *.h upsampling/*.h interleaving/*.h slice/*.h)

cuda_add_library(fcrn-camera SHARED ${fcrnCameraSources})
target_link_libraries( fcrn-camera jetson-utils common ${PATH_TO_TENSORRT_LIB}/libnvinfer.so jetson-inference ${PATH_TO_CUDNN}/libcudnn.so ${PATH_TO_CUBLAS}/libcublas.so )

# transfer all headers to the include directory
foreach(include ${fcrnCameraIncludes})
	message("-- Copying ${include}")
	configure_file(${include} ${PROJECT_INCLUDE_DIR} COPYONLY)
endforeach()

install(TARGETS fcrn-camera DESTINATION lib EXPORT fcrnCameraConfig)
install(EXPORT fcrnCameraConfig DESTINATION share/fcrnCamera/cmake)
