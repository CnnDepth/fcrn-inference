
file(GLOB commonSources *.cpp *.cu)
file(GLOB commonIncludes *.h)

cuda_add_library(common SHARED ${commonSources})
target_link_libraries(common jetson-utils nvinfer jetson-inference cudnn /usr/local/cuda-10.0/targets/aarch64-linux/lib/libcublas.so )

# transfer all headers to the include directory
foreach(include ${commonIncludes})
	message("-- Copying ${include}")
	configure_file(${include} ${PROJECT_INCLUDE_DIR} COPYONLY)
endforeach()

install(TARGETS common DESTINATION lib EXPORT commonConfig)
install(EXPORT commonConfig DESTINATION share/common/cmake)
