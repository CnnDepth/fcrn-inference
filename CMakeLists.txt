cmake_minimum_required(VERSION 2.8)
project(jetson-inference)

# submodule warning
message(" ")
message("Note:  this project uses git submodules in the source tree.")
message("       if you haven't already, run the following command from")
message("       the project's root directory:")
message(" ")
message("           git submodule update --init") 
message("\n")

# find TensorRT package
if ( NOT DEFINED PATH_TO_TENSORRT_LIB )
	set(PATH_TO_TENSORRT_LIB /usr/lib/aarch64-linux-gnu)
endif()
if ( NOT DEFINED PATH_TO_TENSORRT_INCLUDE )
	set(PATH_TO_TENSORRT_INCLUDE /usr/include/aarch64-linux-gnu)
endif()
message("TensorRT libraries path: ${PATH_TO_TENSORRT_LIB}")
message("TensorRT includes path: ${PATH_TO_TENSORRT_INCLUDE}")

if ( NOT EXISTS "${PATH_TO_TENSORRT_LIB}" )
	message(FATAL_ERROR "TensorRT libraries not found. Set correct PATH_TO_TENSORRT_LIB")
endif()
if ( NOT EXISTS "${PATH_TO_TENSORRT_INCLUDE}" )
	message(FATAL_ERROR "TensorRT includes not found. Set correct PATH_TO_TENSORRT_INCLUDE")
endif()

# find CuDNN
if ( NOT DEFINED PATH_TO_CUDNN )
	set( CUDNN_PATHS /usr/local/cuda/lib64 /usr/lib/aarch64-linux-gnu )
	foreach(path ${CUDNN_PATHS})
		if ( EXISTS ${path}/libcudnn.so )
			set( PATH_TO_CUDNN ${path} )
		endif()
	endforeach()
endif()
if ( EXISTS ${PATH_TO_CUDNN}/libcudnn.so )
	message("CuDNN found in ${PATH_TO_CUDNN}")
else()
	message("CuDNN not found. Set correct PATH_TO_CUDNN")
endif()

# find CuBLAS
if ( NOT DEFINED PATH_TO_CUBLAS )
	set( PATH_TO_CUBLAS /usr/local/cuda/lib64 )
endif()
if ( EXISTS ${PATH_TO_CUBLAS}/libcublas.so )
	message("CuBLAS found in ${PATH_TO_CUBLAS}")
else()
	message("Could not find ${PATH_TO_CUBLAS}/libcublas.so. Set correct PATH_TO_CUDNN")
endif()


if( NOT EXISTS "${PROJECT_SOURCE_DIR}/jetson-utils/.git" )
	message("Note:  required git submodules have not been detected.")
	message("       first, please run the following command from the")
	message("       the project's root directory to clone them:")
	message(" ")
	message("          git submodule update --init")
	message(" ")
	message(FATAL_ERROR "missing required git submodules, see instructions above")
endif()


# setup tensorRT flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")	# -std=gnu++11
set(BUILD_DEPS "YES" CACHE BOOL "If YES, will install dependencies into sandbox.  Automatically reset to NO after dependencies are installed.")


# if this is the first time running cmake, perform pre-build dependency install script (or if the user manually triggers re-building the dependencies)
#if( ${BUILD_DEPS} )
#	message("-- Launching pre-build dependency installer script...")
#
#	execute_process(COMMAND sh ../CMakePreBuild.sh
#				WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
#				RESULT_VARIABLE PREBUILD_SCRIPT_RESULT)
#
#	set(BUILD_DEPS "NO" CACHE BOOL "If YES, will install dependencies into sandbox.  Automatically reset to NO after dependencies are installed." FORCE)
#	message("-- Finished installing dependencies")
#endif()


# setup CUDA
find_package(CUDA)
message("-- CUDA version: ${CUDA_VERSION}")

set(
	CUDA_NVCC_FLAGS
	${CUDA_NVCC_FLAGS}; 
    -O3 
    -gencode arch=compute_30,code=sm_30
    -gencode arch=compute_35,code=sm_35
    -gencode arch=compute_37,code=sm_37
    -gencode arch=compute_50,code=sm_50
    -gencode arch=compute_52,code=sm_52
	-gencode arch=compute_53,code=sm_53
	-gencode arch=compute_60,code=sm_60
	-gencode arch=compute_61,code=sm_61
	-gencode arch=compute_62,code=sm_62
	-gencode arch=compute_70,code=sm_70
)

# setup CUDNN
#find_package(cudnn)
#message(" -- cudnn version: ${CUDNN_VERSION}")

if(CUDA_VERSION_MAJOR GREATER 9)
	message("-- CUDA ${CUDA_VERSION_MAJOR} detected, enabling SM_72")

	set(
		CUDA_NVCC_FLAGS
		${CUDA_NVCC_FLAGS}; 
		-gencode arch=compute_72,code=sm_72
		-gencode arch=compute_75,code=sm_75
	)

	# OpenCV used for findHomography() and decomposeHomography()
	# OpenCV version >= 3.0.0 required for decomposeHomography()
	find_package(OpenCV 3.0.0 COMPONENTS core calib3d REQUIRED)
endif()


# setup project output paths
set(PROJECT_OUTPUT_DIR  ${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_PROCESSOR})
set(PROJECT_INCLUDE_DIR ${PROJECT_OUTPUT_DIR}/include)

file(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR})
file(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)

message("-- system arch:  ${CMAKE_SYSTEM_PROCESSOR}")
message("-- output path:  ${PROJECT_OUTPUT_DIR}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)


# build C/C++ interface
include_directories(${PROJECT_INCLUDE_DIR} ${GIE_PATH}/include)
include_directories(/usr/include/gstreamer-1.0 /usr/lib/aarch64-linux-gnu/gstreamer-1.0/include /usr/include/glib-2.0 /usr/lib/x86_64-linux-gnu/gstreamer-1.0/include /usr/include/glib-2.0 /usr/include/libxml2 /usr/lib/aarch64-linux-gnu/glib-2.0/include /usr/lib/x86_64-linux-gnu/glib-2.0/include ${PATH_TO_TENSORRT_INCLUDE})

#file(GLOB inferenceSources *.cpp *.cu)
#file(GLOB inferenceIncludes *.h)

#cuda_add_library(jetson-inference SHARED ${inferenceSources})
#target_link_libraries(jetson-inference nvcaffe_parser nvinfer)		# gstreamer-0.10 gstbase-0.10 gstapp-0.10 


# transfer all headers to the include directory
foreach(include ${inferenceIncludes})
	message("-- Copying ${include}")
	configure_file(${include} ${PROJECT_INCLUDE_DIR} COPYONLY)
endforeach()

add_subdirectory(jetson-utils)
add_subdirectory(common)
add_subdirectory(fcrn-camera)

#create executable example
if ( DEFINED BUILD_INFERENCE_EXAMPLE )
	cuda_add_executable(fcrn-inference main.cpp)
	target_link_libraries( fcrn-inference jetson-utils ${PATH_TO_TENSORRT_LIB}/libnvinfer.so common fcrn-camera ${PATH_TO_CUDNN}/libcudnn.so ${PATH_TO_CUBLAS}/libcublas.so ${PATH_TO_TENSORRT_LIB}/libnvcaffe_parser.so)
endif()

if ( DEFINED BUILD_ENGINE_BUILDER )
	cuda_add_executable( fcrn-engine-builder fcrnEngineBuilder.cpp )
	target_link_libraries( fcrn-engine-builder jetson-utils ${PATH_TO_TENSORRT_LIB}/libnvinfer.so common fcrn-camera ${PATH_TO_CUDNN}/libcudnn.so ${PATH_TO_CUBLAS}/libcublas.so ${PATH_TO_TENSORRT_LIB}/libnvcaffe_parser.so )
endif()


# install includes
foreach(include ${inferenceIncludes})
    install(FILES "${include}" DESTINATION include/jetson-inference)
endforeach()

# install symlink to networks
install(CODE "execute_process( COMMAND ${CMAKE_COMMAND} -E create_symlink ${PROJECT_SOURCE_DIR}/data/networks ${CMAKE_INSTALL_PREFIX}/bin/networks )" )

