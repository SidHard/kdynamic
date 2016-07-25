cmake_minimum_required(VERSION 2.8)

project(dynamic_fusion)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

SET( CMAKE_BUILD_TYPE Debug )

set(THREADS_USE_PTHREADS_WIN32 true)
find_package(Threads)
find_package(CUDA REQUIRED)
find_package(TooN REQUIRED)
find_package(OpenGL)
find_package(Freenect)
find_package(GLUT)
find_package(OpenCV)
find_package(SuiteSparse REQUIRED)


include_directories(
	${CMAKE_BINARY_DIR}/include
	${TOON_INCLUDE_PATHS}
)
include_directories(${EIGEN_INCLUDE_DIRS})
include_directories(${SUITESPARSE_INCLUDE_DIRS})

set(cuda_srcs
	helpers.cu
	helpers.h
	fusion.cu
	fusion.h
        deformationGraph.cpp
        deformationGraph.h
	)
	
cuda_add_library(dynamic_fusion STATIC
	${cuda_srcs}
	OPTIONS
	-O3
	)

if (GLUT_FOUND)

    if(FREENECT_FOUND)
        include_directories(${FREENECT_INCLUDE_PATHS} ${THREADS_PTHREADS_INCLUDE_DIR})
        include_directories(${OpenCV_INCLUDE_DIRS})
        add_definitions(-DLIBFREENECT_INTERFACE)
        add_executable(dfusion_kinect
            kinect.cpp thirdparty/interface.h thirdparty/interface.cpp
            )
        target_link_libraries(dfusion_kinect dynamic_fusion ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES} ${FREENECT_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${OpenCV_LIBS} ${Eigen_LIBRARIES} ${SUITESPARSE_LIBRARIES})
    endif()

endif()
