cmake_minimum_required(VERSION 2.8)

PROJECT(bmreg)

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
 
add_executable(bmreg bmreg.cxx LinkedBlockList.cpp graph.cpp maxflow.cpp)
 
target_link_libraries(bmreg ${ITK_LIBRARIES})