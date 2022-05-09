cmake_minimum_required(VERSION 2.8)

PROJECT(bmreg)

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
 
add_executable(bmreg bmreg.cxx LinkedBlockList.cpp graph.cpp maxflow.cpp)
 
target_link_libraries(bmreg ${ITK_LIBRARIES})