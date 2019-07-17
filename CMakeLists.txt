cmake_minimum_required(VERSION 3.8.1)
project(ecgFilter)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(clBP)

find_package(OpenCV REQUIRED)

find_package(Boost REQUIRED)

add_executable(ecgFilter main.cpp bandpass.cpp parameters.h cvui.h bandpass.h)
target_link_libraries(ecgFilter clBP iir_static fir ${OpenCV_LIBS})