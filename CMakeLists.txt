# rayane

cmake_minimum_required(VERSION 3.16)
project(projet_M1_RCF LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB SOURCES src/*.cpp)
add_executable(projet_M1_RCF ${SOURCES})
target_include_directories(projet_M1_RCF PRIVATE include)
