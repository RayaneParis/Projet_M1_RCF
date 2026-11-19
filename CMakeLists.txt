# mehdi and rayane

cmake_minimum_required(VERSION 3.16)
project(projet_M1_RCF LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(projet_M1_RCF
    src/main.cpp
    src/experiments.cpp
    src/dataset.cpp
    src/distance.cpp
    src/knn.cpp
    src/vote.cpp
    src/svm.cpp
    src/kmeans.cpp
    src/evaluation.cpp
    src/curve.cpp
)

find_package(OpenCV REQUIRED COMPONENTS core ml)
target_link_libraries(projet_M1_RCF PRIVATE ${OpenCV_LIBS})
target_include_directories(projet_M1_RCF PRIVATE include ${OpenCV_INCLUDE_DIRS})