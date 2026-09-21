cmake_minimum_required(VERSION 3.20)
project(EdgeInferenceBenchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(EIB_BUILD_TESTS "Build the GoogleTest suite" ON)
option(EIB_BUILD_TOOLS "Build offline setup tools (SVM trainer)" ON)
option(EIB_ENABLE_SANITIZERS "Enable ASan/UBSan on debug builds" OFF)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(EibHelpers)

# HOGDescriptor lives in the objdetect module, so it is required in addition to
# the core/imgproc/ml/dnn modules named in the design.
find_package(OpenCV REQUIRED COMPONENTS core imgproc ml dnn objdetect)

set(EIB_CORE_SOURCES
    src/dataset/mnist_dataset.cpp
    src/classifier/classifier.cpp
    src/classifier/hog_features.cpp
    src/classifier/classical_classifier.cpp
    src/classifier/cnn_classifier.cpp
    src/classifier/classifier_factory.cpp
    src/benchmark/benchmark_config.cpp
    src/benchmark/thread_scope.cpp
    src/benchmark/metrics.cpp
    src/benchmark/runner.cpp
    src/report/console_reporter.cpp
    src/report/csv_reporter.cpp
    src/report/json_reporter.cpp
    src/platform/run_metadata.cpp
)

add_library(eib_core STATIC ${EIB_CORE_SOURCES})
target_include_directories(eib_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(eib_core PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(eib_core PUBLIC ${OpenCV_LIBS})
target_compile_features(eib_core PUBLIC cxx_std_17)
eib_apply_warnings(eib_core)

add_executable(edge_inference_benchmark apps/main.cpp)
target_link_libraries(edge_inference_benchmark PRIVATE eib_core)
eib_apply_warnings(edge_inference_benchmark)

if(EIB_BUILD_TOOLS)
    add_executable(eib_train_svm scripts/train_svm.cpp)
    target_link_libraries(eib_train_svm PRIVATE eib_core)
    eib_apply_warnings(eib_train_svm)
endif()

if(EIB_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
