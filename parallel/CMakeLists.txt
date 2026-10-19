cmake_minimum_required(VERSION 3.20)
project(parallel_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(PARALLEL_CUDA     "Build the llama.cpp backend with CUDA" ON)
option(PARALLEL_BUILD_APPS     "Build the CLI demo" ON)
option(PARALLEL_BUILD_TESTS    "Build tests" ON)
option(PARALLEL_BUILD_EXAMPLES "Build examples" ON)

# Pinned llama.cpp. Point this at a checkout of the pinned revision.
set(PARALLEL_LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../vendor/llama.cpp"
    CACHE PATH "Path to the pinned llama.cpp checkout")

if(NOT EXISTS "${PARALLEL_LLAMA_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "llama.cpp not found at ${PARALLEL_LLAMA_DIR}. Clone the pinned revision there, or pass -DPARALLEL_LLAMA_DIR=<path>.")
endif()

# Backend configuration. Keep backend-specific types out of public headers.
set(GGML_CUDA ${PARALLEL_CUDA} CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)

add_subdirectory("${PARALLEL_LLAMA_DIR}" "${CMAKE_BINARY_DIR}/_llama.cpp" EXCLUDE_FROM_ALL)

add_library(parallel STATIC
    src/runtime.cpp
    src/workflows/compact.cpp
    src/workflows/reasoning.cpp
)
target_include_directories(parallel
    PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(parallel PUBLIC llama)

if(PARALLEL_BUILD_APPS)
    add_executable(parallel_cli apps/parallel_cli.cpp)
    target_link_libraries(parallel_cli PRIVATE parallel)
endif()

if(PARALLEL_BUILD_EXAMPLES)
    foreach(ex answer_and_memory parallel_reasoning)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/${ex}.cpp")
            add_executable(${ex} "examples/${ex}.cpp")
            target_link_libraries(${ex} PRIVATE parallel)
        endif()
    endforeach()
endif()

if(PARALLEL_BUILD_TESTS)
    enable_testing()
    add_executable(test_runtime tests/test_runtime.cpp)
    target_link_libraries(test_runtime PRIVATE parallel)
    add_test(NAME test_runtime COMMAND test_runtime)
endif()
