cmake_minimum_required(VERSION 3.18)
project(cute_mma LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 80)

set (CMAKE_C_FLAGS_DEBUG "$ENV{CFLAGS} -O0 -g2 -ggdb")
set (CMAKE_C_FLAGS_RELEASE "$ENV{CFLAGS} -O3")

set (CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g2 -ggdb")
set (CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3")

set (CMAKE_EXE_LINKER_FLAGS "-Wl,--as-needed")

find_package (CUDAToolkit REQUIRED)

set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fopenmp --expt-relaxed-constexpr")
if (${CMAKE_BUILD_TYPE} MATCHES "Debug")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_${CMAKE_CUDA_ARCHITECTURES} -g -lineinfo -Xptxas=-v -O0")
else ()
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_${CMAKE_CUDA_ARCHITECTURES},code=sm_${CMAKE_CUDA_ARCHITECTURES} --use_fast_math -O3")
endif ()

# Create a custom target to build all executables
if(DEFINED ENV{CUTLASS_DIR})
    set(CUTLASS_DIR $ENV{CUTLASS_DIR})
else()
    message(STATUS "Environment variable CUTLASS_DIR is not set. Cloning from GitHub...")
    execute_process(
        COMMAND git clone https://github.com/NVIDIA/cutlass.git ${CMAKE_BINARY_DIR}/cutlass
        RESULT_VARIABLE GIT_CLONE_RESULT
    )
    if(NOT GIT_CLONE_RESULT EQUAL "0")
        message(FATAL_ERROR "Failed to clone CUTLASS repository")
    endif()
    set(CUTLASS_DIR ${CMAKE_BINARY_DIR}/cutlass)
endif()

set(ARGPARSE_DIR $ENV{ARGPARSE_DIR})
if(DEFINED ENV{ARGPARSE_DIR})
    set(ARGPARSE_DIR $ENV{ARGPARSE_DIR})
else()
    message(STATUS "Environment variable ARGPARSE_DIR is not set. Cloning from GitHub...")
    execute_process(
        COMMAND git clone https://github.com/p-ranav/argparse.git ${CMAKE_BINARY_DIR}/argparse
        RESULT_VARIABLE GIT_CLONE_RESULT
    )
    if(NOT GIT_CLONE_RESULT EQUAL "0")
        message(FATAL_ERROR "Failed to clone argparse repository")
    endif()
    set(ARGPARSE_DIR ${CMAKE_BINARY_DIR}/argparse)
endif()

# List of common include directories for CUTLASS
set(COMMON_INCLUDE_DIRS
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/tools/util/include
    ${ARGPARSE_DIR}/include
)

if (DEFINED AUTOTUNE)
    set(COMMON_DIR "autotune_configs")
    foreach (dtype IN ITEMS half float)
        file(GLOB CONFIG_DIRS "autotune_configs/${dtype}/*")
        foreach(layout IN ITEMS NT TN)
            set(GROUP_NAME "${dtype}_${layout}")
            add_custom_target(${GROUP_NAME})
            foreach(CONFIG_DIR ${CONFIG_DIRS})
                get_filename_component(CONFIG_NAME ${CONFIG_DIR} NAME)
                set(TARGET_NAME "gemm_${dtype}_${CONFIG_NAME}_${layout}")

                add_executable(${TARGET_NAME} gemm_sm80_tc.cu)
                target_include_directories(${TARGET_NAME} PRIVATE ${CONFIG_DIR} ${COMMON_INCLUDE_DIRS} ${COMMON_DIR})
                target_include_directories(${TARGET_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
                target_compile_definitions(${TARGET_NAME} PRIVATE LAYOUT_${layout})
                set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${dtype}/${layout})
                add_dependencies(${GROUP_NAME} ${TARGET_NAME})
            endforeach()
        endforeach()
    endforeach()
else()
    set(COMMON_DIR "best_configs")
    add_custom_target(best)
    foreach(dtype IN ITEMS half float)
        set(TARGET_NAME "gemm_${dtype}")
        add_executable(${TARGET_NAME} gemm_sm80_tc.cu)
        target_include_directories(${TARGET_NAME} PRIVATE ${COMMON_INCLUDE_DIRS} ${COMMON_DIR})
        target_include_directories(${TARGET_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
        target_include_directories(${TARGET_NAME} PRIVATE best_configs/${dtype})
        target_link_libraries(${TARGET_NAME} PRIVATE CUDA::cublas)
        target_compile_definitions(${TARGET_NAME} PRIVATE LAYOUT_TN)
        target_compile_definitions(${TARGET_NAME} PRIVATE LAYOUT_NT)
        target_compile_definitions(${TARGET_NAME} PRIVATE USE_CUBLAS)
        add_dependencies(best ${TARGET_NAME})
    endforeach()
endif()
