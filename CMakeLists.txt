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
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_${CMAKE_CUDA_ARCHITECTURES} -g -G -lineinfo -Xptxas=-v -O0")
else ()
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_${CMAKE_CUDA_ARCHITECTURES},code=sm_${CMAKE_CUDA_ARCHITECTURES} --use_fast_math -O3")
endif ()

# Create a custom target to build all executables
if(DEFINED ENV{CUTLASS_DIR})
    set(CUTLASS_DIR $ENV{CUTLASS_DIR})
else()
    set(CUTLASS_DIR ${CMAKE_BINARY_DIR}/cutlass)
    if(NOT EXISTS ${CUTLASS_DIR})
        message(STATUS "Environment variable CUTLASS_DIR is not set. Cloning from GitHub...")
        execute_process(
            COMMAND git clone https://github.com/NVIDIA/cutlass.git ${CUTLASS_DIR}
            RESULT_VARIABLE GIT_CLONE_RESULT
        )
        if(NOT GIT_CLONE_RESULT EQUAL "0")
            message(FATAL_ERROR "Failed to clone CUTLASS repository")
        endif()
    else()
        message(STATUS "CUTLASS repository already exists at ${CUTLASS_DIR}")
    endif()
endif()

if(DEFINED ENV{ARGPARSE_DIR})
    set(ARGPARSE_DIR $ENV{ARGPARSE_DIR})
else()
    set(ARGPARSE_DIR ${CMAKE_BINARY_DIR}/argparse)
    if(NOT EXISTS ${ARGPARSE_DIR})
        message(STATUS "Environment variable ARGPARSE_DIR is not set. Cloning from GitHub...")
        execute_process(
            COMMAND git clone https://github.com/p-ranav/argparse.git ${ARGPARSE_DIR}
            RESULT_VARIABLE GIT_CLONE_RESULT
        )
        if(NOT GIT_CLONE_RESULT EQUAL "0")
            message(FATAL_ERROR "Failed to clone argparse repository")
        endif()
    else()
        message(STATUS "argparse repository already exists at ${ARGPARSE_DIR}")
    endif()
endif()

# List of common include directories for CUTLASS
set(COMMON_INCLUDE_DIRS
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/tools/util/include
    ${ARGPARSE_DIR}/include
)

add_executable(playground cute_playground.cu)
target_include_directories(playground PRIVATE ${COMMON_INCLUDE_DIRS})

add_executable(cute_oft cute_oft.cu)
target_include_directories(cute_oft PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(cute_oft PRIVATE CUDA::cublas)
target_compile_definitions(cute_oft PRIVATE USE_CUBLAS)
if (${CMAKE_BUILD_TYPE} MATCHES "Debug")
    target_compile_definitions(cute_oft PRIVATE DEBUG)
endif()

# for each .cu file in folder "playgrounds", create an executable
file(GLOB PLAYGROUND_SOURCES "playgrounds/*.cu")
foreach(PLAYGROUND_SOURCE ${PLAYGROUND_SOURCES})
    get_filename_component(PLAYGROUND_NAME ${PLAYGROUND_SOURCE} NAME_WE)
    add_executable(${PLAYGROUND_NAME} ${PLAYGROUND_SOURCE})
    target_include_directories(${PLAYGROUND_NAME} PRIVATE ${COMMON_INCLUDE_DIRS})
    target_link_libraries(${PLAYGROUND_NAME} PRIVATE CUDA::cublas)
endforeach()



# set(dtypes_o half float)
# set(dtypes_r half float)

# # Ensure both dtypes_o and dtypes_r are the same length
# list(LENGTH dtypes_o dtypes_o_len)
# list(LENGTH dtypes_r dtypes_r_len)
# if (NOT dtypes_o_len EQUAL dtypes_r_len)
#     message(FATAL_ERROR "Length of dtypes_o and dtypes_r must be the same")
# endif()
# math(EXPR dtypes_o_len_minus_1 "${dtypes_o_len} - 1")

# if (DEFINED AUTOTUNE AND AUTOTUNE)
#     foreach (i RANGE 0 ${dtypes_o_len_minus_1})
#         list(GET dtypes_o ${i} dtype_o)
#         list(GET dtypes_r ${i} dtype_r)
#         message(STATUS "Building for ${dtype_o} and ${dtype_r}")
#         foreach(layout IN ITEMS NT TN)
#             file(GLOB CONFIG_DIRS "autotune_configs/${dtype_o}_${dtype_r}/${layout}/*")
#             set(GROUP_NAME "${dtype_o}_${dtype_r}_${layout}")
#             add_custom_target(${GROUP_NAME})
#             foreach(CONFIG_DIR ${CONFIG_DIRS})
#                 get_filename_component(CONFIG_NAME ${CONFIG_DIR} NAME)
#                 set(TARGET_NAME "gemm_${dtype_o}_${dtype_r}_${CONFIG_NAME}_${layout}")

#                 add_executable(${TARGET_NAME} gemm_sm80_tc.cu)
#                 target_include_directories(${TARGET_NAME} PRIVATE ${CONFIG_DIR} ${COMMON_INCLUDE_DIRS})
#                 target_include_directories(${TARGET_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
#                 target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
#                 target_compile_definitions(${TARGET_NAME} PRIVATE LAYOUT_${layout})
#                 set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${dtype_o}_${dtype_r}/${layout})
#                 target_compile_definitions(${TARGET_NAME} PRIVATE CUTE_MMA_DTYPE_O=${dtype_o})
#                 target_compile_definitions(${TARGET_NAME} PRIVATE CUTE_MMA_DTYPE_R=${dtype_r})
#                 add_dependencies(${GROUP_NAME} ${TARGET_NAME})
#             endforeach()
#         endforeach()
#     endforeach()
# else()
#     add_custom_target(default)
#     foreach (i RANGE 0 ${dtypes_o_len_minus_1})
#         list(GET dtypes_o ${i} dtype_o)
#         list(GET dtypes_r ${i} dtype_r)
#         message(STATUS "Building for ${dtype_o} and ${dtype_r}")
#         set(TARGET_NAME "gemm_${dtype_o}_${dtype_r}")
#         add_executable(${TARGET_NAME} gemm_sm80_tc.cu)
#         target_include_directories(${TARGET_NAME} PRIVATE ${COMMON_INCLUDE_DIRS})
#         target_include_directories(${TARGET_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
#         target_include_directories(${TARGET_NAME} PRIVATE default_configs)
#         target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
#         target_link_libraries(${TARGET_NAME} PRIVATE CUDA::cublas)
#         target_compile_definitions(${TARGET_NAME} PRIVATE LAYOUT_TN)
#         target_compile_definitions(${TARGET_NAME} PRIVATE LAYOUT_NT)
#         target_compile_definitions(${TARGET_NAME} PRIVATE CUTE_MMA_DTYPE_O=${dtype_o})
#         target_compile_definitions(${TARGET_NAME} PRIVATE CUTE_MMA_DTYPE_R=${dtype_r})
#         target_compile_definitions(${TARGET_NAME} PRIVATE USE_CUBLAS)
#         if (${CMAKE_BUILD_TYPE} MATCHES "Debug")
#             target_compile_definitions(${TARGET_NAME} PRIVATE DEBUG)
#         endif()
#         add_dependencies(default ${TARGET_NAME})
#     endforeach()
# endif()