cmake_minimum_required(VERSION 3.18)
project(cute_mma LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 80)

# Create a custom target to build all executables
set(CUTLASS_DIR $ENV{CUTLASS_DIR})

# List of common include directories for CUTLASS
set(CUTLASS_INCLUDE_DIRS
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/tools/util/include
)

if (DEFINED AUTOTUNE)
    foreach (dtype IN ITEMS half float)
        file(GLOB CONFIG_DIRS "autotune_configs/${dtype}/*")
        foreach(layout IN ITEMS NT TN)
            set(GROUP_NAME "${dtype}_${layout}")
            add_custom_target(${GROUP_NAME})
            foreach(CONFIG_DIR ${CONFIG_DIRS})
                get_filename_component(CONFIG_NAME ${CONFIG_DIR} NAME)
                set(TARGET_NAME "gemm_${dtype}_${CONFIG_NAME}_${layout}")

                add_executable(${TARGET_NAME} gemm_sm80_tc.cu)
                target_include_directories(${TARGET_NAME} PRIVATE ${CONFIG_DIR} ${CUTLASS_INCLUDE_DIRS})
                target_compile_definitions(${TARGET_NAME} PRIVATE LAYOUT_${layout})
                set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${dtype}/${layout})
                add_dependencies(${GROUP_NAME} ${TARGET_NAME})
            endforeach()
        endforeach()
    endforeach()
else()
    add_custom_target(best)
    foreach(dtype IN ITEMS half float)
        set(TARGET_NAME "gemm_${dtype}")
        add_executable(${TARGET_NAME} gemm_sm80_tc.cu)
        target_include_directories(${TARGET_NAME} PRIVATE ${CUTLASS_INCLUDE_DIRS})
        target_include_directories(${TARGET_NAME} PRIVATE best_configs/${dtype})
        target_compile_definitions(${TARGET_NAME} PRIVATE LAYOUT_TN)
        target_compile_definitions(${TARGET_NAME} PRIVATE LAYOUT_NT)
        add_dependencies(best ${TARGET_NAME})
    endforeach()
endif()
