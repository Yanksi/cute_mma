# Find the cutlass library
# Create a custom target to build all executables
if(DEFINED ENV{CUTLASS_DIR})
    set(CUTLASS_DIR $ENV{CUTLASS_DIR})
    message(STATUS "Using CUTLASS from environment variable CUTLASS_DIR: ${CUTLASS_DIR}")
    if(NOT EXISTS ${CUTLASS_DIR}/include)
        message(FATAL_ERROR "include not found in the specified CUTLASS_DIR")
    endif()
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
# Set the CUTLASS_INCLUDE_DIRS variable, include both the include and tools directories
set(CUTLASS_INCLUDE_DIRS
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/tools/util/include
)