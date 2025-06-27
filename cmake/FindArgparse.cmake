if(DEFINED ENV{ARGPARSE_DIR})
    set(ARGPARSE_DIR $ENV{ARGPARSE_DIR})
    message(STATUS "Using argparse from environment variable ARGPARSE_DIR: ${ARGPARSE_DIR}")
    if(NOT EXISTS ${ARGPARSE_DIR}/include)
        message(FATAL_ERROR "include not found in the specified ARGPARSE_DIR")
    endif()
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

# Set the ARGPARSE_INCLUDE_DIRS variable
set(ARGPARSE_INCLUDE_DIRS
    ${ARGPARSE_DIR}/include
)