CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(Parallel_FSP LANGUAGES CXX)

list (APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake-modules")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_COMPILER mpicxx )
set(CMAKE_CXX_FLAGS " -g -std=c++11 -O2 -fexceptions")

FIND_PACKAGE(PETSc)
INCLUDE_DIRECTORIES(${PETSC_INCLUDES})
ADD_DEFINITIONS(${PETSC_DEFINITIONS})

include_directories(./src)
include_directories(src/FSS)
include_directories(src/Matrix)
include_directories(src/FSP)
include_directories(src/OdeSolver)
include_directories(src/util)

# Set the directory to put our compiled files
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)

# Where to put .a files (archive for static libraries)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Where to put .so files (dynamic libraries)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Where to put the executable files
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set(FSS_LIBS
        StateSetBase
        StateSetConstrained
        )

set(UTIL_LIBS
        cme_util
        )

# Add library
add_subdirectory(src)

# Add tests and examples
add_subdirectory(test)
#add_subdirectory(examples)
