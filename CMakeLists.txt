CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(PFSPAT LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake-modules")
INCLUDE(GNUInstallDirs)

set(default_build_type "Debug")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
            STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
            "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

set(CMAKE_CXX_FLAGS "-std=c++11")
FIND_PACKAGE(PETSc REQUIRED)
FIND_PACKAGE(MPI REQUIRED)

if (MPI_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
else ()
    message(SEND_ERROR "Cannot detect MPI CXX compiler.")
endif ()

INCLUDE_DIRECTORIES(${PETSC_INCLUDES} ${MPI_C_INCLUDE_DIRS} ${MPI_CXX_INCLUDE_DIRS})
ADD_DEFINITIONS(${PETSC_DEFINITIONS})

include_directories(./src)
include_directories(src/FSS)
include_directories(src/Matrix)
include_directories(src/FSP)
include_directories(src/OdeSolver)
include_directories(src/util)
include_directories(src/Models)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Add library
add_subdirectory(src)

# Add tests and examples
add_subdirectory(test)
add_subdirectory(examples)
