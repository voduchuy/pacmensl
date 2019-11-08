CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(PACMENSL LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake-modules")

INCLUDE(GNUInstallDirs)

# Default build type is debug
set(default_build_type "Debug")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
            STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
            "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

if (NOT CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install CACHE PATH "Where to install the library." FORCE)
endif (NOT CMAKE_INSTALL_PREFIX)

set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-std=c++11 -O2 -DNDEBUG")

# Decide whether to build shared or static library
OPTION(BUILD_SHARED_LIBS "Build shared library" ON)

# Decide whether to build tests and where to install them
OPTION(BUILD_TESTS "Build tests" ON)
if (BUILD_TESTS)
    find_package(GTest)
    if (NOT GTEST_FOUND)
        message("GoogleTest not found on your system. Tests will not be built.")
        SET(BUILD_TESTS OFF)
    else ()
        set(TESTS_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/tests CACHE STRING "Installation directory for tests.")
    endif ()
endif ()

# Decide whether to build examples
OPTION(BUILD_EXAMPLES "Build examples" ON)
if (BUILD_EXAMPLES)
    set(EXAMPLES_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/examples CACHE STRING "Installation directory for examples.")
endif ()

# Decide whether to use PARMETIS instead of Zoltan's native Graph partitioner
OPTION(ENABLE_PARMETIS "Enable Parmetis use" ON)
if (NOT ENABLE_PARMETIS)
    add_definitions(-DNUSEPARMETIS)
endif()

FIND_PACKAGE(MPI REQUIRED)
FIND_PACKAGE(PETSc)

if (MPI_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
else ()
    message(SEND_ERROR "Cannot detect MPI CXX compiler.")
endif ()

INCLUDE_DIRECTORIES(${PETSC_INCLUDES} ${MPI_C_INCLUDE_DIRS} ${MPI_CXX_INCLUDE_DIRS})
ADD_DEFINITIONS(${PETSC_DEFINITIONS})

include_directories(./src)
include_directories(src/Partitioner)
include_directories(src/StateSet)
include_directories(src/Matrix)
include_directories(src/Fsp)
include_directories(src/OdeSolver)
include_directories(src/Sys)
include_directories(src/Models)
include_directories(src/SmFish)
include_directories(src/SensFsp)
include_directories(src/PetscWrap)
include_directories(src/StationaryFsp)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Add library
add_subdirectory(src)

# Add tests and examples
if (BUILD_TESTS)
    add_subdirectory(tests)
endif (BUILD_TESTS)

if (BUILD_EXAMPLES)
    add_subdirectory(examples)
endif (BUILD_EXAMPLES)
