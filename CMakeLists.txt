CMAKE_MINIMUM_REQUIRED(VERSION 3.3)

PROJECT(Parallel_FSP)

list (APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake-modules")

set(CMAKE_C_COMPILER mpicc)
set(CMAKE_Fortran_COMPILER mpif90)
set(CMAKE_Fortran_FLAGS " -m64 -O3")
set(CMAKE_CXX_COMPILER mpicxx )
set(CMAKE_CXX_FLAGS " -std=c++11 -O3 -DNDEBUG -g")
set(CMAKE_EXE_LINKER_FLAGS " -lgfortran -lm ")



# Set the directory to put our compiled files
set(CMAKE_BINARY_DIR $ENV{MYDIR}/C++/PETScFSP)

# Where to put .a files (archive for static libraries)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Where to put .so files (dynamic libraries)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Where to put the executable files
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Where to find CMakelists.txt
set(CMAKE_SOURCE_DIR $ENV{MYDIR}/C++/PETScFSP)

# Set the folders to include
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/include)

FIND_PACKAGE(PETSc)

INCLUDE_DIRECTORIES(${PETSC_INCLUDES})
ADD_DEFINITIONS(${PETSC_DEFINITIONS})

set(MY_LIBRARIES fsp_matrix HyperRecOp KExpv Magnus4 cme_util)
set(TEST_PROGRAMS form_toggle_matrix solve_toggle form_partition_toggle_matrix form_toggle_operator solve_toggle_tv solve_hog1p_tv)

foreach ( LIB ${MY_LIBRARIES} )
  add_library(${LIB} ${CMAKE_SOURCE_DIR}/src/${LIB}.cc ${CMAKE_SOURCE_DIR}/include/${LIB}.hpp )
endforeach()

foreach ( PROG ${TEST_PROGRAMS})
  add_executable(${PROG} ${CMAKE_SOURCE_DIR}/test/${PROG}.cc)
  target_link_libraries(${PROG} ${MY_LIBRARIES} ${PETSC_LIBRARIES} -larmadillo)
endforeach()
