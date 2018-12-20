CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(Parallel_FSP)

set(MY_LIBRARIES
        cme_util
        FiniteStateSubset
        FiniteStateSubsetNaive
        FiniteStateSubsetGraph
        FiniteStateSubsetHyperGraph
        MatrixSet
        FiniteProblemSolver
        CVODEFSP
        FSPSolver
        )# KExpv Magnus4 Magnus4FSP  )

set(TEST_PROGRAMS
        test_fss
        test_mat
        test_fps_cvode
        test_fsp_solver
        test_ao
        )

set(EXAMPLES
        solve_hog_adaptive_FSP
        distribute_states
        benchmark
        multiple_single_core_jobs
        )

set(EXT_LIBRARIES
        -larmadillo
        -lsundials_cvode
        -lsundials_nvecpetsc
        -lzoltan
        -lparmetis)


list (APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake-modules")

set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpicxx )
set(CMAKE_CXX_FLAGS " -g -std=c++11 -O3")

FIND_PACKAGE(PETSc)
INCLUDE_DIRECTORIES(${PETSC_INCLUDES})
ADD_DEFINITIONS(${PETSC_DEFINITIONS})


# Set the directory to put our compiled files
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)

# Where to put .a files (archive for static libraries)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Where to put .so files (dynamic libraries)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Where to put the executable files
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Where to find CMakelists.txt
#set(CMAKE_SOURCE_DIR $ENV{MYDIR}/C++/PETScFSP)

# Set the folders to include
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/include)

foreach ( LIB ${MY_LIBRARIES} )
  add_library(${LIB} SHARED ${CMAKE_SOURCE_DIR}/src/${LIB}.cpp ${CMAKE_SOURCE_DIR}/include/${LIB}.h)
  target_link_libraries(${LIB} ${PETSC_LIBRARIES} ${EXT_LIBRARIES})
endforeach()

target_link_libraries(MatrixSet FiniteStateSubset ${PETSC_LIBRARIES} ${EXT_LIBRARIES})
target_link_libraries( FiniteStateSubsetNaive FiniteStateSubset)
target_link_libraries( FiniteStateSubsetGraph FiniteStateSubset)
target_link_libraries( FiniteStateSubsetHyperGraph FiniteStateSubset)
target_link_libraries(FiniteProblemSolver FiniteStateSubset FiniteStateSubsetHyperGraph FiniteStateSubsetGraph FiniteStateSubsetNaive ${PETSC_LIBRARIES} ${EXT_LIBRARIES})
target_link_libraries(CVODEFSP FiniteProblemSolver)
target_link_libraries(FSPSolver FiniteStateSubset FiniteProblemSolver MatrixSet CVODEFSP ${PETSC_LIBRARIES} ${EXT_LIBRARIES})

foreach ( PROG ${TEST_PROGRAMS})
  add_executable(${PROG} ${CMAKE_SOURCE_DIR}/test/${PROG}.cpp)
  target_link_libraries(${PROG} ${MY_LIBRARIES} ${PETSC_LIBRARIES} ${EXT_LIBRARIES})
endforeach()

foreach ( PROG ${EXAMPLES})
    add_executable(${PROG} ${CMAKE_SOURCE_DIR}/examples/${PROG}.cpp)
    target_link_libraries(${PROG} ${MY_LIBRARIES} ${PETSC_LIBRARIES} ${EXT_LIBRARIES})
endforeach()

