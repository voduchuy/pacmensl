CMAKE_MINIMUM_REQUIRED(VERSION 3.3)

PROJECT(Parallel_FSP)

set(MY_LIBRARIES
        cme_util
        FiniteStateSubset
        FiniteStateSubsetLinear
        FiniteStateSubsetParMetis
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
#        solve_toggle_adaptive_FSP
        solve_hog_adaptive_FSP
#        solve_hog1p_tv_with_ts
#        solve_hog1p_tv_with_cvode
        )

set(BENCHMARKS
        benchmark_partitioning_schemes
        benchmark_scalability
        )

set(EXT_LIBRARIES
        -larmadillo
        -lsundials_cvode
        -lsundials_nvecpetsc)


list (APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake-modules")

set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpicxx )
set(CMAKE_CXX_FLAGS " -g -std=c++11 -O2 -DMAGNUS4_VERBOSE  -DHYPER_REC_OP_VERBOSE")
set(CMAKE_EXE_LINKER_FLAGS "-g ")

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
  add_library(${LIB} ${CMAKE_SOURCE_DIR}/src/${LIB}.cpp ${CMAKE_SOURCE_DIR}/include/${LIB}.h)
endforeach()

target_link_libraries(FiniteProblemSolver FiniteStateSubset FiniteStateSubsetParMetis FiniteStateSubsetLinear)
target_link_libraries(FSPSolver FiniteStateSubset FiniteProblemSolver MatrixSet CVODEFSP)

foreach ( PROG ${TEST_PROGRAMS})
  add_executable(${PROG} ${CMAKE_SOURCE_DIR}/test/${PROG}.cpp)
  target_link_libraries(${PROG} ${MY_LIBRARIES} ${PETSC_LIBRARIES} ${EXT_LIBRARIES})
endforeach()

foreach ( PROG ${EXAMPLES})
    add_executable(${PROG} ${CMAKE_SOURCE_DIR}/examples/${PROG}.cpp)
    target_link_libraries(${PROG} ${MY_LIBRARIES} ${PETSC_LIBRARIES} ${EXT_LIBRARIES})
endforeach()

foreach ( PROG ${BENCHMARKS})
    add_executable(${PROG} ${CMAKE_SOURCE_DIR}/benchmarks/${PROG}.cpp)
    target_link_libraries(${PROG} ${MY_LIBRARIES} ${PETSC_LIBRARIES} ${EXT_LIBRARIES})
endforeach()
