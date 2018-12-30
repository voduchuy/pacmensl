CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(Parallel_FSP)

set(UTIL_LIBS
        cme_util
        )

set(FSS_LIBS
        FiniteStateSubset
        FiniteStateSubsetZoltanQuery
        FiniteStateSubsetNaive
        FiniteStateSubsetGraph
        FiniteStateSubsetHyperGraph
        FiniteStateSubsetHierarch
        FiniteStateSubsetRCB
        )

set(MAT_LIBS
        MatrixSet
        )

set(SOLVER_LIBS
        FiniteProblemSolver
        CVODEFSP
        )

set(FSP_LIBRARIES
        FSPSolver
        )

set(TEST_PROGRAMS
        test_fss
        test_mat
        test_fps_cvode
        test_fsp_solver
        test_ao
        )

set(EXAMPLES
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

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpicxx )
set(CMAKE_CXX_FLAGS " -g -std=c++11")

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

foreach( LIB ${UTIL_LIBS})
    add_library(${LIB} SHARED ${CMAKE_SOURCE_DIR}/src/util/${LIB}.cpp ${CMAKE_SOURCE_DIR}/src/util/${LIB}.h)
    target_link_libraries(${LIB}  ${PETSC_LIBRARIES} ${EXT_LIBRARIES})
endforeach()

foreach( LIB ${FSS_LIBS})
    add_library(${LIB} ${CMAKE_SOURCE_DIR}/src/FSS/${LIB}.cpp ${CMAKE_SOURCE_DIR}/src/FSS/${LIB}.h)
    target_link_libraries(${LIB} ${UTIL_LIBS} ${PETSC_LIBRARIES} ${EXT_LIBRARIES})
endforeach()

foreach( LIB ${MAT_LIBS})
    add_library(${LIB} ${CMAKE_SOURCE_DIR}/src/Matrix/${LIB}.cpp ${CMAKE_SOURCE_DIR}/src/Matrix/${LIB}.h)
    target_link_libraries(${LIB} ${UTIL_LIBS} ${FSS_LIBS} ${PETSC_LIBRARIES} ${EXT_LIBRARIES})
endforeach()

foreach( LIB ${SOLVER_LIBS})
    add_library(${LIB} ${CMAKE_SOURCE_DIR}/src/FPSolver/${LIB}.cpp ${CMAKE_SOURCE_DIR}/src/FPSolver/${LIB}.h)
    target_link_libraries(${LIB} ${UTIL_LIBS} ${FSS_LIBS} ${PETSC_LIBRARIES} ${EXT_LIBRARIES})
endforeach()

foreach ( LIB ${FSP_LIBRARIES} )
  add_library(${LIB} SHARED ${CMAKE_SOURCE_DIR}/src/FSP/${LIB}.cpp ${CMAKE_SOURCE_DIR}/src/FSP/${LIB}.h)
  target_link_libraries(${LIB} ${FSS_LIBS} ${MAT_LIBS} ${SOLVER_LIBS} ${PETSC_LIBRARIES} ${EXT_LIBRARIES})
endforeach()

foreach ( PROG ${TEST_PROGRAMS})
  add_executable(${PROG} ${CMAKE_SOURCE_DIR}/test/${PROG}.cpp)
  target_link_libraries(${PROG} ${FSP_LIBRARIES} ${PETSC_LIBRARIES} ${EXT_LIBRARIES})
endforeach()

foreach ( PROG ${EXAMPLES})
    add_executable(${PROG} ${CMAKE_SOURCE_DIR}/examples/${PROG}.cpp)
    target_link_libraries(${PROG} ${FSP_LIBRARIES} ${PETSC_LIBRARIES} ${EXT_LIBRARIES})
endforeach()

