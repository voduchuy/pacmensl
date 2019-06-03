set(EXT_LIBRARIES
        -larmadillo
        -lsundials_cvode
        -lsundials_nvecpetsc
        -lzoltan
        -lparmetis)
#
add_subdirectory(OdeSolver)
add_subdirectory(FSS)
add_subdirectory(Matrix)
add_subdirectory(FSP)
add_subdirectory(util)

set(FSS_LIBS ${FSS_LIBS} PARENT_SCOPE)
set(FSP_LIBS ${FSP_LIBS} PARENT_SCOPE)
set(MAT_LIBS ${MAT_LIBS} PARENT_SCOPE)
set(UTIL_LIBS ${UTIL_LIBS} PARENT_SCOPE)
