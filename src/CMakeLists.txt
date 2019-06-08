set(EXT_LIBRARIES
        -larmadillo
        -lsundials_cvode
        -lsundials_nvecpetsc
        -lzoltan
        -lparmetis)
#
add_subdirectory(util)
add_subdirectory(Models)
add_subdirectory(FSS)
add_subdirectory(Matrix)
add_subdirectory(OdeSolver)
add_subdirectory(FSP)

add_library(PfspatLib SHARED )
target_link_libraries(PfspatLib FSS_OBJ MAT_OBJ FSP_OBJ CME_UTIL_OBJ MODEL_OBJ ODE_OBJ ${PETSC_LIBRARIES} ${EXT_LIBRARIES})

## Rules for installing FSP library
SET_TARGET_PROPERTIES(PfspatLib
        PROPERTIES OUTPUT_NAME pfspat CLEAN_DIRECT_OUTPUT 1)
install(TARGETS PfspatLib DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(FILES pfspat_all.h DESTINATION include/pfspat)
install(FILES pfspat.h DESTINATION include/)