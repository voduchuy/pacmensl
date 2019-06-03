list(APPEND SOLVER_LIBS
        OdeSolverBase
        CVODEFSP
        )

set(SOLVER_LIBS ${SOLVER_LIBS} PARENT_SCOPE)

add_library(OdeSolverBase OdeSolverBase.cpp OdeSolverBase.h)
target_link_libraries(OdeSolverBase ${UTIL_LIBS} ${FSS_LIBS} ${PETSC_LIBRARIES} ${EXT_LIBRARIES})

add_library(CVODEFSP cvode_interface/CVODEFSP.cpp cvode_interface/CVODEFSP.h)
target_link_libraries(CVODEFSP ${UTIL_LIBS} ${FSS_LIBS} ${PETSC_LIBRARIES} ${EXT_LIBRARIES})
