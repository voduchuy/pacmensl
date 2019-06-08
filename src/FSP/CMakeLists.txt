set(FSP_LIBS
        DiscreteDistribution
        FspSolverBase
        )

set(FSP_LIBS ${FSP_LIBS} PARENT_SCOPE)

foreach ( LIB ${FSP_LIBS} )
    add_library(${LIB} SHARED ${CMAKE_SOURCE_DIR}/src/FSP/${LIB}.cpp ${CMAKE_SOURCE_DIR}/src/FSP/${LIB}.h)
    target_link_libraries(${LIB} ${MODEL_LIBS} ${FSS_LIBS} ${MAT_LIBS} ${SOLVER_LIBS} ${PETSC_LIBRARIES} ${EXT_LIBRARIES})
endforeach()
target_link_libraries(FspSolverBase DiscreteDistribution)