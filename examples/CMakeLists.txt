

set(EXAMPLES
        distribute_states
        hog1p
        repressilator
        transcr_reg_6d
        )

foreach ( PROG ${EXAMPLES})
    add_executable(${PROG} ${CMAKE_SOURCE_DIR}/examples/${PROG}.cpp)
    target_link_libraries(${PROG} ${FSP_LIBRARIES} ${PETSC_LIBRARIES} ${EXT_LIBRARIES})
endforeach()