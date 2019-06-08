

set(EXAMPLES
        signal_activated_bursting
        distribute_states
        hog1p
#        repressilator
#        transcr_reg_6d
        )

set(EXAMPLES ${EXAMPLES} PARENT_SCOPE)

foreach ( PROG ${EXAMPLES})
    add_executable(${PROG} ${CMAKE_SOURCE_DIR}/examples/${PROG}.cpp)
    target_link_libraries(${PROG} ${FSP_LIBS} ${PETSC_LIBRARIES} ${EXT_LIBRARIES})
endforeach()