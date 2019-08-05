

set(EXAMPLES
        signal_activated_bursting
        hog1p
        time_invariant_bursting
        repressilator
        transcr_reg_6d
        hog1p_matvec
        simple_petsc_program
        )

set(EXAMPLES ${EXAMPLES} PARENT_SCOPE)

foreach ( PROG ${EXAMPLES})
    add_executable(${PROG} ${CMAKE_SOURCE_DIR}/examples/${PROG}.cpp)
    target_link_libraries(${PROG} PACMENSL)
    install(TARGETS ${PROG} DESTINATION ${EXAMPLES_INSTALL_DIR})
endforeach()