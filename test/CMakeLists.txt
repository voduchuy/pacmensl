set(TEST_PROGRAMS
        test_fss
        test_mat
        test_mat_constrained
        test_fps_cvode
        test_fsp_solver
        test_smfish
        )

foreach ( PROG ${TEST_PROGRAMS})
    add_executable(${PROG} ${PROG}.cpp)
    target_link_libraries(${PROG} Pecmeal)
    install(TARGETS ${PROG} DESTINATION ${TESTS_INSTALL_DIR})
endforeach()
