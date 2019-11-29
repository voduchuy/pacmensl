set(TEST_PROGRAMS
        test_fss
        test_mat
        test_ode
        test_fsp_solver
#        test_smfish
#        test_sensmat
#        test_sensfsp_solver
#        test_petscwrap
#        test_stationaryfsp_solver
        )

include_directories(${GTEST_INCLUDE_DIRS})
foreach ( PROG ${TEST_PROGRAMS})
    add_executable(${PROG} ${PROG}.cpp)
    target_link_libraries(${PROG} PACMENSL ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES})
    install(TARGETS ${PROG} DESTINATION ${TESTS_INSTALL_DIR})
endforeach()

