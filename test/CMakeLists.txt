set(TEST_PROGRAMS
        test_fss
#        test_mat
#        test_fps_cvode
#        test_fsp_solver
#        test_dd
        )

foreach ( PROG ${TEST_PROGRAMS})
    add_executable(${PROG} ${PROG}.cpp)
    target_link_libraries(${PROG} ${UTIL_LIBS} ${FSS_LIBS} ${PETSC_LIBRARIES} ${EXT_LIBRARIES})
    message(STATUS "${FSS_LIBS}")
endforeach()
