set(TEST_PROGRAMS
        test_fss
        test_mat
        test_mat_constrained
        test_fps_cvode
        test_fsp_solver
#        test_dd
        )

foreach ( PROG ${TEST_PROGRAMS})
    add_executable(${PROG} ${PROG}.cpp)
    target_link_libraries(${PROG} ${UTIL_LIBS} ${FSS_LIBS} ${MAT_LIBS} ${FSP_LIBS} ${PETSC_LIBRARIES} ${EXT_LIBRARIES})
endforeach()
