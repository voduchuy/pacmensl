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
    target_link_libraries(${PROG} PfspatLib ${PETSC_LIBRARIES} ${EXT_LIBRARIES} ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
endforeach()
