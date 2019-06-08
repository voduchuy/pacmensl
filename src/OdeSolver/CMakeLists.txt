set(SOLVER_LIBS
        OdeSolverBase
        CvodeFsp
        )

foreach (LIB ${SOLVER_LIBS})
    list(APPEND ode_solver_srcs ${LIB}.cpp)
    list(APPEND ode_solver_headers ${LIB}.h)
endforeach()

add_library(ODE_OBJ OBJECT ${ode_solver_srcs} ${ode_solver_headers})

install(FILES ${ode_solver_headers} DESTINATION include/pfspat/OdeSolver/)
