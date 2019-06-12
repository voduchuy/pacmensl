set(SOLVER_LIBS
        OdeSolverBase
        CvodeFsp
        KrylovFsp
        )

foreach (LIB ${SOLVER_LIBS})
    list(APPEND ode_solver_srcs ${LIB}.cpp)
    list(APPEND ode_solver_headers ${LIB}.h)
endforeach()

add_library(ODE_OBJ OBJECT ${ode_solver_srcs} ${ode_solver_headers})

list(APPEND PECMEAL_OBJECTS "$<TARGET_OBJECTS:ODE_OBJ>")
set(PECMEAL_OBJECTS ${PECMEAL_OBJECTS} PARENT_SCOPE)

install(FILES ${ode_solver_headers} DESTINATION include/pecmeal/)
