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

list(APPEND PACMENSL_OBJECTS "$<TARGET_OBJECTS:ODE_OBJ>")
set(PACMENSL_OBJECTS ${PACMENSL_OBJECTS} PARENT_SCOPE)

install(FILES ${ode_solver_headers} DESTINATION include/PACMENSL/)
