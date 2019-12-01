set(SOLVER_LIBS
        OdeSolverBase
        CvodeFsp
        KrylovFsp
        TsFsp
        EpicFsp
        )

set(epic_headers
        EpicHeaders/AdaptiveKrylovPhi.h
        EpicHeaders/AdaptiveKrylovPhiMultipleTimes.h
        EpicHeaders/CBlasLapack.h
        EpicHeaders/Epic.h
        EpicHeaders/EpicConst.h
        EpicHeaders/EpicSundials.h
        EpicHeaders/EpicTypes.h
        EpicHeaders/EpiRK4SC.h
        EpicHeaders/EpiRK4SV.h
        EpicHeaders/EpiRK5C.h
        EpicHeaders/EpiRK5SC.h
        EpicHeaders/EpiRK5V.h
        EpicHeaders/ExpMHigham.h
        EpicHeaders/IntegratorStats.h
        EpicHeaders/JTimesV.h
        )

foreach (LIB ${SOLVER_LIBS})
    list(APPEND ode_solver_srcs ${LIB}.cpp)
    list(APPEND ode_solver_headers ${LIB}.h)
endforeach()

add_library(ODE_OBJ OBJECT ${ode_solver_srcs} ${ode_solver_headers})

list(APPEND PACMENSL_OBJECTS "$<TARGET_OBJECTS:ODE_OBJ>")
set(PACMENSL_OBJECTS ${PACMENSL_OBJECTS} PARENT_SCOPE)

install(FILES ${ode_solver_headers} DESTINATION include/PACMENSL/)
install(FILES ${epic_headers} DESTINATION include/EpicHeaders)
