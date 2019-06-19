set(FSP_LIBS
        DiscreteDistribution
        FspSolverBase
        )

foreach(LIB ${FSP_LIBS})
    list(APPEND fsp_srcs ${LIB}.cpp)
    list(APPEND fsp_headers ${LIB}.h)
endforeach()

add_library(FSP_OBJ OBJECT ${fsp_srcs} ${fsp_headers})

list(APPEND PACMENSL_OBJECTS "$<TARGET_OBJECTS:FSP_OBJ>")
set(PACMENSL_OBJECTS ${PACMENSL_OBJECTS} PARENT_SCOPE)

install(FILES ${fsp_headers} DESTINATION include/PACMENSL/)
