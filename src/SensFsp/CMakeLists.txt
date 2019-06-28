set(SENSFSP_LIBS
        SensFspMatrix
        ForwardSensSolverBase
        )

foreach (LIB ${SENSFSP_LIBS})
    list(APPEND sensfsp_srcs ${LIB}.cpp)
    list(APPEND sensfsp_headers ${LIB}.h)
endforeach ()

add_library(SENSFSP_OBJ OBJECT ${sensfsp_srcs} ${sensfsp_headers})

list(APPEND PACMENSL_OBJECTS "$<TARGET_OBJECTS:SENSFSP_OBJ>")
set(PACMENSL_OBJECTS ${PACMENSL_OBJECTS} PARENT_SCOPE)

install(FILES ${sensfsp_headers} DESTINATION include/PACMENSL/)
