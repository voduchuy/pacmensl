set(STATIONARYFSP_LIBS
        StationaryFspMatrixConstrained
        StationaryMCSolver
        StationaryFspSolverMultiSinks
        )

foreach(LIB ${STATIONARYFSP_LIBS})
    list(APPEND sfsp_srcs ${LIB}.cpp)
    list(APPEND sfsp_headers ${LIB}.h)
endforeach()

add_library(STATIONARYFSP_OBJ OBJECT ${sfsp_srcs} ${sfsp_headers})

list(APPEND PACMENSL_OBJECTS "$<TARGET_OBJECTS:STATIONARYFSP_OBJ>")
set(PACMENSL_OBJECTS ${PACMENSL_OBJECTS} PARENT_SCOPE)

install(FILES ${sfsp_headers} DESTINATION include/PACMENSL/)