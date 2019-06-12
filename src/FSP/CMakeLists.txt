set(FSP_LIBS
        DiscreteDistribution
        FspSolverBase
        )

foreach(LIB ${FSP_LIBS})
    list(APPEND fsp_srcs ${LIB}.cpp)
    list(APPEND fsp_headers ${LIB}.h)
endforeach()

add_library(FSP_OBJ OBJECT ${fsp_srcs} ${fsp_headers})
target_link_libraries(FSP_OBJ ${EXT_LIBRARIES})

list(APPEND PECMEAL_OBJECTS "$<TARGET_OBJECTS:FSP_OBJ>")
set(PECMEAL_OBJECTS ${PECMEAL_OBJECTS} PARENT_SCOPE)

install(FILES ${fsp_headers} DESTINATION include/pecmeal/)
