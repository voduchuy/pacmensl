set(FSP_LIBS
        DiscreteDistribution
        FspSolverBase
        )

foreach(LIB ${FSP_LIBS})
    list(APPEND fsp_srcs ${LIB}.cpp)
    list(APPEND fsp_headers ${LIB}.h)
endforeach()

add_library(FSP_OBJ ${fsp_srcs} ${fsp_headers})

install(FILES ${fsp_headers} DESTINATION include/pfspat/FSP)
