list(APPEND FSS_LIBS
        StateSet
        StateSetBase
        StateSetConstrained
        )

foreach(LIB ${FSS_LIBS})
    list(APPEND fss_srcs ${LIB}.cpp)
    list(APPEND fss_headers ${LIB}.h)
endforeach()

add_library(FSS_OBJ OBJECT ${fss_srcs} ${fss_headers})

list(APPEND PACMENSL_OBJECTS "$<TARGET_OBJECTS:FSS_OBJ>")
set(PACMENSL_OBJECTS ${PACMENSL_OBJECTS} PARENT_SCOPE)

install(FILES ${fss_headers} DESTINATION include/PACMENSL/)

