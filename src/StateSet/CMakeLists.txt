list(APPEND FSS_LIBS
        StateSetBase
        StateSetConstrained
        )

foreach(LIB ${FSS_LIBS})
    list(APPEND fss_srcs ${LIB}.cpp)
    list(APPEND fss_headers ${LIB}.h)
endforeach()

add_library(FSS_OBJ OBJECT ${fss_srcs} ${fss_headers} $<TARGET_OBJECTS:StatePartitioner>)

list(APPEND PECMEAL_OBJECTS "$<TARGET_OBJECTS:FSS_OBJ>")
set(PECMEAL_OBJECTS ${PECMEAL_OBJECTS} PARENT_SCOPE)

install(FILES ${fss_headers} DESTINATION include/pecmeal/)

