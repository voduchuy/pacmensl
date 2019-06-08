list(APPEND FSS_LIBS
        StateSetBase
        StateSetConstrained
        )

add_subdirectory(Partitioner)

foreach(LIB ${FSS_LIBS})
    list(APPEND fss_srcs ${LIB}.cpp)
    list(APPEND fss_headers ${LIB}.h)
endforeach()

add_library(FSS_OBJ ${fss_srcs} ${fss_headers} $<TARGET_OBJECTS:StatePartitioner>)

install(FILES ${fss_headers} DESTINATION include/pfspat/FSS)

