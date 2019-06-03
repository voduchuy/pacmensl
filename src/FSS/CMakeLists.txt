list(APPEND FSS_LIBS
        StateSetBase
        StateSetConstrained
        )

set (FSS_LIBS ${FSS_LIBS} PARENT_SCOPE)

add_subdirectory(Partitioner)

foreach(LIB ${FSS_LIBS})
    add_library(${LIB} ${LIB}.cpp ${LIB}.h)
    target_link_libraries(${LIB} StatePartitioner ${UTIL_LIBS} ${PETSC_LIBRARIES} ${EXT_LIBRARIES})
endforeach()
