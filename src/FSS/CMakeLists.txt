add_subdirectory(Partitioner)

foreach(LIB ${FSS_LIBS})
    add_library(${LIB} ${LIB}.cpp ${LIB}.h)
    target_link_libraries(${LIB} StatePartitioner ${UTIL_LIBS} ${PETSC_LIBRARIES} ${EXT_LIBRARIES})
endforeach()
