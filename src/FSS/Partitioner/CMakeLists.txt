set(PART_LIBS
        StatePartitionerBase
        StatePartitionerGraph
        StatePartitionerHyperGraph
        )

foreach(LIB ${PART_LIBS})
    add_library(${LIB} ${LIB}.cpp ${LIB}.h)
    target_link_libraries(${LIB} ${UTIL_LIBS} ${PETSC_LIBRARIES} ${EXT_LIBRARIES})
endforeach()

add_library(StatePartitioner StatePartitioner.cpp StatePartitioner.h)
target_link_libraries(StatePartitioner ${PART_LIBS})