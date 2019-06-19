set(PART_LIBS
        StatePartitionerBase
        StatePartitionerGraph
        StatePartitionerHyperGraph
        )

foreach(LIB ${PART_LIBS})
    list(APPEND partitioner_srcs ${LIB}.cpp)
    list(APPEND partitioner_headers ${LIB}.h)
endforeach()

add_library(StatePartitioner OBJECT StatePartitioner.h StatePartitioner.cpp ${partitioner_srcs} ${partitioner_headers})
list(APPEND PACMENSL_OBJECTS "$<TARGET_OBJECTS:StatePartitioner>")
set(PACMENSL_OBJECTS ${PACMENSL_OBJECTS} PARENT_SCOPE)
install(FILES ${partitioner_headers} StatePartitioner.h DESTINATION include/PACMENSL/)
