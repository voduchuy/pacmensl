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

install(FILES ${partitioner_headers} DESTINATION include/pfspat/FSS/Partitioner)
