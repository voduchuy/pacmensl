list(APPEND UTIL_LIBS cme_util)

set (UTIL_LIBS ${UTIL_LIBS} PARENT_SCOPE)

foreach( LIB ${UTIL_LIBS})
    add_library(${LIB} SHARED ${CMAKE_SOURCE_DIR}/src/util/${LIB}.cpp ${CMAKE_SOURCE_DIR}/src/util/${LIB}.h)
    target_link_libraries(${LIB}  ${PETSC_LIBRARIES} ${EXT_LIBRARIES})
endforeach()
