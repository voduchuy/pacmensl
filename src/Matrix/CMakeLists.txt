set(MAT_LIBS
        FspMatrixBase
        FspMatrixConstrained
        )

set(MAT_LIBS ${MAT_LIBS} PARENT_SCOPE)

foreach( LIB ${MAT_LIBS})
    add_library(${LIB} ${CMAKE_SOURCE_DIR}/src/Matrix/${LIB}.cpp ${CMAKE_SOURCE_DIR}/src/Matrix/${LIB}.h)
    target_link_libraries(${LIB} ${UTIL_LIBS} ${FSS_LIBS} ${PETSC_LIBRARIES} ${EXT_LIBRARIES})
endforeach()