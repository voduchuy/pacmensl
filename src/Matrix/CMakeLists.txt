set(MAT_LIBS
        FspMatrixBase
        FspMatrixConstrained
        )

foreach( LIB ${MAT_LIBS})
    list(APPEND mat_srcs ${LIB}.cpp)
    list(APPEND mat_headers ${LIB}.h)
endforeach()

add_library(MAT_OBJ OBJECT ${mat_srcs} ${mat_headers})

list(APPEND PACMENSL_OBJECTS "$<TARGET_OBJECTS:MAT_OBJ>")
set(PACMENSL_OBJECTS ${PACMENSL_OBJECTS} PARENT_SCOPE)

install(FILES ${mat_headers} DESTINATION include/PACMENSL/)