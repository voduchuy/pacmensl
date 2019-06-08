set(MAT_LIBS
        FspMatrixBase
        FspMatrixConstrained
        )

foreach( LIB ${MAT_LIBS})
    list(APPEND mat_srcs ${LIB}.cpp)
    list(APPEND mat_headers ${LIB}.h)
endforeach()

add_library(MAT_OBJ OBJECT ${mat_srcs} ${mat_headers})

install(FILES ${mat_headers} DESTINATION include/pfspat/Matrix)