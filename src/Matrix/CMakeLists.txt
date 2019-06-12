set(MAT_LIBS
        FspMatrixBase
        FspMatrixConstrained
        )

foreach( LIB ${MAT_LIBS})
    list(APPEND mat_srcs ${LIB}.cpp)
    list(APPEND mat_headers ${LIB}.h)
endforeach()

add_library(MAT_OBJ OBJECT ${mat_srcs} ${mat_headers})
target_link_libraries(MAT_OBJ FSS_OBJ)

list(APPEND PECMEAL_OBJECTS "$<TARGET_OBJECTS:MAT_OBJ>")
set(PECMEAL_OBJECTS ${PECMEAL_OBJECTS} PARENT_SCOPE)

install(FILES ${mat_headers} DESTINATION include/pecmeal/)