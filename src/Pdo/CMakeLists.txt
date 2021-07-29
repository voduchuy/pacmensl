set(PDO_LIBS
        Pdo
        )

foreach (LIB ${PDO_LIBS})
    list(APPEND pdo_srcs ${LIB}.cpp)
    list(APPEND pdo_headers ${LIB}.h)
endforeach ()

add_library(PDO_OBJ OBJECT ${pdo_srcs} ${pdo_headers})

list(APPEND PACMENSL_OBJECTS "$<TARGET_OBJECTS:PDO_OBJ>")
set(PACMENSL_OBJECTS ${PACMENSL_OBJECTS} PARENT_SCOPE)

install(FILES ${pdo_headers} DESTINATION include/PACMENSL/)
