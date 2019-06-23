list(APPEND UTIL_LIBS Sys)

set(util_srcs)
foreach( LIB ${UTIL_LIBS})
    list(APPEND util_srcs ${LIB}.cpp)
    list(APPEND util_headers ${LIB}.h)
endforeach()

add_library(CME_UTIL_OBJ OBJECT ${util_srcs})
list(APPEND PACMENSL_OBJECTS "$<TARGET_OBJECTS:CME_UTIL_OBJ>")
set(PACMENSL_OBJECTS ${PACMENSL_OBJECTS} PARENT_SCOPE)
install(FILES ${util_headers} DESTINATION include/PACMENSL)



