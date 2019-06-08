list(APPEND UTIL_LIBS cme_util)

set(util_srcs)
foreach( LIB ${UTIL_LIBS})
    list(APPEND util_srcs ${LIB}.cpp)
    list(APPEND util_headers ${LIB}.h)
endforeach()

add_library(CME_UTIL_OBJ ${util_srcs})
install(FILES ${util_headers} DESTINATION include/pfspat/util)



