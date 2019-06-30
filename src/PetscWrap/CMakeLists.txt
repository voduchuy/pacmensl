list(APPEND PETSCWRAP_LIBS PetscWrap)

foreach( LIB ${PETSCWRAP_LIBS})
    list(APPEND pw_srcs ${LIB}.cpp)
    list(APPEND pw_headers ${LIB}.h)
endforeach()

add_library(PWRAP_OBJ OBJECT ${pw_srcs})
list(APPEND PACMENSL_OBJECTS "$<TARGET_OBJECTS:PWRAP_OBJ>")
set(PACMENSL_OBJECTS ${PACMENSL_OBJECTS} PARENT_SCOPE)
install(FILES ${pw_headers} DESTINATION include/PACMENSL)



