set(EXT_LIBRARIES
        -larmadillo
        -lsundials_cvode
        -lsundials_nvecpetsc
        -lzoltan
        -lparmetis
        ${PETSC_LIBRARIES}
        )
#
add_subdirectory(Util)
add_subdirectory(Models)
add_subdirectory(Partitioner)
add_subdirectory(StateSet)
add_subdirectory(Matrix)
add_subdirectory(OdeSolver)
add_subdirectory(FSP)
add_subdirectory(SmFish)

if (BUILD_SHARED_LIBS)
    add_library(Pecmeal SHARED pecmeal.h ${PECMEAL_OBJECTS})
    target_link_libraries(Pecmeal ${EXT_LIBRARIES})
else (BUILD_SHARED_LIBS)
    add_library(Pecmeal STATIC pecmeal.h ${PECMEAL_OBJECTS})
    target_link_libraries(Pecmeal ${EXT_LIBRARIES})
endif (BUILD_SHARED_LIBS)

## Rules for installing FSP library
SET_TARGET_PROPERTIES(Pecmeal
        PROPERTIES OUTPUT_NAME pecmeal CLEAN_DIRECT_OUTPUT 1)
install(TARGETS Pecmeal DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(FILES pecmeal_all.h DESTINATION include/pecmeal)
install(FILES pecmeal.h DESTINATION include/)