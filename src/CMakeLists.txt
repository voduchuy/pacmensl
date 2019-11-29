set(EXT_LIBRARIES
        -larmadillo
        -lsundials_cvodes
        -lsundials_nvecpetsc
        -lzoltan
        ${PETSC_LIBRARIES}
        )

if (ENABLE_PARMETIS)
    list(APPEND EXT_LIBRARIES -lparmetis)
else()
    add_definitions(-DNUSEPARMETIS)
endif()

#
add_subdirectory(Sys)
add_subdirectory(PetscWrap)
add_subdirectory(Models)
add_subdirectory(Partitioner)
add_subdirectory(StateSet)
add_subdirectory(Matrix)
add_subdirectory(OdeSolver)
add_subdirectory(Fsp)
add_subdirectory(SmFish)
add_subdirectory(SensFsp)
add_subdirectory(StationaryFsp)

if (BUILD_SHARED_LIBS)
    add_library(PACMENSL SHARED ${PACMENSL_OBJECTS})
    target_link_libraries(PACMENSL ${EXT_LIBRARIES})
else (BUILD_SHARED_LIBS)
    add_library(PACMENSL STATIC ${PACMENSL_OBJECTS})
    target_link_libraries(PACMENSL ${EXT_LIBRARIES})
endif (BUILD_SHARED_LIBS)

## Rules for installing Fsp library
SET_TARGET_PROPERTIES(PACMENSL
        PROPERTIES OUTPUT_NAME pacmensl CLEAN_DIRECT_OUTPUT 1)
install(TARGETS PACMENSL DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(FILES pacmensl_all.h DESTINATION include/PACMENSL)
install(FILES pacmensl.h DESTINATION include/)