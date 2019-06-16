add_library(SMFISH_OBJ OBJECT SmFishSnapshot.h SmFishSnapshot.cpp)

list(APPEND PECMEAL_OBJECTS "$<TARGET_OBJECTS:SMFISH_OBJ>")
set(PECMEAL_OBJECTS ${PECMEAL_OBJECTS} PARENT_SCOPE)

install(FILES SmFishSnapshot.h DESTINATION include/pecmeal/)
