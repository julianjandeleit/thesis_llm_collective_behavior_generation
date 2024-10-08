# Set up ARGoS compilation information
include_directories(${CMAKE_SOURCE_DIR} ${ARGOS_INCLUDE_DIRS})
link_directories(${ARGOS_LIBRARY_DIRS})
link_directories(
	${CMAKE_BINARY_DIR}/src
    )

add_library(custom_loopfunc SHARED Template.h Template.cpp)
target_link_libraries(custom_loopfunc argos3plugin_${ARGOS_BUILD_FOR}_epuck)


