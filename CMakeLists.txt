cmake_minimum_required(VERSION 3.5)

set(TARGET_NAME mlake)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})

include_directories(src/include)

set(EXTENSION_SOURCES
    src/mlake_extension.cpp
    src/mlake_bridge.cpp
    src/mlake_arrow_scan.cpp
    src/mlake_audio.cpp)

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

# Where an installed build looks for the Mojo bridge when MLAKE_BRIDGE is not
# set. It is a default and not a link-time dependency on purpose: the
# extension loads with dlopen, so a .duckdb_extension built here still loads on
# a machine where the bridge lives somewhere else.
if(NOT MLAKE_BRIDGE_PATH)
  if(APPLE)
    set(MLAKE_BRIDGE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/mojo/build/libmlake_bridge.dylib")
  else()
    set(MLAKE_BRIDGE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/mojo/build/libmlake_bridge.so")
  endif()
endif()

foreach(target ${EXTENSION_NAME} ${LOADABLE_EXTENSION_NAME})
  target_compile_definitions(${target} PRIVATE MLAKE_BRIDGE_DEFAULT_PATH="${MLAKE_BRIDGE_PATH}")
  target_link_libraries(${target} ${CMAKE_DL_LIBS})
endforeach()

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
