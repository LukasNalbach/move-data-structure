cmake_minimum_required(VERSION 3.21)

# the move data structure: a disjoint interval sequence with move queries in constant time, and its balancing
project(move_data_structure LANGUAGES C CXX)

# add external/string-utils, if the parent project has not added it
if(NOT TARGET string_utils)
  file(GLOB move_data_structure_string_utils_contents "${CMAKE_CURRENT_SOURCE_DIR}/external/string-utils/*")

  if(move_data_structure_string_utils_contents STREQUAL "")
    find_package(Git REQUIRED)
    message(STATUS "Downloading the git submodule external/string-utils")
    execute_process(COMMAND "${GIT_EXECUTABLE}" submodule update --init -- external/string-utils
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
  endif()

  # the prefix of the options and environment variables (MOVE_DATA_STRUCTURE_MARCH_NATIVE,
  # MOVE_DATA_STRUCTURE_TEST_SECONDS, ...)
  if(PROJECT_IS_TOP_LEVEL)
    set(STRING_UTILS_PREFIX MOVE_DATA_STRUCTURE)
    include("${CMAKE_CURRENT_SOURCE_DIR}/external/string-utils/cmake/settings.cmake")
  endif()

  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/external/string-utils" EXCLUDE_FROM_ALL)
endif()

option(MOVE_DATA_STRUCTURE_BUILD_TESTS "Build the tests" ${PROJECT_IS_TOP_LEVEL})
set(MOVE_DATA_STRUCTURE_MAX_SHORT_VALUE "" CACHE STRING
  "Largest value stored with 32 bits in arrays of runtime width (empty: 2^32 - 1); set a small value to test the \
40-bit values")

add_library(move_data_structure INTERFACE)
target_include_directories(move_data_structure INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include/")
target_link_libraries(move_data_structure INTERFACE string_utils)

# the largest value stored with 32 bits (see max_short_value in include/data_structures/uint_bytes_array.hpp), for
# the targets of this directory and the users of move_data_structure
if(NOT MOVE_DATA_STRUCTURE_MAX_SHORT_VALUE STREQUAL "")
  add_compile_definitions(MOVE_DATA_STRUCTURE_MAX_SHORT_VALUE=${MOVE_DATA_STRUCTURE_MAX_SHORT_VALUE})
  target_compile_definitions(move_data_structure INTERFACE
    MOVE_DATA_STRUCTURE_MAX_SHORT_VALUE=${MOVE_DATA_STRUCTURE_MAX_SHORT_VALUE})
endif()

# construction and serialization of the interval blocks (see
# include/move_data_structure/move_interval_blocks/move_interval_blocks.hpp)
add_library(move_interval_blocks STATIC src/move_interval_blocks.cpp)
target_include_directories(move_interval_blocks PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include/")
target_link_libraries(move_interval_blocks PRIVATE string_utils)
target_link_libraries(move_data_structure INTERFACE move_interval_blocks)

if(STRING_UTILS_EXTERN_TEMPLATES)
  # library with the sorts of the construction listed in include/move_data_structure/construction/construction.hpp
  add_library(move_data_structure_instantiations STATIC src/instantiations.cpp)
  target_include_directories(move_data_structure_instantiations PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include/")
  target_link_libraries(move_data_structure_instantiations PRIVATE string_utils)
  target_link_libraries(move_data_structure INTERFACE move_data_structure_instantiations)
endif()

if(MOVE_DATA_STRUCTURE_BUILD_TESTS)
  include(GoogleTest)
  enable_testing()
  string_utils_add_components(tests)

  string_utils_add_executable(test-move-data-structure tests/test-move-data-structure.cpp "${CMAKE_BINARY_DIR}/tests/"
    move_data_structure string_utils_test)
  set_target_properties(test-move-data-structure PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)

  gtest_discover_tests(test-move-data-structure)
endif()
