cmake_minimum_required(VERSION 4.4)
project(wave_renderer)

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_DIR)

find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)
find_package(FLAC)
find_package(nanobind CONFIG REQUIRED)

# add_executable(decoder decoder.cpp)
nanobind_add_module(decoder decoder.cpp)

target_sources(decoder
  PRIVATE
    decoder.cpp
)
  
target_compile_features(decoder
  PRIVATE
  cxx_std_23
)

set_target_properties(decoder PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
  
target_link_libraries(decoder
  PRIVATE
    Arrow::arrow_shared
    Parquet::parquet_shared
    FLAC
    FLAC++
)

