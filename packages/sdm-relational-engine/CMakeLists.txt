cmake_minimum_required(VERSION 3.18)
project(relationallib)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "RELEASE")

find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6
  )
  FetchContent_MakeAvailable(pybind11)
endif()

set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS deployment version" FORCE)

pybind11_add_module(relationallib src/sdm_relational_engine/csrc/neighbor_sampler.cpp)

# When building in editable mode, place the compiled extension next to the
# Python sources so `pip install -e .` works as expected.
set_target_properties(relationallib PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/sdm_relational_engine"
)

# Install the extension into the sdm_relational_engine package when building a wheel.
install(TARGETS relationallib DESTINATION sdm_relational_engine)
