cmake_minimum_required(VERSION 3.5)

project(dcl LANGUAGES CXX C)

enable_language(Fortran)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
option(USE_MKL "Use MKL" False)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package (Eigen3 REQUIRED NO_MODULE)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# user defined options for MKL
option(MKL_USE_parallel "Use MKL parallel" True)
option(MKL_USE_sdl "Single Dynamic Library or static/dynamic" False)
set(MKL_USE_interface "lp64" CACHE STRING "for Intel(R)64 compatible arch: ilp64/lp64 or for ia32 arch: cdecl/stdcall")

if (USE_MKL)
    add_definitions(-DUSE_MKL=1)
    find_package(MKL REQUIRED)
endif()

if (MKL_FOUND)
    include_directories(${MKL_INCLUDE_DIR})
else()
    include_directories(${CMAKE_CURRENT_LIST_DIR}/openblas)
endif()

add_subdirectory(cubature)

add_library(dcl libamos.h libamos.cpp libeispack.cpp libeispack.h libdcl.cpp libdcl.h)
if (MKL_FOUND)
    target_link_libraries(dcl Threads::Threads Eigen3::Eigen gfortran m ${MKL_LIBRARIES} cubature)
else()
    add_subdirectory(openblas)
    target_link_libraries(dcl Threads::Threads Eigen3::Eigen gfortran m openblas lapack quadmath cubature)
endif()
target_include_directories(dcl
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:.>
)
add_executable(test test.cpp)

target_link_libraries(test dcl Threads::Threads Eigen3::Eigen quadmath)