cmake_minimum_required(VERSION 2.8)
project(CCDPP)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -DNDEBUG -pthread -march=native -ftree-vectorize")

set(alg
    alg/CCDPP.h
    alg/CCDPP_MPI.h)

set(struct
    struct/SparseMatrix.h
    struct/Rating.h)

set(util
    util/Base.h
    util/FileUtil.h
    util/Parameter.h
    util/RandomUtil.h
    util/Monitor.h)

set(threadpool
    boost/threadpool.hpp)

find_package(Boost COMPONENTS thread program_options system filesystem)

IF(NOT ${Boost_FOUND})
    SET(BOOST_ROOT ~/local) # default
    SET(Boost_NO_SYSTEM_PATHS ON) # force to use own build
    find_package(Boost COMPONENTS thread program_options system filesystem REQUIRED)
ENDIF(NOT ${Boost_FOUND})

if(Boost_FOUND)
    message(STATUS "Boost found")
    include_directories( ${Boost_INCLUDE_DIR})
    link_libraries(${Boost_LIBRARIES})
endif(Boost_FOUND)

find_package(MPI REQUIRED)

if (MPI_FOUND)
    message(STATUS "MPI found")
    set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
    set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS}")
    include_directories(${MPI_INCLUDE_PATH})
    link_libraries(${MPI_LIBRARIES})
endif()

add_library(lib SHARED ${util} ${alg} ${struct} ${threadpool})
set_target_properties(lib PROPERTIES LINKER_LANGUAGE CXX)

set(CCDPP runCCDPP.cpp)
add_executable(runCCDPP ${CCDPP})

target_link_libraries(runCCDPP lib)
target_link_libraries(runCCDPP ${Boost_LIBRARIES})

set(CCDPP_MPI runCCDPP_MPI.cpp)
add_executable(runCCDPP_MPI ${CCDPP_MPI})

target_link_libraries(runCCDPP_MPI lib)
target_link_libraries(runCCDPP_MPI ${Boost_LIBRARIES})
target_link_libraries(runCCDPP_MPI ${MPI_LIBRARIES})