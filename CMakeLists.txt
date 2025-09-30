cmake_minimum_required(VERSION 3.30)
project(hcp_for_edp LANGUAGES C)

set(CMAKE_C_STANDARD 11)

find_package(MPI REQUIRED)

add_executable(hcp_for_edp
        assignment03/gbispo/paralleljacobiellipti.c)

target_link_libraries(hcp_for_edp PRIVATE m MPI::MPI_C)