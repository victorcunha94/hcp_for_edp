cmake_minimum_required(VERSION 3.30)
project(hcp_for_edp LANGUAGES C)

set(CMAKE_C_STANDARD 11)

find_package(OpenMP REQUIRED)

add_executable(hcp_for_edp
    assignment02/C-gbispo/main2.c
        assignment02/C-gbispo/metodos_numericos.c
        assignment02/C-gbispo/metodos_numericos.h
        assignment02/C-gbispo/plotagem.c
        assignment02/C-gbispo/plotagem.h)

target_link_libraries(hcp_for_edp PRIVATE m OpenMP::OpenMP_C)
