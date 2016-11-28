cmake_minimum_required(VERSION 3.3.1)

project(CompositeSIMPLE LANGUAGES CXX C)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
include(FindMKL)
include(FindPETSc)

find_package(MKL REQUIRED)
find_package(PETSc REQUIRED)

# Add executable called "main" that is built from the source files
# "demo.cxx" and "demo_b.cxx". The extensions are automatically found.
add_executable (main main.c fieldsplit.c)

target_link_libraries(
    main
    PUBLIC
        PETSC::PETSC
)