# Creates the imported target PETSC::PETSC
message(STATUS "Searching for PETSC")

include(CMakeFindDependencyMacro)
find_file(PETSC_CMAKE_INCLUDE_FILE PETScBuildInternal.cmake
    PATH_SUFFIXES
        lib/petsc/conf
        cmake
)
if (EXISTS "${PETSC_CMAKE_INCLUDE_FILE}")
    include(${PETSC_CMAKE_INCLUDE_FILE})
    find_path(PETSC_INCLUDE_DIR petsc.h)
    list(APPEND PETSC_INCLUDE_DIR "${PETSC_PACKAGE_INCLUDES}")
    find_library(PETSC_LIBRARY NAMES petsc)
else()
    message(STATUS "PETScBuildInternal.cmake not found: ${PETSC_CMAKE_INCLUDE_FILE}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PETSC
    REQUIRED_VARS
        PETSC_LIBRARY
        PETSC_INCLUDE_DIR
        PETSC_CMAKE_INCLUDE_FILE
        PETSC_PACKAGE_LIBS
)

if(PETSC_FOUND AND NOT TARGET PETSC::PETSC)
    if (WIN32)
        add_library(PETSC::PETSC STATIC IMPORTED)
    else()
        add_library(PETSC::PETSC SHARED IMPORTED)
    endif()

    set_target_properties(PETSC::PETSC PROPERTIES
        IMPORTED_LOCATION "${PETSC_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${PETSC_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${PETSC_PACKAGE_LIBS}"
    )
endif()