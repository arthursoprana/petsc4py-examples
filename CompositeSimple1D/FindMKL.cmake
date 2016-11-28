# Find the Intel Math Kernel Library
#
# Creates these targets:
#    - MKL::Runtime
#    - MKL::Dev
#
# Also defines some variables, although they should be avoided.
# Use the imported targets instead.
#
# - MKL_FOUND - system has MKL
# - MKL_LIBRARIES - MKL libraries
# - MKL_INCLUDE_DIR - the MKL include directory
#    -> Only if the component DEV is specified at the find_package() call
#
# By default only the MKL RT (runtime) is searched for, if the headers are desired (mkl.h and others)
# specify the COMPONENT "DEV":
#
#    eg.: find_package(MKL COMPONENT DEV)

set(MKL_LIBRARIES "")

# Find libraries
# --------------

# mkl_rt
find_library(MKL_RT_LIBRARY mkl_rt)
list(APPEND MKL_LIBRARIES ${MKL_RT_LIBRARY})

# intel OpenMP
find_library(MKL_IOMP5_LIBRARY
    NAMES
        iomp5md
        iomp5
        libiomp5md
        libiomp5
)
list(APPEND MKL_LIBRARIES ${MKL_IOMP5_LIBRARY})
if (WIN32)
    set(OpenMP_CXX_FLAGS "/openmp")
else()
    set(OpenMP_CXX_FLAGS "-fopenmp")
endif()

# pthread is needed by default
if (UNIX)
    list(APPEND MKL_LIBRARIES pthread)
endif()

include(FindPackageHandleStandardArgs)

# Find headers if asked for
if ("DEV" IN_LIST MKL_FIND_COMPONENTS)
    find_path(MKL_INCLUDE_DIR
        mkl.h
        PATHS
            ${INCLUDE_INSTALL_DIR}
    )
    find_package_handle_standard_args(MKL DEFAULT_MSG MKL_INCLUDE_DIR)
    mark_as_advanced(MKL_INCLUDE_DIR)
endif()

find_package_handle_standard_args(MKL DEFAULT_MSG MKL_LIBRARIES MKL_IOMP5_LIBRARY MKL_RT_LIBRARY)
mark_as_advanced(MKL_LIBRARIES MKL_IOMP5_LIBRARY MKL_RT_LIBRARY)

if (MKL_FOUND)
    if (NOT TARGET MKL::Runtime)
        if (WIN32)
            add_library(MKL::Runtime STATIC IMPORTED)
        else()
            add_library(MKL::Runtime SHARED IMPORTED)
        endif()
        set_target_properties(MKL::Runtime
            PROPERTIES
                IMPORTED_LOCATION "${MKL_RT_LIBRARY}"
                INTERFACE_LINK_LIBRARIES "${MKL_IOMP5_LIBRARY}"
                INTERFACE_COMPILE_OPTIONS "${OpenMP_CXX_FLAGS}"
                # The mkl_rt.so does not have the DT_SONAME entry in the .dynamic section, so cmake links this library with the full path
                # instead of only its name.
                # Due to this, libraries which linked with MKL would have on its dynamic section as "$CONDA_ENVS_PATH/lib/mkl_rt.so"
                #   obs.:
                #       - replace $CONDA_ENVS_PATH with the environment in which the library was compiled
                #       - To check the dynamic section: (readelf -d | grep mkl_rt.so)
                #
                # It is wrong to have the full path there, since the dependency resolution should point to the mkl_rt.so installed at the 'current' conda environment.
                IMPORTED_NO_SONAME TRUE
        )
        if (WIN32)
            set_target_properties(MKL::Runtime
                PROPERTIES
                    INTERFACE_LINK_FLAGS_DEBUG "/nodefaultlib:vcompd"
                    INTERFACE_LINK_FLAGS_DEBUGLITE "/nodefaultlib:vcomp"
                    INTERFACE_LINK_FLAGS_RELWITHDEBINFO "/nodefaultlib:vcomp"
                    INTERFACE_LINK_FLAGS_RELEASE "/nodefaultlib:vcomp"
            )
        endif()
    endif()
    if (MKL_INCLUDE_DIR AND NOT TARGET MKL::Dev)
        add_library(MKL::Dev INTERFACE IMPORTED GLOBAL)
        set_target_properties(MKL::Dev PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES "MKL::Runtime"
        )
    endif()
endif()