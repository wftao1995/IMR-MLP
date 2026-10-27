cmake_minimum_required(VERSION 3.18)
project(mlip_maxvol_plugin LANGUAGES CXX)

get_filename_component(IMRMLP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(LAMMPS_HEADER_DIR "" CACHE PATH "Source headers of the target LAMMPS executable")
set(IMRMLP_LIBRARY "${IMRMLP_ROOT}/build/mpi/libimrmlp-mpi.so"
    CACHE FILEPATH "Matching MPI IMR-MLP library")
set(IMRMLP_FLAGS_FILE "${IMRMLP_ROOT}/build/mpi/.flags"
    CACHE FILEPATH "Compile flags recorded by the matching IMR-MLP build")

if(NOT EXISTS "${LAMMPS_HEADER_DIR}/lammpsplugin.h")
  message(FATAL_ERROR "Set LAMMPS_HEADER_DIR to the target LAMMPS src directory")
endif()
if(NOT EXISTS "${IMRMLP_LIBRARY}" OR NOT EXISTS "${IMRMLP_FLAGS_FILE}")
  message(FATAL_ERROR "Build libimrmlp-mpi first: make BUILD_MODE=mpi libimrmlp.so")
endif()
file(READ "${IMRMLP_FLAGS_FILE}" IMRMLP_FLAGS_TEXT)
separate_arguments(IMRMLP_ABI_FLAGS UNIX_COMMAND "${IMRMLP_FLAGS_TEXT}")
if(NOT "-DIMRMLP_MPI" IN_LIST IMRMLP_ABI_FLAGS)
  message(FATAL_ERROR "An MPI IMRMLP build is required; serial IMRMLP exports incompatible MPI stubs")
endif()
list(REMOVE_ITEM IMRMLP_ABI_FLAGS -MMD -MP)
# Make paths in the recorded Makefile command independent of CMake's build dir.
set(ABSOLUTE_IMRMLP_FLAGS)
set(IS_INCLUDE_FILE FALSE)
foreach(FLAG IN LISTS IMRMLP_ABI_FLAGS)
  if(IS_INCLUDE_FILE)
    if(NOT IS_ABSOLUTE "${FLAG}")
      set(FLAG "${IMRMLP_ROOT}/${FLAG}")
    endif()
    set(IS_INCLUDE_FILE FALSE)
  elseif(FLAG STREQUAL "-include")
    set(IS_INCLUDE_FILE TRUE)
  elseif(FLAG MATCHES "^-I(.+)$")
    if(NOT IS_ABSOLUTE "${CMAKE_MATCH_1}")
      set(FLAG "-I${IMRMLP_ROOT}/${CMAKE_MATCH_1}")
    endif()
  endif()
  list(APPEND ABSOLUTE_IMRMLP_FLAGS "${FLAG}")
endforeach()

find_package(MPI REQUIRED COMPONENTS CXX)
add_library(mlip_maxvol_plugin MODULE maxvol_plugin.cpp fix_mlip_maxvol.cpp)
target_compile_features(mlip_maxvol_plugin PRIVATE cxx_std_23)
target_compile_options(mlip_maxvol_plugin PRIVATE ${ABSOLUTE_IMRMLP_FLAGS})
target_include_directories(mlip_maxvol_plugin PRIVATE "${LAMMPS_HEADER_DIR}" "${IMRMLP_ROOT}")
target_link_libraries(mlip_maxvol_plugin PRIVATE MPI::MPI_CXX "${IMRMLP_LIBRARY}")
get_filename_component(IMRMLP_LIBRARY_DIR "${IMRMLP_LIBRARY}" DIRECTORY)
set_target_properties(mlip_maxvol_plugin PROPERTIES PREFIX "" BUILD_RPATH "${IMRMLP_LIBRARY_DIR}")
if(APPLE)
  target_link_options(mlip_maxvol_plugin PRIVATE -Wl,-undefined,dynamic_lookup)
endif()

include(CTest)
set(LAMMPS_EXECUTABLE "" CACHE FILEPATH "MPI LAMMPS executable with PLUGIN for regression tests")
set(IMRMLP_EXECUTABLE "${IMRMLP_ROOT}/build/release/imrmlp"
    CACHE FILEPATH "imrmlp executable for offline reference grades")
set(IMRMLP_TEST_MTP "${IMRMLP_ROOT}/unitest/test-data/potential.mtp"
    CACHE FILEPATH "Two-species MTP fixture for ordinary MaxVol regression")
if(BUILD_TESTING AND LAMMPS_EXECUTABLE)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_executable(generate_family_models tests/generate_family_models.cpp)
  target_compile_features(generate_family_models PRIVATE cxx_std_23)
  target_compile_options(generate_family_models PRIVATE ${ABSOLUTE_IMRMLP_FLAGS})
  target_include_directories(generate_family_models PRIVATE "${IMRMLP_ROOT}")
  target_link_libraries(generate_family_models PRIVATE MPI::MPI_CXX "${IMRMLP_LIBRARY}")
  add_test(NAME maxvol_lammps_mpi
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_regression.py"
      --lmp "${LAMMPS_EXECUTABLE}" --plugin "$<TARGET_FILE:mlip_maxvol_plugin>"
      --imrmlp-libdir "${IMRMLP_LIBRARY_DIR}"
      --imrmlp "${IMRMLP_EXECUTABLE}" --mpiexec "${MPIEXEC_EXECUTABLE}"
      --model "${IMRMLP_TEST_MTP}")
  set_tests_properties(maxvol_lammps_mpi PROPERTIES TIMEOUT 300)
  add_test(NAME maxvol_lammps_mtp_family
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_family_regression.py"
      --lmp "${LAMMPS_EXECUTABLE}" --plugin "$<TARGET_FILE:mlip_maxvol_plugin>"
      --generator "$<TARGET_FILE:generate_family_models>"
      --imrmlp-libdir "${IMRMLP_LIBRARY_DIR}" --imrmlp "${IMRMLP_EXECUTABLE}"
      --mpiexec "${MPIEXEC_EXECUTABLE}")
  set_tests_properties(maxvol_lammps_mtp_family PROPERTIES TIMEOUT 300)
endif()
