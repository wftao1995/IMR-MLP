cmake_minimum_required(VERSION 3.22)
project(mtp_standalone_lammps LANGUAGES CXX)

# Build upstream's own lammps library and main.cpp, without changing its source
# tree. All ordinary LAMMPS cache options (BUILD_MPI, PKG_*, etc.) remain usable.
set(LAMMPS_SOURCE_DIR "" CACHE PATH "Upstream LAMMPS root or src directory")
if(EXISTS "${LAMMPS_SOURCE_DIR}/lammps.h")
  get_filename_component(_lammps_root "${LAMMPS_SOURCE_DIR}/.." ABSOLUTE)
else()
  get_filename_component(_lammps_root "${LAMMPS_SOURCE_DIR}" ABSOLUTE)
endif()
if(NOT EXISTS "${_lammps_root}/cmake/CMakeLists.txt" OR
   NOT EXISTS "${_lammps_root}/src/lammps.h")
  message(FATAL_ERROR "LAMMPS_SOURCE_DIR must identify an upstream LAMMPS root or src directory")
endif()

set(MTP_BACKEND CPU CACHE STRING "Execution backend: CPU, CUDA or HIP")
set_property(CACHE MTP_BACKEND PROPERTY STRINGS CPU CUDA HIP)
string(TOUPPER "${MTP_BACKEND}" MTP_BACKEND)
if(NOT MTP_BACKEND MATCHES "^(CPU|CUDA|HIP)$")
  message(FATAL_ERROR "MTP_BACKEND must be CPU, CUDA or HIP")
endif()
option(MTP_OPENMP "Embed mlip/opt/omp and enable the LAMMPS OPENMP package" OFF)
if(MTP_OPENMP)
  if(DEFINED PKG_OPENMP AND NOT PKG_OPENMP)
    message(FATAL_ERROR "MTP_OPENMP=ON requires PKG_OPENMP=ON; remove the conflicting OFF setting")
  endif()
  set(PKG_OPENMP ON CACHE BOOL "Build LAMMPS OPENMP package")
  if(DEFINED BUILD_OMP AND NOT BUILD_OMP)
    message(FATAL_ERROR "MTP_OPENMP=ON requires BUILD_OMP=ON; remove the conflicting OFF setting")
  endif()
  set(BUILD_OMP ON CACHE BOOL "Build LAMMPS with OpenMP support")
endif()

# Enable the device language at the common parent of the runtime and LAMMPS
# targets. In particular a CPU build never probes a CUDA/HIP toolkit.
if(MTP_BACKEND STREQUAL "CUDA")
  enable_language(CUDA)
elseif(MTP_BACKEND STREQUAL "HIP")
  enable_language(HIP)
endif()

add_subdirectory("${_lammps_root}/cmake" upstream)
if(NOT TARGET lammps OR NOT TARGET lmp OR
   NOT COMMAND AddStyleHeader OR NOT COMMAND GenerateStyleHeader)
  message(FATAL_ERROR "This LAMMPS CMake version lacks the required lammps/lmp targets or style-header helpers")
endif()

# Reuse exactly the standalone core build, without its plugin, registration
# runner, or separately compiled adapters. Pair sources below inherit LAMMPS's
# MPI/stubs, integer-size, compiler and OPENMP settings directly from lammps.
set(MTP_BUILD_ADAPTERS OFF)
enable_testing()
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/.." runtime)

get_filename_component(_mlip "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(_pairs "${_mlip}/opt/pair_MLIP_opt")
set(_styles "mlip/opt")
if(MTP_OPENMP)
  list(APPEND _pairs "${_mlip}/opt/pair_MLIP_omp")
  list(APPEND _styles "mlip/opt/omp")
  target_include_directories(lammps PRIVATE "${_lammps_root}/src/OPENMP")
endif()
if(NOT MTP_BACKEND STREQUAL "CPU")
  list(APPEND _pairs "${_mlip}/pair_MLIP_gpu_direct")
  list(APPEND _styles "mlip/gpudirect")
  target_link_libraries(lammps PRIVATE MTP::runtime mtp_gpu)
else()
  target_link_libraries(lammps PRIVATE MTP::runtime)
endif()

# Reject old installed copies even if their filenames differ. Silently
# overwriting a factory entry can select an implementation from another build.
get_property(_existing_pairs GLOBAL PROPERTY PAIR)
foreach(_header IN LISTS _existing_pairs)
  file(READ "${_header}" _contents)
  foreach(_style IN LISTS _styles)
    if(_contents MATCHES "PairStyle[ \t\r\n]*\\([ \t\r\n]*${_style}[ \t\r\n]*,")
      message(FATAL_ERROR "Duplicate pair style ${_style} in ${_header}; use a clean LAMMPS source tree without an installed MLIP copy")
    endif()
  endforeach()
endforeach()

foreach(_pair IN LISTS _pairs)
  target_sources(lammps PRIVATE "${_pair}.cpp")
  AddStyleHeader("${_pair}.h" PAIR)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|IntelLLVM")
    set_property(SOURCE "${_pair}.cpp" TARGET_DIRECTORY lammps APPEND
      PROPERTY COMPILE_OPTIONS -fopenmp-simd)
  endif()
  if(MTP_ARCH_FLAGS)
    separate_arguments(_arch NATIVE_COMMAND "${MTP_ARCH_FLAGS}")
    set_property(SOURCE "${_pair}.cpp" TARGET_DIRECTORY lammps APPEND
      PROPERTY COMPILE_OPTIONS ${_arch})
  endif()
endforeach()
target_compile_features(lammps PUBLIC cxx_std_17)
target_include_directories(lammps PRIVATE "${_mlip}" "${_mlip}/opt")

# Regenerate only the pair header using upstream's generator. Reconfiguration
# starts with upstream's normal style list, so disabling a backend removes its
# registration as well. The supplied source tree is never modified.
get_target_property(_lammps_binary lammps BINARY_DIR)
GenerateStyleHeader("${_lammps_binary}/styles" PAIR pair)
set_target_properties(lmp PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
message(STATUS "Standalone pair styles embedded in standard lmp: ${_styles}")

if(MTP_BUILD_TESTS)
  find_package(Python3 QUIET COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    # Always exercise the embedded CPU style, including in GPU builds. Device
    # checks are explicit commands so ordinary CTest needs no allocated GPU.
    add_test(NAME mtp_lmp_cpu COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../tests/run_regression.py"
      --lmp "$<TARGET_FILE:lmp>" --ranks 1
      --work "${CMAKE_CURRENT_BINARY_DIR}/regression-cpu")
    set_tests_properties(mtp_lmp_cpu PROPERTIES TIMEOUT 300 LABELS lammps)
    if(MTP_OPENMP)
      add_test(NAME mtp_lmp_omp COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../tests/run_regression.py"
        --lmp "$<TARGET_FILE:lmp>" --ranks 1 --style mlip/opt/omp --threads 2
        --work "${CMAKE_CURRENT_BINARY_DIR}/regression-omp")
      set_tests_properties(mtp_lmp_omp PROPERTIES TIMEOUT 300
        PROCESSORS 2 LABELS lammps)
    endif()
  endif()
endif()
