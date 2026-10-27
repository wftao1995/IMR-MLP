cmake_minimum_required(VERSION 3.20)
project(IMR-MLP VERSION 1.2.4 LANGUAGES CXX)
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
find_program(IMRMLP_MAKE NAMES gmake make REQUIRED)
set(IMRMLP_BUILD_JOBS 4 CACHE STRING "Parallel Make compilation jobs")
execute_process(COMMAND "${Python_EXECUTABLE}" -c
    "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
    OUTPUT_VARIABLE IMRMLP_EXTENSION OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
set(IMRMLP_MODULE "${CMAKE_SOURCE_DIR}/imrmlp/_imrmlp${IMRMLP_EXTENSION}")
add_custom_target(imrmlp_python ALL
    COMMAND "${IMRMLP_MAKE}" -j${IMRMLP_BUILD_JOBS} BUILD_MODE=release
        "PYTHON=${Python_EXECUTABLE}" python
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    BYPRODUCTS "${IMRMLP_MODULE}" "${CMAKE_SOURCE_DIR}/build/release/libimrmlp.so"
    COMMENT "Building IMR-MLP and its Python extension")
install(FILES "${IMRMLP_MODULE}" "${CMAKE_SOURCE_DIR}/build/release/libimrmlp.so"
    DESTINATION imrmlp)
install(DIRECTORY imrmlp/ DESTINATION imrmlp FILES_MATCHING PATTERN "*.py")
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/config/license-files.txt" IMRMLP_LICENSE_FILES)
foreach(license IN LISTS IMRMLP_LICENSE_FILES)
    get_filename_component(license_dir "${license}" DIRECTORY)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/${license}"
        DESTINATION "imrmlp/licenses/${license_dir}")
endforeach()
