cmake_minimum_required(VERSION 3.16)
project(MTP_KIM_Driver VERSION 1.0.0 LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find the modern KIM API item-building package.  The pre-2.2 compatibility
# package does not support SOURCES and is broken with current KIM API releases.
find_package(KIM-API 2.2 REQUIRED CONFIG)
find_package(KIM-API-ITEMS 2.2 REQUIRED CONFIG)

# KIM API 2.4's item macros query configuration-specific locations directly.
# Some distro packages install only RELEASE locations, even for single-config
# generators, so provide the equivalent locations for common build types.
foreach(_kim_tool base64-encode collections-info)
    get_target_property(_kim_tool_release KIM-API::${_kim_tool}
                        IMPORTED_LOCATION_RELEASE)
    if(_kim_tool_release)
        foreach(_kim_config NOCONFIG DEBUG RELWITHDEBINFO MINSIZEREL)
            set_property(TARGET KIM-API::${_kim_tool} PROPERTY
                         IMPORTED_LOCATION_${_kim_config} "${_kim_tool_release}")
        endforeach()
    endif()
endforeach()

# ── Build the MTP core as a static library ──────────────────────────
add_library(mtp_core STATIC
    mtp_kim_core.cpp
    mtp_kim_core.h
    mtp_kim_radial.h
)
target_include_directories(mtp_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# ── Build as KIM model driver ────────────────────────────────────────
set(MTP_KIM_DRIVER_NAME "MTP__MD_000000000000_000" CACHE STRING
    "OpenKIM model-driver identifier")
add_kim_api_model_driver_library(
    NAME "${MTP_KIM_DRIVER_NAME}"
    CREATE_ROUTINE_NAME MTPDriver_Create
    CREATE_ROUTINE_LANGUAGE cpp
)
target_sources(${MTP_KIM_DRIVER_NAME} PRIVATE
    kim_model_driver.cpp
    mtp_kim_core.h
    mtp_kim_radial.h
)
target_link_libraries(${MTP_KIM_DRIVER_NAME} PRIVATE mtp_core)

if(BUILD_TESTING)
    add_executable(mtp_core_tests test_mtp_core.cpp)
    target_link_libraries(mtp_core_tests PRIVATE mtp_core)
    add_test(NAME mtp_core_tests COMMAND mtp_core_tests)
endif()
