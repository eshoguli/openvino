# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file (GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

file (GLOB HDR ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

include_directories($<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>)

set(TARGET_NAME "blob_dumper")

ie_add_sample(NAME blob_dumper
              SOURCES ${SRC}
              HEADERS ${HDR}
              DEPENDENCIES format_reader inference_engine inference_engine_legacy
              OPENCV_DEPENDENCIES imgcodecs)

if ( CMAKE_COMPILER_IS_GNUCC )
    target_compile_options(blob_dumper PRIVATE "-Wno-strict-aliasing")
endif()

include_directories(SYSTEM ${IE_MAIN_SOURCE_DIR}/src/legacy_api/include/legacy)
target_include_directories(${TARGET_NAME} PUBLIC ${PUBLIC_HEADERS_DIR} $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>)
