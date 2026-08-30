cmake_minimum_required(VERSION 3.16)

project(rotquant_native VERSION 0.1.0 LANGUAGES C CXX)

include(CMakePackageConfigHelpers)
include(CheckCXXCompilerFlag)
include(GNUInstallDirs)

option(ROTQUANT_NATIVE_WARNINGS_AS_ERRORS "Treat native warnings as errors" ON)
option(ROTQUANT_NATIVE_BUILD_TOOLS "Build native conformance and benchmark tools" ON)
option(ROTQUANT_NATIVE_ENABLE_AVX2 "Compile the x86 AVX2 kernel when supported" ON)

function(rotquant_native_enable_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4)
        if(ROTQUANT_NATIVE_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE /WX)
        endif()
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
        if(ROTQUANT_NATIVE_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE -Werror)
        endif()
    endif()
endfunction()

add_library(rotquant_native
    src/native_v2.cpp
    src/native_v2_c.cpp
)
add_library(RotQuant::native ALIAS rotquant_native)
set_target_properties(rotquant_native PROPERTIES EXPORT_NAME native)
target_include_directories(rotquant_native PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(rotquant_native PUBLIC cxx_std_17)

set(ROTQUANT_NATIVE_X86 OFF)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i.86)$")
    set(ROTQUANT_NATIVE_X86 ON)
elseif(APPLE AND CMAKE_OSX_ARCHITECTURES MATCHES "(^|;)x86_64($|;)")
    set(ROTQUANT_NATIVE_X86 ON)
endif()

if(ROTQUANT_NATIVE_ENABLE_AVX2 AND ROTQUANT_NATIVE_X86)
    if(MSVC)
        set(ROTQUANT_NATIVE_AVX2_FLAG "/arch:AVX2")
        set(ROTQUANT_NATIVE_HAS_AVX2_FLAG ON)
    else()
        check_cxx_compiler_flag("-mavx2" ROTQUANT_NATIVE_HAS_AVX2_FLAG)
        set(ROTQUANT_NATIVE_AVX2_FLAG "-mavx2")
    endif()
    if(ROTQUANT_NATIVE_HAS_AVX2_FLAG)
        target_sources(rotquant_native PRIVATE src/native_v2_avx2.cpp)
        set_source_files_properties(
            src/native_v2_avx2.cpp
            PROPERTIES COMPILE_OPTIONS "${ROTQUANT_NATIVE_AVX2_FLAG}"
        )
        target_compile_definitions(
            rotquant_native PRIVATE ROTQUANT_NATIVE_COMPILED_AVX2=1
        )
    endif()
endif()

if(BUILD_SHARED_LIBS)
    target_compile_definitions(rotquant_native
        PRIVATE ROTQUANT_NATIVE_BUILDING_LIBRARY
        PUBLIC ROTQUANT_NATIVE_SHARED
    )
endif()
rotquant_native_enable_warnings(rotquant_native)

if(ROTQUANT_NATIVE_BUILD_TOOLS)
    add_executable(rotquant-native-cli src/cli.cpp)
    target_link_libraries(rotquant-native-cli PRIVATE RotQuant::native)
    rotquant_native_enable_warnings(rotquant-native-cli)

    add_executable(rotquant-native-bench src/benchmark.cpp)
    target_link_libraries(rotquant-native-bench PRIVATE RotQuant::native)
    rotquant_native_enable_warnings(rotquant-native-bench)

    add_executable(rotquant-native-conformance tests/conformance.cpp)
    target_link_libraries(rotquant-native-conformance PRIVATE RotQuant::native)
    rotquant_native_enable_warnings(rotquant-native-conformance)

    add_executable(rotquant-native-c-conformance tests/c_conformance.c)
    target_link_libraries(rotquant-native-c-conformance PRIVATE RotQuant::native)
    rotquant_native_enable_warnings(rotquant-native-c-conformance)

    enable_testing()
    add_test(NAME rotquant-native-conformance COMMAND rotquant-native-conformance)
    add_test(
        NAME rotquant-native-c-conformance
        COMMAND rotquant-native-c-conformance
    )
endif()

install(
    TARGETS rotquant_native
    EXPORT RotQuantNativeTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(
    EXPORT RotQuantNativeTargets
    FILE RotQuantNativeTargets.cmake
    NAMESPACE RotQuant::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/RotQuantNative
)

configure_package_config_file(
    cmake/RotQuantNativeConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/RotQuantNativeConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/RotQuantNative
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/RotQuantNativeConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/RotQuantNativeConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/RotQuantNativeConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/RotQuantNative
)
