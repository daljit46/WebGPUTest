cmake_minimum_required(VERSION 3.20)

project(WebGPUTest LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")


include(CompilerCache)
use_compiler_cache()

set(DAWN_FETCH_DEPENDENCIES ON)

if (APPLE)
    set(USE_VULKAN OFF)
    set(USE_METAL ON)
else()
    set(USE_VULKAN ON)
    set(USE_METAL OFF)
endif()
set(DAWN_USE_GLFW ON)
set(DAWN_ENABLE_D3D11 OFF)
set(DAWN_ENABLE_D3D12 OFF)
set(DAWN_ENABLE_METAL ${USE_METAL})
set(DAWN_ENABLE_NULL OFF)
set(DAWN_ENABLE_DESKTOP_GL OFF)
set(DAWN_ENABLE_OPENGLES OFF)
set(DAWN_ENABLE_VULKAN ${USE_VULKAN})
set(TINT_BUILD_SPV_READER OFF)
set(DAWN_BUILD_SAMPLES OFF)
set(TINT_BUILD_CMD_TOOLS OFF)
set(TINT_BUILD_SAMPLES OFF)
set(TINT_BUILD_DOCS OFF)
set(TINT_BUILD_TESTS OFF)
set(TINT_BUILD_FUZZERS OFF)
set(TINT_BUILD_SPIRV_TOOLS_FUZZER OFF)
set(TINT_BUILD_AST_FUZZER OFF)
set(TINT_BUILD_REGEX_FUZZER OFF)
set(TINT_BUILD_BENCHMARKS OFF)
set(TINT_BUILD_TESTS OFF)
set(TINT_BUILD_AS_OTHER_OS OFF)
set(TINT_BUILD_REMOTE_COMPILE OFF)

add_subdirectory(dawn EXCLUDE_FROM_ALL)
add_subdirectory(glfw3webgpu)

set(IMGUI_SOURCES
    imgui/imgui.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_widgets.cpp
    imgui/imgui_tables.cpp
)

add_executable(WebGPUTest 
    main.cpp
    utils.cpp
    ${IMGUI_SOURCES}
)

target_link_libraries(WebGPUTest PRIVATE
    webgpu_dawn
    webgpu_cpp
    webgpu_glfw
    glfw3webgpu
)

target_include_directories(WebGPUTest PRIVATE
    imgui
)

install(TARGETS WebGPUTest
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
