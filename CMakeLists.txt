cmake_minimum_required(VERSION 3.20)

project(WebGPUTest LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")


include(CompilerCache)

find_package(Threads REQUIRED)

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
    imgui/imconfig.h
    imgui/imgui.h
    imgui/imgui.cpp
    imgui/misc/cpp/imgui_stdlib.h
    imgui/misc/cpp/imgui_stdlib.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_internal.h
    imgui/imgui_widgets.cpp
    imgui/imgui_tables.cpp
    imgui/backends/imgui_impl_glfw.h
    imgui/backends/imgui_impl_glfw.cpp
    imgui/backends/imgui_impl_wgpu.h
    imgui/backends/imgui_impl_wgpu.cpp
)

add_executable(WebGPUTest 
    main.cpp
    utils.cpp
    ${IMGUI_SOURCES}
    application.h application.cpp
)

target_link_libraries(WebGPUTest PRIVATE
    webgpu_dawn
    webgpu_cpp
    webgpu_glfw
    glfw3webgpu
    Threads::Threads
)

target_include_directories(WebGPUTest PRIVATE
    imgui
)

add_custom_target(Shaders SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader.wgsl
)

add_custom_command(TARGET ${CMAKE_PROJECT_NAME} PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
    "$<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders")

add_custom_command(TARGET ${CMAKE_PROJECT_NAME} PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    "${CMAKE_CURRENT_SOURCE_DIR}/assets"
    "$<TARGET_FILE_DIR:${PROJECT_NAME}>/assets")

install(TARGETS WebGPUTest
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
