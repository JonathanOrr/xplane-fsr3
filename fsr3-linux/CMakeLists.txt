# Linux build of AMD FidelityFX SDK v1.1.4's FSR 3.1 upscaler (3.1.5 after patches/fsr-3.1.5-backport.patch) with its Vulkan backend.
# AMD ships this SDK for Windows only. This file builds just the pieces the upscaler needs:
#   1. FidelityFX_SC, the SDK's shader permutation compiler, ported to Linux (GLSL path only).
#   2. The upscaler's Vulkan shader permutation headers, generated with it.
#   3. libffx_fsr3upscaler_vk.a: the upscaler component, the Vulkan backend and shared code.
# See PORTING.md for every change made to AMD's sources.
cmake_minimum_required(VERSION 3.20)
project(fsr3_linux CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(FFX_SDK "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/FidelityFX-SDK/sdk" CACHE PATH "FidelityFX SDK v1.1.4 sdk directory")
set(FFX_SC_LIBS "${FFX_SDK}/tools/ffx_shader_compiler/libs")
find_package(Threads REQUIRED)
find_package(Vulkan REQUIRED)
find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)

# 1. Shader compiler (host tool).
add_executable(FidelityFX_SC
    ffx_sc/ffx_sc.cpp ffx_sc/glsl_compiler.cpp ffx_sc/utils.cpp
    "${FFX_SC_LIBS}/MD5/md5.cpp"
    "${FFX_SC_LIBS}/SPIRV-Reflect/spirv_reflect.c"
    "${FFX_SC_LIBS}/tiny-process-library/process.cpp"
    "${FFX_SC_LIBS}/tiny-process-library/process_unix.cpp")
target_include_directories(FidelityFX_SC PRIVATE ffx_sc
    "${FFX_SC_LIBS}/MD5" "${FFX_SC_LIBS}/SPIRV-Reflect" "${FFX_SC_LIBS}/tiny-process-library")
target_link_libraries(FidelityFX_SC PRIVATE Threads::Threads)

# 2. Shader permutation headers.
set(FFX_SHADER_OUT "${CMAKE_CURRENT_BINARY_DIR}/shaders")
file(GLOB FSR3_GLSL "${FFX_SDK}/src/backends/vk/shaders/fsr3upscaler/*.glsl")
file(GLOB FSR3_GPU_HEADERS "${FFX_SDK}/include/FidelityFX/gpu/fsr3upscaler/*.h" "${FFX_SDK}/include/FidelityFX/gpu/fsr1/*.h"
     "${FFX_SDK}/include/FidelityFX/gpu/spd/*.h" "${FFX_SDK}/include/FidelityFX/gpu/*.h")
add_custom_command(
    OUTPUT "${FFX_SHADER_OUT}/stamp"
    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/gen_shaders.sh" $<TARGET_FILE:FidelityFX_SC> "${FFX_SDK}" "${FFX_SHADER_OUT}"
    COMMAND ${CMAKE_COMMAND} -E touch "${FFX_SHADER_OUT}/stamp"
    DEPENDS FidelityFX_SC gen_shaders.sh ${FSR3_GLSL} ${FSR3_GPU_HEADERS}
    COMMENT "Generating FSR 3.1 upscaler Vulkan shader permutations")
add_custom_target(fsr3_shaders DEPENDS "${FFX_SHADER_OUT}/stamp")

# 3. Upscaler + Vulkan backend.
add_library(ffx_fsr3upscaler_vk STATIC
    "${FFX_SDK}/src/components/fsr3upscaler/ffx_fsr3upscaler.cpp"
    "${FFX_SDK}/src/backends/vk/ffx_vk.cpp"
    "${FFX_SDK}/src/backends/shared/ffx_shader_blobs.cpp"
    "${FFX_SDK}/src/backends/shared/blob_accessors/ffx_fsr3upscaler_shaderblobs.cpp"
    "${FFX_SDK}/src/shared/ffx_assert.cpp"
    "${FFX_SDK}/src/shared/ffx_message.cpp"
    "${FFX_SDK}/src/shared/ffx_breadcrumbs_list.cpp"
    "${FFX_SDK}/src/shared/ffx_object_management.cpp")
add_dependencies(ffx_fsr3upscaler_vk fsr3_shaders)
target_compile_options(ffx_fsr3upscaler_vk PRIVATE
    -include "${CMAKE_CURRENT_SOURCE_DIR}/ffx_linux_compat.h" -w)
# wchar_t is 4 bytes on Linux, so the private contexts outgrow the Windows-sized public blobs
# (measured: FSR3 upscaler private context 837,848 bytes vs 524,288). Callers must see the same value.
target_compile_definitions(ffx_fsr3upscaler_vk PUBLIC FFX_GCC "FFX_SDK_DEFAULT_CONTEXT_SIZE=(1024 * 256)"
                                               PRIVATE FFX_FSR3UPSCALER)
target_include_directories(ffx_fsr3upscaler_vk
    PUBLIC  "${FFX_SDK}/include"
    PRIVATE "${FFX_SDK}/include/FidelityFX/host" "${FFX_SDK}/src/shared" "${FFX_SDK}/src/backends/shared"
            "${FFX_SDK}/src/components" "${FFX_SHADER_OUT}"
            "${FFX_SDK}/include/FidelityFX/gpu" "${FFX_SDK}/include/FidelityFX/gpu/fsr3upscaler")
target_link_libraries(ffx_fsr3upscaler_vk PUBLIC Vulkan::Headers)

# Host shim: the backend's eight direct Vulkan calls go to functions the host supplies
# (ffx_host_vk.h). The linker rewrites those references in every consumer.
target_sources(ffx_fsr3upscaler_vk PRIVATE ffx_host_vk.cpp ffx_fg_stub.cpp)
target_include_directories(ffx_fsr3upscaler_vk PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
set(FFX_HOST_VK_WRAP
    vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties2 vkGetPhysicalDeviceFeatures
    vkGetPhysicalDeviceFeatures2 vkGetPhysicalDeviceMemoryProperties vkEnumerateDeviceExtensionProperties
    vkGetDeviceProcAddr vkCreateBuffer)
foreach(fn ${FFX_HOST_VK_WRAP})
    target_link_options(ffx_fsr3upscaler_vk INTERFACE "LINKER:--wrap=${fn}")
endforeach()

# Self-test: FSR 3.1 against a supersampled ground truth (tests/fsr3_selftest.cpp), on Lavapipe and,
# with -DFSR3_TEST_GPU=<part of a Vulkan device name>, on that GPU too. The Khronos validation layer is
# used when it is installed, or unpacked into VULKAN_VALIDATION_LAYER_DIR.
enable_testing()
add_executable(fsr3_selftest tests/fsr3_selftest.cpp)
target_link_libraries(fsr3_selftest PRIVATE ffx_fsr3upscaler_vk Vulkan::Vulkan)
set(FSR3_TEST_GPU "" CACHE STRING "Part of a Vulkan device name (e.g. 9070) for the GPU tests; empty: Lavapipe only")
set(VULKAN_VALIDATION_LAYER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/validation-layers/layer" CACHE PATH
    "Optional unpacked Khronos validation layer")
set(FSR3_VALIDATION_ENV "")
if(EXISTS "${VULKAN_VALIDATION_LAYER_DIR}/VkLayer_khronos_validation.json")
    set(FSR3_VALIDATION_ENV "FSR_VALIDATION=1;VK_ADD_LAYER_PATH=${VULKAN_VALIDATION_LAYER_DIR}")
elseif(EXISTS "/usr/share/vulkan/explicit_layer.d/VkLayer_khronos_validation.json")
    set(FSR3_VALIDATION_ENV "FSR_VALIDATION=1")
endif()
add_test(NAME fsr3_selftest_lavapipe COMMAND fsr3_selftest)
set_tests_properties(fsr3_selftest_lavapipe PROPERTIES ENVIRONMENT "FSR_DEVICE=llvmpipe;${FSR3_VALIDATION_ENV}" SKIP_RETURN_CODE 2)
if(FSR3_TEST_GPU)
    add_test(NAME fsr3_selftest_gpu COMMAND fsr3_selftest)
    set_tests_properties(fsr3_selftest_gpu PROPERTIES
        ENVIRONMENT "FSR_DEVICE=${FSR3_TEST_GPU};FSR_FEATURES=full;${FSR3_VALIDATION_ENV}" SKIP_RETURN_CODE 2)
endif()
