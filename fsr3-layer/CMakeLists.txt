# VK_LAYER_XPLANE_fsr3: AMD FSR 3.1 in place of X-Plane 12's FSR 1 EASU pass. See README.md.
cmake_minimum_required(VERSION 3.20)
project(xp_fsr3_layer VERSION 0.1.2 LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_program(GLSLANG glslangValidator REQUIRED)

# FSR 3.1 upscaler + Vulkan backend for Linux (../fsr3-linux).
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../fsr3-linux ${CMAKE_CURRENT_BINARY_DIR}/fsr3-linux)

# X-Plane is needed only by the tests (its shaders are read, never modified). The layer recognises
# X-Plane's shaders at runtime (src/shader_classify.cpp), so it builds without an install.
set(XPLANE_ROOT "" CACHE PATH "X-Plane 12 installation, for the tests")
set(XP_SPV "${XPLANE_ROOT}/Resources/shaders/bin/spv")

# The layer's own compute passes.
set(SPV_DIR ${CMAKE_CURRENT_BINARY_DIR}/spv)
file(MAKE_DIRECTORY ${SPV_DIR})
function(compile_shader out src)
    add_custom_command(OUTPUT ${SPV_DIR}/${out}.spv
        COMMAND ${GLSLANG} -V --target-env vulkan1.1 ${ARGN} ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${src} -o ${SPV_DIR}/${out}.spv
        DEPENDS shaders/${src} VERBATIM)
endfunction()
compile_shader(inputs_2d inputs.comp -DINPUT_ARRAY=0)
compile_shader(inputs_array inputs.comp -DINPUT_ARRAY=1)
compile_shader(copy_in copy_in.comp)
compile_shader(copy_out copy_out.comp)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/layer_shaders.h
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed_spirv.py ${CMAKE_CURRENT_BINARY_DIR}/layer_shaders.h
            kInputs2D=${SPV_DIR}/inputs_2d.spv kInputsArray=${SPV_DIR}/inputs_array.spv
            kCopyIn=${SPV_DIR}/copy_in.spv kCopyOut=${SPV_DIR}/copy_out.spv
    DEPENDS tools/embed_spirv.py ${SPV_DIR}/inputs_2d.spv ${SPV_DIR}/inputs_array.spv ${SPV_DIR}/copy_in.spv ${SPV_DIR}/copy_out.spv
    VERBATIM)

add_library(VkLayer_xplane_fsr3 SHARED src/layer.cpp src/engine.cpp src/shader_classify.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/layer_shaders.h)
target_include_directories(VkLayer_xplane_fsr3 PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
# A layer must not link libvulkan: every call goes down the chain (the FidelityFX backend's direct
# calls are --wrap'ed by ffx_fsr3upscaler_vk's link options; see ../fsr3-linux/ffx_host_vk.h).
target_link_libraries(VkLayer_xplane_fsr3 PRIVATE ffx_fsr3upscaler_vk Vulkan::Headers pthread)
target_compile_options(VkLayer_xplane_fsr3 PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
# Portability floor, checked by tests/check_portable.py: glibc 2.34 (X-Plane 12's own) and libstdc++
# GLIBCXX_3.4.29 (GCC 11, shipped by every distribution with glibc 2.34). Under Steam the container
# uses the newer of the host's and the runtime's libstdc++. The C++ runtime is not linked statically:
# a current toolchain's libstdc++.a/libgcc_eh.a need glibc 2.35-2.38 (arc4random, _dl_find_object,
# __isoc23_strtoul).
target_link_options(VkLayer_xplane_fsr3 PRIVATE -Wl,-Bsymbolic -Wl,--exclude-libs,ALL -Wl,--no-undefined)
set_target_properties(VkLayer_xplane_fsr3 PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_compile_definitions(VkLayer_xplane_fsr3 PRIVATE XP_FSR3_VERSION="${PROJECT_VERSION}")
configure_file(layer.json.in ${CMAKE_CURRENT_BINARY_DIR}/VkLayer_xplane_fsr3.json @ONLY)

# ---- Tests: the layer driven by a fake X-Plane built from X-Plane's own installed shaders.
enable_testing()
add_executable(classify_check tests/classify_check.cpp src/shader_classify.cpp)
target_include_directories(classify_check PRIVATE src)
if(NOT EXISTS "${XP_SPV}/fsr.xsa" OR NOT EXISTS "${XP_SPV}/ssr_deferred.xsa")
    message(STATUS "XPLANE_ROOT not set to an X-Plane 12 install: tests not built")
    return()
endif()
add_test(NAME shader_classify
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_classify.py ${CMAKE_CURRENT_BINARY_DIR} "${XPLANE_ROOT}")
set_tests_properties(shader_classify PROPERTIES SKIP_RETURN_CODE 77)
add_test(NAME portable COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_portable.py $<TARGET_FILE:VkLayer_xplane_fsr3>)
add_executable(fake_xplane tests/fake_xplane.cpp)
target_link_libraries(fake_xplane PRIVATE Vulkan::Vulkan pthread)
target_compile_options(fake_xplane PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
# Extract the two shaders the harness needs from the installed archives (read only).
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/xp/ssr_deferred_1.spv ${CMAKE_CURRENT_BINARY_DIR}/xp/ssr_deferred_2.spv ${CMAKE_CURRENT_BINARY_DIR}/xp/fsr_4.spv
    COMMAND ${Python3_EXECUTABLE} -c "import zipfile,sys;[open(sys.argv[3]+'/'+n,'wb').write(zipfile.ZipFile(a).read(n)) for a,n in ((sys.argv[1],'ssr_deferred_1.spv'),(sys.argv[1],'ssr_deferred_2.spv'),(sys.argv[2],'fsr_4.spv'))]"
            "${XP_SPV}/ssr_deferred.xsa" "${XP_SPV}/fsr.xsa" ${CMAKE_CURRENT_BINARY_DIR}/xp
    DEPENDS "${XP_SPV}/ssr_deferred.xsa" "${XP_SPV}/fsr.xsa" VERBATIM)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/xp)
add_custom_target(xp_shaders ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/xp/ssr_deferred_1.spv ${CMAKE_CURRENT_BINARY_DIR}/xp/ssr_deferred_2.spv
    ${CMAKE_CURRENT_BINARY_DIR}/xp/fsr_4.spv)
add_test(NAME fake_xplane_lavapipe
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_fake_xplane.py ${CMAKE_CURRENT_BINARY_DIR} llvmpipe)
if(FSR3_TEST_GPU)  # defined in ../fsr3-linux
    add_test(NAME fake_xplane_gpu
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_fake_xplane.py ${CMAKE_CURRENT_BINARY_DIR} "${FSR3_TEST_GPU}")
    set_tests_properties(fake_xplane_gpu PROPERTIES SKIP_RETURN_CODE 77)
endif()
# Raster G-buffer shaders for fake_xplane --raster.
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/xp/gbuffer.vert.spv ${CMAKE_CURRENT_BINARY_DIR}/xp/gbuffer.frag.spv
    COMMAND ${GLSLANG} -V ${CMAKE_CURRENT_SOURCE_DIR}/tests/gbuffer.vert -o ${CMAKE_CURRENT_BINARY_DIR}/xp/gbuffer.vert.spv
    COMMAND ${GLSLANG} -V ${CMAKE_CURRENT_SOURCE_DIR}/tests/gbuffer.frag -o ${CMAKE_CURRENT_BINARY_DIR}/xp/gbuffer.frag.spv
    DEPENDS tests/gbuffer.vert tests/gbuffer.frag VERBATIM)
add_custom_target(test_shaders ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/xp/gbuffer.vert.spv ${CMAKE_CURRENT_BINARY_DIR}/xp/gbuffer.frag.spv)
