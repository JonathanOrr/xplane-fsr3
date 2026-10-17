// Recognises X-Plane's FSR 1 EASU and scene SSR compute shaders at vkCreateShaderModule, from what
// their SPIR-V binds and the uniform layouts the layer's own shaders read. Nothing is tied to one
// X-Plane build: a shader is recognised if and only if its interface is the one this layer was
// written for, so a future X-Plane that changes it leaves FSR 1 running instead of feeding FSR 3.1
// wrong data. tools/classify_shaders.py is the reference; tests/run_classify.py compares the two over
// every installed shader.
#pragma once
#include <cstddef>
#include <cstdint>

enum class ShaderRole : uint8_t { None, Easu, SsrScene2D, SsrSceneArray };

const char* roleName(ShaderRole role);

// words/count: the SPIR-V module, in 32-bit words. Malformed input gives None. When the module has the
// names of an EASU or scene SSR shader but fails a check, *reason (if given) says which one; it is
// left null for modules that are not candidates at all (the vast majority).
ShaderRole classifyShader(const uint32_t* words, size_t count, const char** reason = nullptr);
