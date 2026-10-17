// Prints the layer's runtime classification of each SPIR-V file: "<file> <role> <reason|->", then the
// total classification time. Driven by tests/run_classify.py.
#include "shader_classify.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <vector>

int main(int argc, char** argv)
{
    double seconds = 0;
    size_t bytes   = 0;
    for (int i = 1; i < argc; ++i)
    {
        std::ifstream         f(argv[i], std::ios::binary);
        std::vector<char>     data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        std::vector<uint32_t> words(data.size() / 4);
        std::copy(data.begin(), data.begin() + words.size() * 4, reinterpret_cast<char*>(words.data()));
        const char* reason = nullptr;
        const auto  t      = std::chrono::steady_clock::now();
        ShaderRole  role   = classifyShader(words.data(), words.size(), &reason);
        seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
        bytes += data.size();
        std::printf("%s %s %s\n", argv[i], roleName(role), reason ? reason : "-");
    }
    std::printf("total %d files %zu bytes %.3f ms\n", argc - 1, bytes, seconds * 1000.0);
    return 0;
}
