// fsr3_selftest: checks that the Linux build of FSR 3.1 performs a correct temporal upscale.
//
// The scene is analytic and rendered on the CPU at render resolution: a zone plate and a rotated
// hard-edged checkerboard, with detail above the render resolution's Nyquist limit but below the
// display's. Each frame uses FSR's own jitter sequence, and the camera pans by a sub-pixel amount.
// Colour, depth and exact motion vectors go to the upscaler for N frames. The last output is then
// compared with a supersampled ground truth at display resolution.
//
// Pass criteria:
//   * FSR beats a bilinear upscale of an unjittered render-resolution frame by >= 2 dB PSNR.
//   * With the motion vectors' sign flipped (control), FSR does worse than with correct vectors.
//   * Zero validation errors when VK_LAYER_KHRONOS_validation is enabled (FSR_VALIDATION=1).
//
// Env: FSR_DEVICE=<device name substring> (default "llvmpipe"), FSR_OUT=<dir> for images,
//      FSR_FRAMES=<n> (default 48), FSR_VALIDATION=1, FSR_FEATURES=min|fp16|wave64|full.
#include <FidelityFX/host/ffx_fsr3upscaler.h>
#include <FidelityFX/host/backends/vk/ffx_vk.h>
#include "ffx_host_vk.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace
{
int g_validationErrors = 0;

#define CHECK(expr)                                                                         \
    do {                                                                                    \
        VkResult r_ = (expr);                                                               \
        if (r_ != VK_SUCCESS) { fprintf(stderr, "%s failed: %d (line %d)\n", #expr, r_, __LINE__); exit(2); } \
    } while (0)
#define FFX_CHECK(expr)                                                                     \
    do {                                                                                    \
        FfxErrorCode e_ = (expr);                                                           \
        if (e_ != FFX_OK) { fprintf(stderr, "%s failed: 0x%x (line %d)\n", #expr, e_, __LINE__); exit(2); } \
    } while (0)

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT,
                                             const VkDebugUtilsMessengerCallbackDataEXT* data, void*)
{
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
    {
        ++g_validationErrors;
        if (g_validationErrors <= 20)
            fprintf(stderr, "VALIDATION: %s\n", data->pMessage);
    }
    return VK_FALSE;
}

const char* env(const char* name, const char* fallback)
{
    const char* v = getenv(name);
    return v && *v ? v : fallback;
}

// ---------------------------------------------------------------------------------------------
// Scene. Coordinates are world units = display pixels; the camera offset is added before sampling.
constexpr int kDisplayW = 1280, kDisplayH = 720;
constexpr int kRenderW = 640, kRenderH = 360;

void scene(double x, double y, float rgb[3])
{
    // Zone plate: local frequency k*r/pi cycles per display pixel, 0.45 at r = 380.
    const double k  = 0.45 * M_PI / 380.0;
    const double dx = x - 640.0, dy = y - 360.0;
    const double zp = 0.5 + 0.5 * std::cos(k * (dx * dx + dy * dy));
    // Rotated checkerboard with 7-pixel cells: hard edges at every angle of the rotation.
    const double u = x * std::cos(0.3) + y * std::sin(0.3), v = -x * std::sin(0.3) + y * std::cos(0.3);
    const bool   checker = ((static_cast<long>(std::floor(u / 7.0)) + static_cast<long>(std::floor(v / 7.0))) & 1) != 0;
    const double c = checker ? 0.9 : 0.1;
    // Left of x = 640: zone plate; right: checkerboard. Smooth 8-pixel blend at the boundary.
    const double t = std::clamp((x - 636.0) / 8.0, 0.0, 1.0);
    rgb[0] = static_cast<float>(zp * (1 - t) + c * t);
    rgb[1] = static_cast<float>(zp * 0.8 * (1 - t) + (checker ? 0.7 : 0.2) * t);
    rgb[2] = static_cast<float>((1 - zp) * (1 - t) + (checker ? 0.2 : 0.8) * t);
}

struct Image3
{
    int                w = 0, h = 0;
    std::vector<float> px;  // RGB
    float*             at(int x, int y) { return &px[(static_cast<size_t>(y) * w + x) * 3]; }
};

// Point-sampled render: pixel (i,j)'s centre shows the scene point at (centre - jitter), in render
// pixels, per the upscaler's convention (fLrUvJittered = fHrUv + Jitter() / RenderSize()).
Image3 renderFrame(double jitterX, double jitterY, double camX, double camY)
{
    Image3 im{kRenderW, kRenderH, std::vector<float>(static_cast<size_t>(kRenderW) * kRenderH * 3)};
    const double sx = static_cast<double>(kDisplayW) / kRenderW, sy = static_cast<double>(kDisplayH) / kRenderH;
    for (int j = 0; j < kRenderH; ++j)
        for (int i = 0; i < kRenderW; ++i)
            scene((i + 0.5 - jitterX) * sx + camX, (j + 0.5 - jitterY) * sy + camY, im.at(i, j));
    return im;
}

Image3 groundTruth(double camX, double camY, int ss)
{
    Image3 im{kDisplayW, kDisplayH, std::vector<float>(static_cast<size_t>(kDisplayW) * kDisplayH * 3)};
    for (int y = 0; y < kDisplayH; ++y)
        for (int x = 0; x < kDisplayW; ++x)
        {
            double acc[3] = {0, 0, 0};
            for (int b = 0; b < ss; ++b)
                for (int a = 0; a < ss; ++a)
                {
                    float c[3];
                    scene(x + (a + 0.5) / ss + camX, y + (b + 0.5) / ss + camY, c);
                    for (int k = 0; k < 3; ++k) acc[k] += c[k];
                }
            for (int k = 0; k < 3; ++k) im.at(x, y)[k] = static_cast<float>(acc[k] / (ss * ss));
        }
    return im;
}

Image3 bilinearUpscale(Image3& src)
{
    Image3 out{kDisplayW, kDisplayH, std::vector<float>(static_cast<size_t>(kDisplayW) * kDisplayH * 3)};
    for (int y = 0; y < kDisplayH; ++y)
        for (int x = 0; x < kDisplayW; ++x)
        {
            const double u = (x + 0.5) * src.w / kDisplayW - 0.5, v = (y + 0.5) * src.h / kDisplayH - 0.5;
            const int    x0 = static_cast<int>(std::floor(u)), y0 = static_cast<int>(std::floor(v));
            const double fx = u - x0, fy = v - y0;
            auto sample = [&](int xx, int yy) { return src.at(std::clamp(xx, 0, src.w - 1), std::clamp(yy, 0, src.h - 1)); };
            for (int k = 0; k < 3; ++k)
                out.at(x, y)[k] = static_cast<float>((1 - fx) * (1 - fy) * sample(x0, y0)[k] + fx * (1 - fy) * sample(x0 + 1, y0)[k] +
                                                     (1 - fx) * fy * sample(x0, y0 + 1)[k] + fx * fy * sample(x0 + 1, y0 + 1)[k]);
        }
    return out;
}

// PSNR over the interior (16-pixel border excluded: no history exists for pixels panned in).
double psnr(Image3& a, Image3& b)
{
    double se = 0;
    long   n  = 0;
    for (int y = 16; y < a.h - 16; ++y)
        for (int x = 16; x < a.w - 16; ++x)
            for (int k = 0; k < 3; ++k)
            {
                const double d = std::clamp(a.at(x, y)[k], 0.f, 1.f) - std::clamp(b.at(x, y)[k], 0.f, 1.f);
                se += d * d;
                ++n;
            }
    return 10.0 * std::log10(1.0 / (se / n));
}

void writePPM(const std::string& path, Image3& im)
{
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return;
    fprintf(f, "P6\n%d %d\n255\n", im.w, im.h);
    for (size_t i = 0; i < im.px.size(); ++i)
    {
        const float c = std::clamp(im.px[i], 0.f, 1.f);
        fputc(static_cast<int>(std::pow(c, 1 / 2.2) * 255 + 0.5), f);
    }
    fclose(f);
}

// ---------------------------------------------------------------------------------------------
// Vulkan plumbing.
struct Gpu
{
    VkInstance               instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    VkPhysicalDevice         physical = VK_NULL_HANDLE;
    VkDevice                 device = VK_NULL_HANDLE;
    VkQueue                  queue = VK_NULL_HANDLE;
    uint32_t                 queueFamily = 0;
    VkCommandPool            pool = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memory{};
    FfxHostVkEnabled         enabled;
    FfxHostVkFunctions       host;
    std::string              name;
};

uint32_t memoryType(Gpu& g, uint32_t bits, VkMemoryPropertyFlags want)
{
    for (uint32_t i = 0; i < g.memory.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (g.memory.memoryTypes[i].propertyFlags & want) == want)
            return i;
    fprintf(stderr, "no memory type\n");
    exit(2);
}

struct Buffer
{
    VkBuffer       buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void*          mapped = nullptr;
    VkDeviceSize   size = 0;
};

Buffer makeHostBuffer(Gpu& g, VkDeviceSize size, VkBufferUsageFlags usage)
{
    Buffer b;
    b.size = size;
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = size;
    ci.usage = usage;
    CHECK(vkCreateBuffer(g.device, &ci, nullptr, &b.buffer));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(g.device, b.buffer, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = memoryType(g, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    CHECK(vkAllocateMemory(g.device, &ai, nullptr, &b.memory));
    CHECK(vkBindBufferMemory(g.device, b.buffer, b.memory, 0));
    CHECK(vkMapMemory(g.device, b.memory, 0, size, 0, &b.mapped));
    return b;
}

struct Image
{
    VkImage           image = VK_NULL_HANDLE;
    VkDeviceMemory    memory = VK_NULL_HANDLE;
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
};

Image makeImage(Gpu& g, VkFormat format, uint32_t w, uint32_t h, VkImageUsageFlags usage)
{
    Image im;
    im.info.imageType = VK_IMAGE_TYPE_2D;
    im.info.format = format;
    im.info.extent = {w, h, 1};
    im.info.mipLevels = 1;
    im.info.arrayLayers = 1;
    im.info.samples = VK_SAMPLE_COUNT_1_BIT;
    im.info.tiling = VK_IMAGE_TILING_OPTIMAL;
    im.info.usage = usage;
    im.info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    CHECK(vkCreateImage(g.device, &im.info, nullptr, &im.image));
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(g.device, im.image, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = memoryType(g, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    CHECK(vkAllocateMemory(g.device, &ai, nullptr, &im.memory));
    CHECK(vkBindImageMemory(g.device, im.image, im.memory, 0));
    return im;
}

void barrier(VkCommandBuffer cb, VkImage image, VkImageAspectFlags aspect, VkImageLayout from, VkImageLayout to)
{
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {aspect, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
}

// The library's link options wrap eight Vulkan symbols in this executable too (see ffx_host_vk.h),
// so the real functions come from vkGetInstanceProcAddr, which is not wrapped.
FfxHostVkFunctions hostFunctions(VkInstance instance)
{
    auto get = [&](const char* name) { return vkGetInstanceProcAddr(instance, name); };
    FfxHostVkFunctions fn;
    fn.GetPhysicalDeviceProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(get("vkGetPhysicalDeviceProperties"));
    fn.GetPhysicalDeviceProperties2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(get("vkGetPhysicalDeviceProperties2"));
    fn.GetPhysicalDeviceFeatures = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures>(get("vkGetPhysicalDeviceFeatures"));
    fn.GetPhysicalDeviceFeatures2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(get("vkGetPhysicalDeviceFeatures2"));
    fn.GetPhysicalDeviceMemoryProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(get("vkGetPhysicalDeviceMemoryProperties"));
    fn.EnumerateDeviceExtensionProperties = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(get("vkEnumerateDeviceExtensionProperties"));
    fn.GetDeviceProcAddr = reinterpret_cast<PFN_vkGetDeviceProcAddr>(get("vkGetDeviceProcAddr"));
    fn.CreateBuffer = reinterpret_cast<PFN_vkCreateBuffer>(get("vkCreateBuffer"));
    return fn;
}

Gpu createGpu()
{
    Gpu g;
    const bool validation = std::string(env("FSR_VALIDATION", "0")) == "1";
    std::vector<const char*> layers, extensions;
    if (validation)
    {
        layers.push_back("VK_LAYER_KHRONOS_validation");
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "fsr3_selftest";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    ici.enabledLayerCount = static_cast<uint32_t>(layers.size());
    ici.ppEnabledLayerNames = layers.data();
    ici.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    ici.ppEnabledExtensionNames = extensions.data();
    CHECK(vkCreateInstance(&ici, nullptr, &g.instance));
    g.host = hostFunctions(g.instance);
    ffxHostVkSet(g.host, g.enabled);
    if (validation)
    {
        VkDebugUtilsMessengerCreateInfoEXT mci{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
        mci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
        mci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT;
        mci.pfnUserCallback = debugCallback;
        auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(g.instance, "vkCreateDebugUtilsMessengerEXT"));
        CHECK(create(g.instance, &mci, nullptr, &g.messenger));
    }

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(g.instance, &n, nullptr);
    std::vector<VkPhysicalDevice> devices(n);
    vkEnumeratePhysicalDevices(g.instance, &n, devices.data());
    const std::string want = env("FSR_DEVICE", "llvmpipe");
    for (auto d : devices)
    {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(d, &p);
        if (std::string(p.deviceName).find(want) != std::string::npos)
        {
            g.physical = d;
            g.name = p.deviceName;
            break;
        }
    }
    if (!g.physical)
    {
        fprintf(stderr, "no device matching FSR_DEVICE=%s\n", want.c_str());
        exit(2);
    }
    vkGetPhysicalDeviceMemoryProperties(g.physical, &g.memory);

    vkGetPhysicalDeviceQueueFamilyProperties(g.physical, &n, nullptr);
    std::vector<VkQueueFamilyProperties> families(n);
    vkGetPhysicalDeviceQueueFamilyProperties(g.physical, &n, families.data());
    for (uint32_t i = 0; i < n; ++i)
        if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT && families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
        {
            g.queueFamily = i;
            break;
        }

    // Default: no extensions, no optional features. This is the most conservative configuration
    // and the one the layer will see if X-Plane enables none of them. FSR_FEATURES=full enables
    // fp16 and subgroup size control when supported, to exercise the 16-bit / wave64 permutations.
    const std::string features = env("FSR_FEATURES", "min");
    const bool wantFp16 = features == "full" || features == "fp16";
    const bool wantWave64 = features == "full" || features == "wave64";
    VkPhysicalDeviceVulkan12Features supported12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features supported13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    supported12.pNext = &supported13;
    VkPhysicalDeviceFeatures2 supported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    supported.pNext = &supported12;
    vkGetPhysicalDeviceFeatures2(g.physical, &supported);

    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f12.pNext = &f13;
    std::vector<const char*> deviceExtensions;
    {
        if (wantFp16 && supported12.shaderFloat16 && supported.features.shaderInt16)
        {
            f12.shaderFloat16 = VK_TRUE;
            g.enabled.shaderFloat16 = true;
            g.enabled.shaderInt16 = true;
            deviceExtensions.push_back(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
        }
        if (wantWave64 && supported13.subgroupSizeControl)
        {
            f13.subgroupSizeControl = VK_TRUE;
            f13.computeFullSubgroups = supported13.computeFullSubgroups;
            deviceExtensions.push_back(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
        }
    }
    for (auto e : deviceExtensions)
        g.enabled.deviceExtensions.push_back(e);

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = g.queueFamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    VkPhysicalDeviceFeatures2 enabledFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    enabledFeatures.pNext = &f12;
    enabledFeatures.features.shaderInt16 = g.enabled.shaderInt16;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &enabledFeatures;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    dci.ppEnabledExtensionNames = deviceExtensions.data();
    CHECK(vkCreateDevice(g.physical, &dci, nullptr, &g.device));
    vkGetDeviceQueue(g.device, g.queueFamily, 0, &g.queue);

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = g.queueFamily;
    CHECK(vkCreateCommandPool(g.device, &pci, nullptr, &g.pool));
    return g;
}

uint16_t toHalf(float f)
{
    _Float16 h = static_cast<_Float16>(f);
    uint16_t bits;
    std::memcpy(&bits, &h, 2);
    return bits;
}
float fromHalf(uint16_t bits)
{
    _Float16 h;
    std::memcpy(&h, &bits, 2);
    return static_cast<float>(h);
}

// ---------------------------------------------------------------------------------------------
struct RunResult
{
    Image3 output;
};

// Runs `frames` frames of FSR 3.1 with the camera moving `vx, vy` display pixels per frame.
// mvSign = +1 gives correct motion vectors, -1 the flipped control.
RunResult runFsr(Gpu& g, int frames, double vx, double vy, double mvSign)
{
    // Backend and context.
    ffxHostVkSet(g.host, g.enabled);

    const size_t scratchSize = ffxGetScratchMemorySizeVK(g.physical, FFX_FSR3UPSCALER_CONTEXT_COUNT);
    // The backend places its own structures in this memory; give it cache-line alignment.
    std::unique_ptr<uint8_t, decltype(&free)> scratchMemory(static_cast<uint8_t*>(aligned_alloc(64, (scratchSize + 63) / 64 * 64)), &free);
    uint8_t* scratch = scratchMemory.get();
    std::memset(scratch, 0, scratchSize);  // ffxGetInterfaceVK reads refCount from it before initialising
    VkDeviceContext vkdc{g.device, g.physical, g.host.GetDeviceProcAddr};
    FfxFsr3UpscalerContextDescription cd{};
    FFX_CHECK(ffxGetInterfaceVK(&cd.backendInterface, ffxGetDeviceVK(&vkdc), scratch, scratchSize, FFX_FSR3UPSCALER_CONTEXT_COUNT));
    cd.flags = FFX_FSR3UPSCALER_ENABLE_DEBUG_CHECKING;
    cd.maxRenderSize = {kRenderW, kRenderH};
    cd.maxUpscaleSize = {kDisplayW, kDisplayH};
    cd.fpMessage = [](FfxMsgType type, const wchar_t* message) {
        fprintf(stderr, "FSR %s: %ls\n", type == FFX_MESSAGE_TYPE_ERROR ? "error" : "warning", message);
    };
    auto context = std::make_unique<FfxFsr3UpscalerContext>();
    FFX_CHECK(ffxFsr3UpscalerContextCreate(context.get(), &cd));

    // Resources: colour RGBA16F, depth D32, motion RG16F at render size; output RGBA16F at display size.
    Image color = makeImage(g, VK_FORMAT_R16G16B16A16_SFLOAT, kRenderW, kRenderH, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    Image depth = makeImage(g, VK_FORMAT_D32_SFLOAT, kRenderW, kRenderH, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    Image motion = makeImage(g, VK_FORMAT_R16G16_SFLOAT, kRenderW, kRenderH, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    Image output = makeImage(g, VK_FORMAT_R16G16B16A16_SFLOAT, kDisplayW, kDisplayH,
                             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    // Shared resources the upscaler writes and reads within a dispatch. With the component API the
    // caller provides them (formats from ffxFsr3UpscalerGetSharedResourceDescriptions).
    const VkImageUsageFlags rw = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;  // the upscaler clears them
    Image dilatedDepth = makeImage(g, VK_FORMAT_R32_SFLOAT, kRenderW, kRenderH, rw);
    Image dilatedMotion = makeImage(g, VK_FORMAT_R16G16_SFLOAT, kRenderW, kRenderH, rw);
    Image prevNearestDepth = makeImage(g, VK_FORMAT_R32_UINT, kRenderW, kRenderH, rw);
    Buffer colorUp = makeHostBuffer(g, static_cast<VkDeviceSize>(kRenderW) * kRenderH * 8, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    Buffer depthUp = makeHostBuffer(g, static_cast<VkDeviceSize>(kRenderW) * kRenderH * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    Buffer motionUp = makeHostBuffer(g, static_cast<VkDeviceSize>(kRenderW) * kRenderH * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    Buffer readback = makeHostBuffer(g, static_cast<VkDeviceSize>(kDisplayW) * kDisplayH * 8, VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    // Constant depth; motion is uniform for a pure pan.
    for (int i = 0; i < kRenderW * kRenderH; ++i)
        static_cast<float*>(depthUp.mapped)[i] = 0.5f;

    VkCommandBuffer cb;
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = g.pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    CHECK(vkAllocateCommandBuffers(g.device, &cai, &cb));
    VkFence fence;
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    CHECK(vkCreateFence(g.device, &fci, nullptr, &fence));

    const int32_t phaseCount = ffxFsr3UpscalerGetJitterPhaseCount(kRenderW, kDisplayW);
    VkImageLayout colorLayout = VK_IMAGE_LAYOUT_UNDEFINED, depthLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                  motionLayout = VK_IMAGE_LAYOUT_UNDEFINED, outputLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    for (int f = 0; f < frames; ++f)
    {
        float jx = 0, jy = 0;
        FFX_CHECK(ffxFsr3UpscalerGetJitterOffset(&jx, &jy, f, phaseCount));
        if (std::string(env("FSR_NO_JITTER", "0")) == "1")
            jx = jy = 0;  // what X-Plane gives today: no projection jitter
        const double camX = vx * f, camY = vy * f;

        Image3 frame = renderFrame(jx, jy, camX, camY);
        auto* c = static_cast<uint16_t*>(colorUp.mapped);
        for (int i = 0; i < kRenderW * kRenderH; ++i)
        {
            c[i * 4 + 0] = toHalf(frame.px[i * 3 + 0]);
            c[i * 4 + 1] = toHalf(frame.px[i * 3 + 1]);
            c[i * 4 + 2] = toHalf(frame.px[i * 3 + 2]);
            c[i * 4 + 3] = toHalf(1.0f);
        }
        // Previous UV minus current UV: a world point now at screen s was at s + v last frame.
        const float mu = static_cast<float>(mvSign * vx / kDisplayW), mv = static_cast<float>(mvSign * vy / kDisplayH);
        auto* m = static_cast<uint16_t*>(motionUp.mapped);
        for (int i = 0; i < kRenderW * kRenderH; ++i)
        {
            m[i * 2 + 0] = toHalf(f == 0 ? 0.f : mu);
            m[i * 2 + 1] = toHalf(f == 0 ? 0.f : mv);
        }

        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        CHECK(vkBeginCommandBuffer(cb, &bi));
        barrier(cb, color.image, VK_IMAGE_ASPECT_COLOR_BIT, colorLayout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        barrier(cb, depth.image, VK_IMAGE_ASPECT_DEPTH_BIT, depthLayout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        barrier(cb, motion.image, VK_IMAGE_ASPECT_COLOR_BIT, motionLayout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {kRenderW, kRenderH, 1};
        vkCmdCopyBufferToImage(cb, colorUp.buffer, color.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        vkCmdCopyBufferToImage(cb, motionUp.buffer, motion.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        vkCmdCopyBufferToImage(cb, depthUp.buffer, depth.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        barrier(cb, color.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        barrier(cb, depth.image, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        barrier(cb, motion.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        barrier(cb, output.image, VK_IMAGE_ASPECT_COLOR_BIT, outputLayout, VK_IMAGE_LAYOUT_GENERAL);
        if (f == 0)
            for (Image* im : {&dilatedDepth, &dilatedMotion, &prevNearestDepth})
                barrier(cb, im->image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        colorLayout = depthLayout = motionLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        outputLayout = VK_IMAGE_LAYOUT_GENERAL;

        FfxFsr3UpscalerDispatchDescription dd{};
        dd.commandList = ffxGetCommandListVK(cb);
        dd.color = ffxGetResourceVK(reinterpret_cast<void*>(color.image), ffxGetImageResourceDescriptionVK(color.image, color.info),
                                    L"color", FFX_RESOURCE_STATE_COMPUTE_READ);
        dd.depth = ffxGetResourceVK(reinterpret_cast<void*>(depth.image), ffxGetImageResourceDescriptionVK(depth.image, depth.info),
                                    L"depth", FFX_RESOURCE_STATE_COMPUTE_READ);
        dd.motionVectors = ffxGetResourceVK(reinterpret_cast<void*>(motion.image), ffxGetImageResourceDescriptionVK(motion.image, motion.info),
                                            L"motion", FFX_RESOURCE_STATE_COMPUTE_READ);
        dd.output = ffxGetResourceVK(reinterpret_cast<void*>(output.image),
                                     ffxGetImageResourceDescriptionVK(output.image, output.info, FFX_RESOURCE_USAGE_UAV), L"output",
                                     FFX_RESOURCE_STATE_UNORDERED_ACCESS);
        auto shared = [](Image& im, const wchar_t* name) {
            return ffxGetResourceVK(reinterpret_cast<void*>(im.image), ffxGetImageResourceDescriptionVK(im.image, im.info, FFX_RESOURCE_USAGE_UAV),
                                    name, FFX_RESOURCE_STATE_UNORDERED_ACCESS);
        };
        dd.dilatedDepth = shared(dilatedDepth, L"dilatedDepth");
        dd.dilatedMotionVectors = shared(dilatedMotion, L"dilatedMotionVectors");
        dd.reconstructedPrevNearestDepth = shared(prevNearestDepth, L"reconstructedPrevNearestDepth");
        dd.jitterOffset = {jx, jy};
        dd.motionVectorScale = {static_cast<float>(kRenderW), static_cast<float>(kRenderH)};  // vectors are in UV units
        dd.renderSize = {kRenderW, kRenderH};
        dd.upscaleSize = {kDisplayW, kDisplayH};
        dd.enableSharpening = false;
        dd.frameTimeDelta = 16.7f;
        dd.preExposure = 1.0f;
        dd.reset = (f == 0);
        dd.cameraNear = 0.1f;
        dd.cameraFar = 1000.0f;
        dd.cameraFovAngleVertical = 1.0f;
        dd.viewSpaceToMetersFactor = 1.0f;
        FFX_CHECK(ffxFsr3UpscalerContextDispatch(context.get(), &dd));

        if (f == frames - 1)
        {
            barrier(cb, output.image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            VkBufferImageCopy rc{};
            rc.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            rc.imageExtent = {kDisplayW, kDisplayH, 1};
            vkCmdCopyImageToBuffer(cb, output.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback.buffer, 1, &rc);
            outputLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        }
        CHECK(vkEndCommandBuffer(cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        CHECK(vkQueueSubmit(g.queue, 1, &si, fence));
        CHECK(vkWaitForFences(g.device, 1, &fence, VK_TRUE, UINT64_MAX));
        CHECK(vkResetFences(g.device, 1, &fence));
    }

    RunResult result;
    result.output = Image3{kDisplayW, kDisplayH, std::vector<float>(static_cast<size_t>(kDisplayW) * kDisplayH * 3)};
    auto* out = static_cast<uint16_t*>(readback.mapped);
    for (int i = 0; i < kDisplayW * kDisplayH; ++i)
        for (int k = 0; k < 3; ++k)
            result.output.px[i * 3 + k] = fromHalf(out[i * 4 + k]);

    vkDeviceWaitIdle(g.device);
    FFX_CHECK(ffxFsr3UpscalerContextDestroy(context.get()));
    vkDestroyFence(g.device, fence, nullptr);
    vkFreeCommandBuffers(g.device, g.pool, 1, &cb);
    for (Image* im : {&color, &depth, &motion, &output, &dilatedDepth, &dilatedMotion, &prevNearestDepth})
    {
        vkDestroyImage(g.device, im->image, nullptr);
        vkFreeMemory(g.device, im->memory, nullptr);
    }
    for (Buffer* b : {&colorUp, &depthUp, &motionUp, &readback})
    {
        vkDestroyBuffer(g.device, b->buffer, nullptr);
        vkFreeMemory(g.device, b->memory, nullptr);
    }
    return result;
}
}  // namespace

int main()
{
    Gpu g = createGpu();
    const int frames = atoi(env("FSR_FRAMES", "48"));
    const std::string outDir = env("FSR_OUT", "");
    printf("device: %s; features: %s; frames: %d; FSR %u.%u.%u\n", g.name.c_str(), env("FSR_FEATURES", "min"), frames,
           FFX_FSR3UPSCALER_VERSION_MAJOR, FFX_FSR3UPSCALER_VERSION_MINOR, FFX_FSR3UPSCALER_VERSION_PATCH);

    const double vx = 0.37, vy = 0.11;  // display pixels per frame
    const double camX = vx * (frames - 1), camY = vy * (frames - 1);
    Image3 truth = groundTruth(camX, camY, 8);
    Image3 low = renderFrame(0, 0, camX, camY);
    Image3 bilinear = bilinearUpscale(low);

    RunResult moving = runFsr(g, frames, vx, vy, +1.0);
    RunResult flipped = runFsr(g, frames, vx, vy, -1.0);
    Image3 truthStatic = groundTruth(0, 0, 8);
    Image3 lowStatic = renderFrame(0, 0, 0, 0);
    Image3 bilinearStatic = bilinearUpscale(lowStatic);
    RunResult still = runFsr(g, frames, 0, 0, +1.0);

    const double pBilinear = psnr(bilinear, truth), pMoving = psnr(moving.output, truth), pFlipped = psnr(flipped.output, truth);
    const double pBilinearStatic = psnr(bilinearStatic, truthStatic), pStatic = psnr(still.output, truthStatic);
    printf("panning camera:  bilinear %.2f dB, FSR 3.1 %.2f dB, FSR with flipped motion vectors %.2f dB\n", pBilinear, pMoving, pFlipped);
    printf("static camera:   bilinear %.2f dB, FSR 3.1 %.2f dB\n", pBilinearStatic, pStatic);
    printf("validation errors: %d\n", g_validationErrors);

    if (!outDir.empty())
    {
        writePPM(outDir + "/truth.ppm", truth);
        writePPM(outDir + "/bilinear.ppm", bilinear);
        writePPM(outDir + "/fsr_moving.ppm", moving.output);
        writePPM(outDir + "/fsr_flipped.ppm", flipped.output);
        writePPM(outDir + "/fsr_static.ppm", still.output);
        writePPM(outDir + "/render_low.ppm", low);
    }

    const bool ok = pMoving >= pBilinear + 2.0 && pStatic >= pBilinearStatic + 2.0 && pFlipped < pMoving && g_validationErrors == 0;
    printf("%s\n", ok ? "PASS" : "FAIL");

    vkDestroyCommandPool(g.device, g.pool, nullptr);
    vkDestroyDevice(g.device, nullptr);
    if (g.messenger)
        reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(g.instance, "vkDestroyDebugUtilsMessengerEXT"))(
            g.instance, g.messenger, nullptr);
    vkDestroyInstance(g.instance, nullptr);
    return ok ? 0 : 1;
}
