// fake_xplane: drives VK_LAYER_XPLANE_fsr3 the way X-Plane 12 does, without the simulator.
//
// Its compute pipelines are built from X-Plane's own installed SPIR-V: ssr_deferred_1.spv (scene
// SSR, 2D-array variant) and fsr_4.spv (FSR 1 EASU), with descriptor layouts matching their
// interfaces (spirv-cross of the installed shaders). Each frame records, in one ONE_TIME_SUBMIT
// command buffer, uploads of a CPU-rendered scene (colour; gbuffer_pos = reversed-Z depth;
// gbuffer_vel flags), the SSR dispatch, then the EASU dispatch, and presents to a headless
// swapchain. No jitter, as in X-Plane.
//
// The scene: a fine checkerboard wall 10 m away while the camera translates sideways (world
// reprojection, u_reproj), and a "cockpit panel" 1 m away across the bottom quarter that moves with
// the camera (gbuffer_vel bit 4, identity u_local_reproj). The final EASU output - X-Plane's EASU,
// or FSR 3.1 when the layer replaces it - is compared with a supersampled ground truth.
//
// Args: <ssr_deferred_1.spv> <fsr_4.spv> [--flip-reproj] [--frames N] [--out DIR] [--raster <vert.spv> <frag.spv>]
// --msaa-ssr <ssr_deferred_2.spv>: X-Plane with MSAA on. The scene SSR is X-Plane's multisampled
// variant, recorded (bind and dispatch) into a command buffer that is never submitted, since this
// harness has no multisampled G-buffer; the regular SSR is not dispatched.
//       [--split | --split-reversed] [--zoom R] [--load N]
// --load N (with --raster): X-Plane-like API volume to measure the layer's CPU cost. Each frame
// resets a descriptor pool, allocates and updates N sets, and records N x {bind pipeline, bind
// descriptor set (dynamic offset), set viewport, draw} in the G-buffer pass. The JSON reports the
// mean CPU microseconds per frame of both parts (load_alloc_us, load_record_us).
// --load-pools P spreads the per-frame sets over P pools (each reset every frame); --load-persistent K
// allocates K long-lived sets once (as an engine's material sets), which every tracked reset sees.
// --load-threads T moves that load onto T persistent worker threads, as X-Plane's 24 workers with
// per-thread descriptor pools (as measured in X-Plane 12.4 with XP_FSR3_PROFILE=1). Each frame every worker
// resets its pool, allocates its share of the N sets (half with a storage-image layout, the kind the
// layer tracks), updates them through descriptor update templates, creates and destroys uniform
// buffers, and records 7 binds per set (graphics for the plain half, compute for the tracked half:
// the live ratio was ~8500 binds to ~1230 updates) into its own command buffer, which is never
// submitted. Meanwhile the main thread records SSR and EASU, whose layer hooks read the same
// tables. load_threads_us: mean wall time from waking the workers to the last one finishing.
// --zoom R multiplies the focal length by (1 + R) every frame (X-Plane's scroll-wheel zoom): the
// projection changes between frames while u_reproj stays a pure camera move, as X-Plane's does
// (live telemetry: rigid u_reproj at every FOV).
// --split records SSR and EASU in two command buffers submitted in that order, as X-Plane does
// (first live run: EASU never shared a recording with SSR). --split-reversed submits them in the
// wrong order: a negative control for the layer's submission-order check.
// --raster draws the scene as geometry in a G-buffer render pass instead of uploading CPU images, so
// the layer's viewport jitter applies. As in X-Plane (first live run), gbuffer_pos and gbuffer_vel
// are then the depth and stencil views of the pass's reversed-Z depth/stencil buffer (stencil 4 =
// own aircraft), not colour attachments.
// Env:  FAKE_DEVICE=<name substring> (default llvmpipe), FAKE_VALIDATION=1, FAKE_SIZE (see kRW).
// Prints one JSON line: {"psnr_wall":..,"psnr_panel":..,"validation_errors":..}.
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

#define CHECK(expr)                                                                                    \
    do {                                                                                               \
        VkResult r_ = (expr);                                                                          \
        if (r_ != VK_SUCCESS) { fprintf(stderr, "%s failed: %d (line %d)\n", #expr, r_, __LINE__); exit(2); } \
    } while (0)

int g_validationErrors = 0;
VKAPI_ATTR VkBool32 VKAPI_CALL onMessage(VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT,
                                         const VkDebugUtilsMessengerCallbackDataEXT* data, void*)
{
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
    {
        if (++g_validationErrors <= 15) fprintf(stderr, "VALIDATION: %s\n", data->pMessage);
    }
    return VK_FALSE;
}

// ---------------------------------------------------------------------------------------------
// Scene and camera. View space: -Z forward (as X-Plane's ssr_deferred reconstructs it), metres.
// FAKE_SIZE=<render w>x<render h>:<display w>x<display h> overrides the sizes, to time the GPU passes
// at a real sim's size (e.g. 4233x1666:5504x2166).
struct Sizes { uint32_t rw = 640, rh = 360, dw = 1280, dh = 720; };
Sizes sizesFromEnv()
{
    Sizes s;
    if (const char* e = std::getenv("FAKE_SIZE"))
        if (sscanf(e, "%ux%u:%ux%u", &s.rw, &s.rh, &s.dw, &s.dh) != 4) { fprintf(stderr, "bad FAKE_SIZE %s\n", e); exit(2); }
    return s;
}
const Sizes        kSizes = sizesFromEnv();
const uint32_t     kRW = kSizes.rw, kRH = kSizes.rh;   // render (EASU input)
const uint32_t     kDW = kSizes.dw, kDH = kSizes.dh;   // display (EASU output)
constexpr float    kNear = 0.05f, kFovY = 1.0471976f, kWall = 10.0f, kPanel = 1.0f;
// Panel covers screen v >= this. FAKE_PANEL_ROW overrides it without --raster (0: cockpit fills the
// screen, as when the sim zooms onto the gauges).
const float        kPanelRow = std::getenv("FAKE_PANEL_ROW") ? float(atof(std::getenv("FAKE_PANEL_ROW"))) : 0.75f;
const float        kF = 1.0f / std::tan(kFovY * 0.5f), kAspect = float(kDW) / float(kDH);
float              g_f = kF;  // this frame's focal length (changes with --zoom)

struct Vec3 { float x, y, z; };

// uv in [0,1]^2 -> view-space ray direction (z = -1); the same mapping inputs.comp inverts.
Vec3 ray(float u, float v) { return {(2 * u - 1) * kAspect / g_f, (2 * v - 1) / g_f, -1.0f}; }

void wallColor(float wx, float wy, float c[3])
{
    const float a = 0.2f, cs = std::cos(a), sn = std::sin(a);
    const float u = wx * cs + wy * sn, v = -wx * sn + wy * cs;
    const bool  k = ((long(std::floor(u / 0.035f)) + long(std::floor(v / 0.035f))) & 1) != 0;
    c[0] = k ? 0.9f : 0.15f;
    c[1] = k ? 0.8f : 0.2f;
    c[2] = k ? 0.3f : 0.6f;
}

void panelColor(float px, float py, float c[3])
{
    // Instrument-like rings and fine stripes, fixed to the camera.
    const float r = std::sqrt((px * 2.0f - std::round(px * 2.0f)) * (px * 2.0f - std::round(px * 2.0f)) + (py * 4.0f - 1.6f) * (py * 4.0f - 1.6f));
    const float ring = 0.5f + 0.5f * std::cos(r * 90.0f);
    const float stripe = (long(std::floor(px / 0.006f)) & 1) ? 0.8f : 0.3f;
    c[0] = 0.2f + 0.6f * ring;
    c[1] = 0.25f + 0.5f * stripe * ring;
    c[2] = 0.3f;
}

// Shades one sample; returns depth (reversed infinite: near / distance) and gbuffer_vel flags.
float shade(float u, float v, Vec3 cam, float c[3], uint32_t& flags)
{
    Vec3 d = ray(u, v);
    if (v >= kPanelRow)
    {
        panelColor(d.x * kPanel, d.y * kPanel, c);
        flags = 4;
        return kNear / kPanel;
    }
    wallColor(d.x * kWall + cam.x, d.y * kWall + cam.y, c);
    flags = 0;
    return kNear / kWall;
}

// ---------------------------------------------------------------------------------------------
struct Gpu {
    VkInstance       instance{};
    VkDebugUtilsMessengerEXT messenger{};
    VkPhysicalDevice physical{};
    VkDevice         device{};
    VkQueue          queue{};
    uint32_t         family = 0;
    VkPhysicalDeviceMemoryProperties memory{};
    VkPhysicalDeviceProperties props{};
};

uint32_t memType(Gpu& g, uint32_t bits, VkMemoryPropertyFlags want)
{
    for (uint32_t i = 0; i < g.memory.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (g.memory.memoryTypes[i].propertyFlags & want) == want) return i;
    fprintf(stderr, "no memory type\n");
    exit(2);
}

struct Img {
    VkImage image{};
    VkDeviceMemory memory{};
    VkImageView view{};
    VkFormat format{};
    uint32_t w = 0, h = 0;
};

Img makeImage(Gpu& g, VkFormat format, uint32_t w, uint32_t h, VkImageUsageFlags usage, bool array,
              VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT)
{
    Img im;
    im.format = format; im.w = w; im.h = h;
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D; ci.format = format; ci.extent = {w, h, 1}; ci.mipLevels = 1; ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT; ci.tiling = VK_IMAGE_TILING_OPTIMAL; ci.usage = usage;
    CHECK(vkCreateImage(g.device, &ci, nullptr, &im.image));
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(g.device, im.image, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = memType(g, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    CHECK(vkAllocateMemory(g.device, &ai, nullptr, &im.memory));
    CHECK(vkBindImageMemory(g.device, im.image, im.memory, 0));
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = im.image;
    vi.viewType = array ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    vi.format = format;
    vi.subresourceRange = {aspect, 0, 1, 0, 1};
    CHECK(vkCreateImageView(g.device, &vi, nullptr, &im.view));
    return im;
}

struct Buf { VkBuffer buffer{}; VkDeviceMemory memory{}; uint8_t* mapped = nullptr; VkDeviceSize size = 0; };
Buf makeBuffer(Gpu& g, VkDeviceSize size, VkBufferUsageFlags usage)
{
    Buf b;
    b.size = size;
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = size; ci.usage = usage;
    CHECK(vkCreateBuffer(g.device, &ci, nullptr, &b.buffer));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(g.device, b.buffer, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = memType(g, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    CHECK(vkAllocateMemory(g.device, &ai, nullptr, &b.memory));
    CHECK(vkBindBufferMemory(g.device, b.buffer, b.memory, 0));
    void* p;
    CHECK(vkMapMemory(g.device, b.memory, 0, VK_WHOLE_SIZE, 0, &p));
    b.mapped = static_cast<uint8_t*>(p);
    return b;
}

void barrier(VkCommandBuffer cb, VkImage image, VkImageLayout from, VkImageLayout to)
{
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    b.oldLayout = from; b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
}

std::vector<uint32_t> readSpv(const char* path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "cannot read %s\n", path); exit(2); }
    std::vector<uint32_t> words(static_cast<size_t>(f.tellg()) / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(words.data()), words.size() * 4);
    return words;
}

uint16_t half(float f) { _Float16 h = static_cast<_Float16>(f); uint16_t b; std::memcpy(&b, &h, 2); return b; }
float fromHalf(uint16_t b) { _Float16 h; std::memcpy(&h, &b, 2); return static_cast<float>(h); }
uint32_t fbits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

void writePPM(const std::string& path, const std::vector<float>& rgb, uint32_t w, uint32_t h)
{
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return;
    fprintf(f, "P6\n%u %u\n255\n", w, h);
    for (float c : rgb) fputc(int(std::pow(std::clamp(c, 0.f, 1.f), 1 / 2.2f) * 255 + 0.5f), f);
    fclose(f);
}

double psnr(const std::vector<float>& a, const std::vector<float>& b, uint32_t w, uint32_t y0, uint32_t y1, uint32_t border)
{
    double se = 0;
    long n = 0;
    for (uint32_t y = y0; y < y1; ++y)
        for (uint32_t x = border; x < w - border; ++x)
            for (int k = 0; k < 3; ++k)
            {
                double d = std::clamp(a[(y * w + x) * 3 + k], 0.f, 1.f) - std::clamp(b[(y * w + x) * 3 + k], 0.f, 1.f);
                se += d * d;
                ++n;
            }
    return 10 * std::log10(1.0 / (se / n));
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 3) { fprintf(stderr, "usage: fake_xplane <ssr_deferred_1.spv> <fsr_4.spv> [--flip-reproj] [--frames N] [--out DIR]\n"); return 2; }
    bool flip = false;
    int frames = 48;
    std::string out;
    const char* rasterVert = nullptr;
    const char* rasterFrag = nullptr;
    bool split = false, reversed = false;
    float zoom = 0.0f;
    int load = 0, loadPools = 1, loadPersistent = 0, loadThreads = 0;
    const char* msaaSsr = nullptr;
    for (int i = 3; i < argc; ++i)
    {
        if (!std::strcmp(argv[i], "--flip-reproj")) flip = true;
        else if (!std::strcmp(argv[i], "--frames") && i + 1 < argc) frames = atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!std::strcmp(argv[i], "--raster") && i + 2 < argc) { rasterVert = argv[++i]; rasterFrag = argv[++i]; }
        else if (!std::strcmp(argv[i], "--split")) split = true;
        else if (!std::strcmp(argv[i], "--split-reversed")) split = reversed = true;
        else if (!std::strcmp(argv[i], "--zoom") && i + 1 < argc) zoom = float(atof(argv[++i]));
        else if (!std::strcmp(argv[i], "--load") && i + 1 < argc) load = atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--load-pools") && i + 1 < argc) loadPools = std::max(1, atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--load-persistent") && i + 1 < argc) loadPersistent = atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--load-threads") && i + 1 < argc) loadThreads = std::max(0, atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--msaa-ssr") && i + 1 < argc) msaaSsr = argv[++i];
    }
    const char* deviceName = getenv("FAKE_DEVICE") ? getenv("FAKE_DEVICE") : "llvmpipe";
    const bool validation = getenv("FAKE_VALIDATION") && !std::strcmp(getenv("FAKE_VALIDATION"), "1");

    // ---- Instance, device, headless swapchain.
    Gpu g;
    std::vector<const char*> layers, iext = {VK_KHR_SURFACE_EXTENSION_NAME, "VK_EXT_headless_surface"};
    if (validation) { layers.push_back("VK_LAYER_KHRONOS_validation"); iext.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME); }
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "fake_xplane";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    ici.enabledLayerCount = uint32_t(layers.size()); ici.ppEnabledLayerNames = layers.data();
    ici.enabledExtensionCount = uint32_t(iext.size()); ici.ppEnabledExtensionNames = iext.data();
    CHECK(vkCreateInstance(&ici, nullptr, &g.instance));
    if (validation)
    {
        VkDebugUtilsMessengerCreateInfoEXT mci{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
        mci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        mci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT;
        mci.pfnUserCallback = onMessage;
        CHECK(reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(g.instance, "vkCreateDebugUtilsMessengerEXT"))(
            g.instance, &mci, nullptr, &g.messenger));
    }
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(g.instance, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    vkEnumeratePhysicalDevices(g.instance, &n, pds.data());
    for (auto pd : pds)
    {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(pd, &p);
        if (strstr(p.deviceName, deviceName)) { g.physical = pd; g.props = p; break; }
    }
    if (!g.physical) { fprintf(stderr, "no device matching %s\n", deviceName); return 2; }
    vkGetPhysicalDeviceMemoryProperties(g.physical, &g.memory);
    vkGetPhysicalDeviceQueueFamilyProperties(g.physical, &n, nullptr);
    std::vector<VkQueueFamilyProperties> fams(n);
    vkGetPhysicalDeviceQueueFamilyProperties(g.physical, &n, fams.data());
    for (uint32_t i = 0; i < n; ++i)
        if ((fams[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && (fams[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) { g.family = i; break; }

    VkSurfaceKHR surface;
    VkHeadlessSurfaceCreateInfoEXT hs{VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT};
    CHECK(reinterpret_cast<PFN_vkCreateHeadlessSurfaceEXT>(vkGetInstanceProcAddr(g.instance, "vkCreateHeadlessSurfaceEXT"))(
        g.instance, &hs, nullptr, &surface));
    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = g.family; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    const char* dext[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci; dci.enabledExtensionCount = 1; dci.ppEnabledExtensionNames = dext;
    CHECK(vkCreateDevice(g.physical, &dci, nullptr, &g.device));
    vkGetDeviceQueue(g.device, g.family, 0, &g.queue);

    VkSurfaceCapabilitiesKHR caps;
    CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(g.physical, surface, &caps));
    vkGetPhysicalDeviceSurfaceFormatsKHR(g.physical, surface, &n, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(n);
    vkGetPhysicalDeviceSurfaceFormatsKHR(g.physical, surface, &n, formats.data());
    VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    sci.surface = surface;
    sci.minImageCount = std::max(caps.minImageCount, 2u);
    sci.imageFormat = formats[0].format;
    sci.imageColorSpace = formats[0].colorSpace;
    sci.imageExtent = {kDW, kDH};
    sci.imageArrayLayers = 1;
    sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    sci.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    sci.clipped = VK_TRUE;
    VkSwapchainKHR swapchain;
    CHECK(vkCreateSwapchainKHR(g.device, &sci, nullptr, &swapchain));
    vkGetSwapchainImagesKHR(g.device, swapchain, &n, nullptr);
    std::vector<VkImage> swapImages(n);
    vkGetSwapchainImagesKHR(g.device, swapchain, &n, swapImages.data());

    // ---- X-Plane's shaders and descriptor interfaces.
    auto ssrCode = readSpv(argv[1]);
    auto easuCode = readSpv(argv[2]);
    std::vector<VkShaderModule> modules;
    auto module = [&](const std::vector<uint32_t>& code) {
        VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        ci.codeSize = code.size() * 4; ci.pCode = code.data();
        VkShaderModule m;
        CHECK(vkCreateShaderModule(g.device, &ci, nullptr, &m));
        modules.push_back(m);
        return m;
    };
    auto setLayout = [&](std::vector<VkDescriptorSetLayoutBinding> b) {
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        ci.bindingCount = uint32_t(b.size()); ci.pBindings = b.data();
        VkDescriptorSetLayout l;
        CHECK(vkCreateDescriptorSetLayout(g.device, &ci, nullptr, &l));
        return l;
    };
    const auto CIS = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, ST = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    const auto UB = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, UBD = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    const auto CS = VK_SHADER_STAGE_COMPUTE_BIT;
    // ssr_deferred_1: 0 gbuffer_mat, 1 gbuffer_nrm, 2 u_environment_data, 3 gbuffer_pos, 4 gbuffer_vel,
    // 6 tex_brdf (2D), 7 tex_hiZ, 8 tex_ssr, 10 i_output, 11 u_gbuffer_data, 12 u_ssr_data.
    VkDescriptorSetLayout ssrSetLayout = setLayout({{0, CIS, 1, CS, nullptr}, {1, CIS, 1, CS, nullptr}, {2, UB, 1, CS, nullptr},
                                                    {3, CIS, 1, CS, nullptr}, {4, CIS, 1, CS, nullptr}, {6, CIS, 1, CS, nullptr},
                                                    {7, CIS, 1, CS, nullptr}, {8, CIS, 1, CS, nullptr}, {10, ST, 1, CS, nullptr},
                                                    {11, UBD, 1, CS, nullptr}, {12, UBD, 1, CS, nullptr}});
    // fsr_4 (EASU): 0 u_input_sampler, 1 u_input_texture (2D array), 2 i_output_texture, 3 u_fsr_data.
    VkDescriptorSetLayout easuSetLayout = setLayout({{0, VK_DESCRIPTOR_TYPE_SAMPLER, 1, CS, nullptr},
                                                     {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, CS, nullptr}, {2, ST, 1, CS, nullptr},
                                                     {3, UBD, 1, CS, nullptr}});
    auto pipeline = [&](VkDescriptorSetLayout sl, VkShaderModule m, VkPipelineLayout& layout) {
        VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pl.setLayoutCount = 1; pl.pSetLayouts = &sl;
        CHECK(vkCreatePipelineLayout(g.device, &pl, nullptr, &layout));
        VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cp.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, m, "main", nullptr};
        cp.layout = layout;
        VkPipeline p;
        CHECK(vkCreateComputePipelines(g.device, VK_NULL_HANDLE, 1, &cp, nullptr, &p));
        return p;
    };
    VkPipelineLayout ssrLayout, easuLayout;
    VkPipeline ssrPipeline = pipeline(ssrSetLayout, module(ssrCode), ssrLayout);
    VkPipeline easuPipeline = pipeline(easuSetLayout, module(easuCode), easuLayout);
    VkPipelineLayout msaaLayout = VK_NULL_HANDLE;
    VkPipeline       msaaPipeline = msaaSsr ? pipeline(ssrSetLayout, module(readSpv(msaaSsr)), msaaLayout) : VK_NULL_HANDLE;

    // ---- Resources (X-Plane's are 2D-array images with one layer per eye).
    const VkImageUsageFlags up = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    const VkImageUsageFlags att = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    Img color = makeImage(g, VK_FORMAT_R16G16B16A16_SFLOAT, kRW, kRH, up | att | VK_IMAGE_USAGE_STORAGE_BIT, true);
    Img pos = makeImage(g, VK_FORMAT_R32_SFLOAT, kRW, kRH, up | att, true);
    Img vel = makeImage(g, VK_FORMAT_R32_UINT, kRW, kRH, up | att, true);
    Img depth{};  // --raster: the depth/stencil buffer, sampled by SSR as gbuffer_pos / gbuffer_vel
    VkImageView depthPosView = VK_NULL_HANDLE, stencilVelView = VK_NULL_HANDLE;
    if (rasterVert)
    {
        depth = makeImage(g, VK_FORMAT_D32_SFLOAT_S8_UINT, kRW, kRH, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                          false, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
        for (auto [aspect, view] : {std::pair{VK_IMAGE_ASPECT_DEPTH_BIT, &depthPosView}, std::pair{VK_IMAGE_ASPECT_STENCIL_BIT, &stencilVelView}})
        {
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image = depth.image; vi.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY; vi.format = depth.format;
            vi.subresourceRange = {VkImageAspectFlags(aspect), 0, 1, 0, 1};
            CHECK(vkCreateImageView(g.device, &vi, nullptr, view));
        }
    }
    Img mat = makeImage(g, VK_FORMAT_R8G8B8A8_UNORM, kRW, kRH, up, true);
    Img nrm = makeImage(g, VK_FORMAT_R8G8B8A8_UNORM, kRW, kRH, up, true);
    Img hiz = makeImage(g, VK_FORMAT_R32_SFLOAT, kRW, kRH, up, true);
    Img ssrHistory = makeImage(g, VK_FORMAT_R16G16B16A16_SFLOAT, kRW, kRH, up, true);
    Img brdf = makeImage(g, VK_FORMAT_R8G8B8A8_UNORM, 32, 32, up, false);
    Img ssrOut = makeImage(g, VK_FORMAT_R16G16B16A16_SFLOAT, kRW, kRH, VK_IMAGE_USAGE_STORAGE_BIT, true);
    Img easuOut = makeImage(g, VK_FORMAT_R16G16B16A16_SFLOAT, kDW, kDH,
                            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, true);
    Buf staging = makeBuffer(g, kRW * kRH * (8 + 4 + 4), VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    Buf readback = makeBuffer(g, kDW * kDH * 8, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    // Per-frame uniform slot: u_environment_data +0, u_fsr_data +256, u_gbuffer_data +512, u_ssr_data +1024.
    const VkDeviceSize uboStride = std::max<VkDeviceSize>(2048, g.props.limits.minUniformBufferOffsetAlignment);
    Buf ubo = makeBuffer(g, uboStride * 8, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    VkSampler sampler;
    VkSamplerCreateInfo smp{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    smp.magFilter = smp.minFilter = VK_FILTER_LINEAR;
    smp.addressModeU = smp.addressModeV = smp.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    CHECK(vkCreateSampler(g.device, &smp, nullptr, &sampler));
    VkSampler nearest;
    smp.magFilter = smp.minFilter = VK_FILTER_NEAREST;
    CHECK(vkCreateSampler(g.device, &smp, nullptr, &nearest));

    VkDescriptorPoolSize ps[] = {{CIS, 16}, {ST, 4}, {UB, 4}, {UBD, 8}, {VK_DESCRIPTOR_TYPE_SAMPLER, 2}, {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2}};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 2; dpi.poolSizeCount = 6; dpi.pPoolSizes = ps;
    VkDescriptorPool pool;
    CHECK(vkCreateDescriptorPool(g.device, &dpi, nullptr, &pool));
    VkDescriptorSet ssrSet, easuSet;
    VkDescriptorSetAllocateInfo dsa{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsa.descriptorPool = pool; dsa.descriptorSetCount = 1;
    dsa.pSetLayouts = &ssrSetLayout;
    CHECK(vkAllocateDescriptorSets(g.device, &dsa, &ssrSet));
    dsa.pSetLayouts = &easuSetLayout;
    CHECK(vkAllocateDescriptorSets(g.device, &dsa, &easuSet));
    {
        const VkImageLayout RO = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, DRO = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        std::vector<VkDescriptorImageInfo> ii = {
            {nearest, mat.view, RO}, {nearest, nrm.view, RO},
            {nearest, rasterVert ? depthPosView : pos.view, rasterVert ? DRO : RO}, {nearest, rasterVert ? stencilVelView : vel.view, rasterVert ? DRO : RO},
            {sampler, brdf.view, RO}, {nearest, hiz.view, RO}, {sampler, ssrHistory.view, RO}, {VK_NULL_HANDLE, ssrOut.view, VK_IMAGE_LAYOUT_GENERAL},
            {sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED}, {VK_NULL_HANDLE, color.view, RO}, {VK_NULL_HANDLE, easuOut.view, VK_IMAGE_LAYOUT_GENERAL}};
        std::vector<VkDescriptorBufferInfo> bi = {{ubo.buffer, 0, 128}, {ubo.buffer, 0, 240}, {ubo.buffer, 0, 560}, {ubo.buffer, 0, 80}};
        auto W = [&](VkDescriptorSet s, uint32_t b, VkDescriptorType t, const VkDescriptorImageInfo* i, const VkDescriptorBufferInfo* bu) {
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = s; w.dstBinding = b; w.descriptorCount = 1; w.descriptorType = t; w.pImageInfo = i; w.pBufferInfo = bu;
            return w;
        };
        std::vector<VkWriteDescriptorSet> ws = {
            W(ssrSet, 0, CIS, &ii[0], nullptr), W(ssrSet, 1, CIS, &ii[1], nullptr), W(ssrSet, 3, CIS, &ii[2], nullptr),
            W(ssrSet, 4, CIS, &ii[3], nullptr), W(ssrSet, 6, CIS, &ii[4], nullptr), W(ssrSet, 7, CIS, &ii[5], nullptr),
            W(ssrSet, 8, CIS, &ii[6], nullptr), W(ssrSet, 10, ST, &ii[7], nullptr), W(ssrSet, 2, UB, nullptr, &bi[0]),
            W(ssrSet, 11, UBD, nullptr, &bi[1]), W(ssrSet, 12, UBD, nullptr, &bi[2]),
            W(easuSet, 0, VK_DESCRIPTOR_TYPE_SAMPLER, &ii[8], nullptr), W(easuSet, 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &ii[9], nullptr),
            W(easuSet, 2, ST, &ii[10], nullptr), W(easuSet, 3, UBD, nullptr, &bi[3])};
        vkUpdateDescriptorSets(g.device, uint32_t(ws.size()), ws.data(), 0, nullptr);
    }

    // ---- --raster: G-buffer render pass drawing the scene as geometry.
    VkDescriptorSetLayout loadSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout loadLayout = VK_NULL_HANDLE;
    VkPipeline loadPipeline = VK_NULL_HANDLE;
    VkDescriptorPool loadPool = VK_NULL_HANDLE;  // --load-persistent sets
    std::vector<VkDescriptorPool> loadPoolList;   // per-frame pools
    std::vector<VkDescriptorSet> loadSets;
    double loadAllocUs = 0, loadRecordUs = 0;
    VkRenderPass gbufferPass = VK_NULL_HANDLE;
    VkFramebuffer gbufferFb = VK_NULL_HANDLE;
    VkPipeline rasterPipeline = VK_NULL_HANDLE;
    VkPipelineLayout rasterLayout = VK_NULL_HANDLE;
    std::vector<VkImageView> attachViews;
    if (rasterVert)
    {
        for (Img* im : {&color})
        {
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image = im->image; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = im->format;
            vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VkImageView v;
            CHECK(vkCreateImageView(g.device, &vi, nullptr, &v));
            attachViews.push_back(v);
        }
        attachViews.push_back(depth.view);
        const VkImageLayout RO = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkAttachmentDescription ad[2] = {};
        const VkFormat fmts[2] = {color.format, depth.format};
        for (int i = 0; i < 2; ++i)
        {
            ad[i].format = fmts[i]; ad[i].samples = VK_SAMPLE_COUNT_1_BIT;
            ad[i].loadOp = ad[i].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            ad[i].storeOp = ad[i].stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
            ad[i].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            ad[i].finalLayout = i == 0 ? RO : VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;  // sampled by SSR
        }
        VkAttachmentReference cr[1] = {{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}};
        VkAttachmentReference dr{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription sp{};
        sp.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sp.colorAttachmentCount = 1; sp.pColorAttachments = cr; sp.pDepthStencilAttachment = &dr;
        VkSubpassDependency dep{};
        dep.srcSubpass = 0; dep.dstSubpass = VK_SUBPASS_EXTERNAL;
        dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        dep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dep.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        VkSubpassDependency dep0{};
        dep0.srcSubpass = VK_SUBPASS_EXTERNAL; dep0.dstSubpass = 0;
        dep0.srcStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        dep0.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dep0.srcAccessMask = 0; dep0.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        VkSubpassDependency deps[2] = {dep0, dep};
        VkRenderPassCreateInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        rp.attachmentCount = 2; rp.pAttachments = ad; rp.subpassCount = 1; rp.pSubpasses = &sp; rp.dependencyCount = 2; rp.pDependencies = deps;
        CHECK(vkCreateRenderPass(g.device, &rp, nullptr, &gbufferPass));
        VkFramebufferCreateInfo fb{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fb.renderPass = gbufferPass; fb.attachmentCount = 2; fb.pAttachments = attachViews.data();
        fb.width = kRW; fb.height = kRH; fb.layers = 1;
        CHECK(vkCreateFramebuffer(g.device, &fb, nullptr, &gbufferFb));

        VkPushConstantRange pr{VK_SHADER_STAGE_VERTEX_BIT, 0, 96};
        VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pl.pushConstantRangeCount = 1; pl.pPushConstantRanges = &pr;
        CHECK(vkCreatePipelineLayout(g.device, &pl, nullptr, &rasterLayout));
        VkShaderModule vs = module(readSpv(rasterVert)), fs = module(readSpv(rasterFrag));
        VkPipelineShaderStageCreateInfo st[2] = {
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vs, "main", nullptr},
            {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fs, "main", nullptr}};
        VkPipelineVertexInputStateCreateInfo vin{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount = 1; vp.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = VK_TRUE; ds.depthCompareOp = VK_COMPARE_OP_GREATER;  // reversed Z
        ds.stencilTestEnable = VK_TRUE;  // stencil = gbuffer_vel flags: the reference set per draw
        ds.front = ds.back = {VK_STENCIL_OP_KEEP, VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_ALWAYS, 0xFF, 0xFF, 0};
        VkPipelineColorBlendAttachmentState cba[1] = {};
        cba[0].colorWriteMask = 0xF;
        VkPipelineColorBlendStateCreateInfo cb2{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb2.attachmentCount = 1; cb2.pAttachments = cba;
        VkDynamicState dyn[3] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_STENCIL_REFERENCE};
        VkPipelineDynamicStateCreateInfo dsi{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dsi.dynamicStateCount = 3; dsi.pDynamicStates = dyn;
        VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        gp.stageCount = 2; gp.pStages = st; gp.pVertexInputState = &vin; gp.pInputAssemblyState = &ia; gp.pViewportState = &vp;
        gp.pRasterizationState = &rs; gp.pMultisampleState = &ms; gp.pDepthStencilState = &ds; gp.pColorBlendState = &cb2;
        gp.pDynamicState = &dsi; gp.layout = rasterLayout; gp.renderPass = gbufferPass;
        CHECK(vkCreateGraphicsPipelines(g.device, VK_NULL_HANDLE, 1, &gp, nullptr, &rasterPipeline));
        if (load > 0)
        {
            // The same shaders behind a layout with one dynamic-UBO set, as X-Plane's draws bind.
            VkDescriptorSetLayoutBinding lb{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
            VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            lci.bindingCount = 1; lci.pBindings = &lb;
            CHECK(vkCreateDescriptorSetLayout(g.device, &lci, nullptr, &loadSetLayout));
            VkPipelineLayoutCreateInfo lpl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            lpl.setLayoutCount = 1; lpl.pSetLayouts = &loadSetLayout; lpl.pushConstantRangeCount = 1; lpl.pPushConstantRanges = &pr;
            CHECK(vkCreatePipelineLayout(g.device, &lpl, nullptr, &loadLayout));
            gp.layout = loadLayout;
            CHECK(vkCreateGraphicsPipelines(g.device, VK_NULL_HANDLE, 1, &gp, nullptr, &loadPipeline));
            const uint32_t perPool = uint32_t((load + loadPools - 1) / loadPools);
            VkDescriptorPoolSize lps{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, perPool};
            VkDescriptorPoolCreateInfo lpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            lpi.maxSets = perPool; lpi.poolSizeCount = 1; lpi.pPoolSizes = &lps;
            loadPoolList.resize(loadPools);
            for (auto& lp : loadPoolList) CHECK(vkCreateDescriptorPool(g.device, &lpi, nullptr, &lp));
            loadSets.resize(load);
            if (loadPersistent > 0)
            {
                VkDescriptorPoolSize pps{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, uint32_t(loadPersistent)};
                VkDescriptorPoolCreateInfo ppi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
                ppi.maxSets = uint32_t(loadPersistent); ppi.poolSizeCount = 1; ppi.pPoolSizes = &pps;
                CHECK(vkCreateDescriptorPool(g.device, &ppi, nullptr, &loadPool));
                std::vector<VkDescriptorSetLayout> pl(loadPersistent, loadSetLayout);
                std::vector<VkDescriptorSet> ps(loadPersistent);
                VkDescriptorSetAllocateInfo pa{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
                pa.descriptorPool = loadPool; pa.descriptorSetCount = uint32_t(loadPersistent); pa.pSetLayouts = pl.data();
                CHECK(vkAllocateDescriptorSets(g.device, &pa, ps.data()));
            }
        }
    }

    VkCommandPool cmdPool;
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; cpi.queueFamilyIndex = g.family;
    CHECK(vkCreateCommandPool(g.device, &cpi, nullptr, &cmdPool));
    VkCommandBuffer cbs[2];  // [1]: EASU onwards with --split
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = cmdPool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 2;
    CHECK(vkAllocateCommandBuffers(g.device, &cai, cbs));
    VkCommandBuffer cb = cbs[0];
    VkCommandBuffer msaaCb = VK_NULL_HANDLE;  // --msaa-ssr: recorded every frame, never submitted
    cai.commandBufferCount = 1;
    if (msaaSsr) CHECK(vkAllocateCommandBuffers(g.device, &cai, &msaaCb));
    VkFence fence;
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    CHECK(vkCreateFence(g.device, &fci, nullptr, &fence));
    VkSemaphore acquired;
    std::vector<VkSemaphore> rendered(swapImages.size());  // one per swapchain image: presentation holds it
    VkSemaphoreCreateInfo sem{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    CHECK(vkCreateSemaphore(g.device, &sem, nullptr, &acquired));
    for (auto& r : rendered) CHECK(vkCreateSemaphore(g.device, &sem, nullptr, &r));

    // ---- --load-threads: the load on worker threads (see the header).
    struct LoadWorker {
        VkDescriptorPool pool{};
        VkCommandPool cmdPool{};
        VkCommandBuffer cb{};
        std::vector<VkDescriptorSet> sets, tracked;
    };
    std::vector<LoadWorker> workers;
    VkDescriptorSetLayout trackedSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout trackedLayout = VK_NULL_HANDLE;
    VkDescriptorUpdateTemplate uboTemplate = VK_NULL_HANDLE, trackedTemplate = VK_NULL_HANDLE;
    std::mutex loadMutex;
    std::condition_variable loadStart, loadDone;
    uint64_t loadGeneration = 0;
    int loadRemaining = 0;
    bool loadQuit = false;
    std::vector<std::thread> loadThreadList;
    double loadThreadsUs = 0;
    if (loadThreads > 0 && load > 0 && rasterVert)
    {
        constexpr int kBindsPerSet = 7, kBuffersPerFrame = 8;
        VkDescriptorSetLayoutBinding tb{0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo tci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        tci.bindingCount = 1; tci.pBindings = &tb;
        CHECK(vkCreateDescriptorSetLayout(g.device, &tci, nullptr, &trackedSetLayout));
        VkPipelineLayoutCreateInfo tpl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        tpl.setLayoutCount = 1; tpl.pSetLayouts = &trackedSetLayout;
        CHECK(vkCreatePipelineLayout(g.device, &tpl, nullptr, &trackedLayout));
        auto makeTemplate = [&](VkDescriptorSetLayout layout, VkDescriptorType type, size_t stride, VkDescriptorUpdateTemplate* out) {
            VkDescriptorUpdateTemplateEntry e{0, 0, 1, type, 0, stride};
            VkDescriptorUpdateTemplateCreateInfo ti{VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO};
            ti.descriptorUpdateEntryCount = 1; ti.pDescriptorUpdateEntries = &e;
            ti.templateType = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET; ti.descriptorSetLayout = layout;
            CHECK(vkCreateDescriptorUpdateTemplate(g.device, &ti, nullptr, out));
        };
        makeTemplate(loadSetLayout, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, sizeof(VkDescriptorBufferInfo), &uboTemplate);
        makeTemplate(trackedSetLayout, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, sizeof(VkDescriptorImageInfo), &trackedTemplate);
        const uint32_t perThread = uint32_t((load + loadThreads - 1) / loadThreads);
        const uint32_t trackedN = perThread / 2, plainN = perThread - trackedN;
        workers.resize(loadThreads);
        for (auto& w : workers)
        {
            VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, std::max(1u, plainN)},
                                          {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, std::max(1u, trackedN)}};
            VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            dpi.maxSets = perThread; dpi.poolSizeCount = 2; dpi.pPoolSizes = ps;
            CHECK(vkCreateDescriptorPool(g.device, &dpi, nullptr, &w.pool));
            VkCommandPoolCreateInfo wci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            wci.queueFamilyIndex = g.family;
            CHECK(vkCreateCommandPool(g.device, &wci, nullptr, &w.cmdPool));
            VkCommandBufferAllocateInfo wai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            wai.commandPool = w.cmdPool; wai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; wai.commandBufferCount = 1;
            CHECK(vkAllocateCommandBuffers(g.device, &wai, &w.cb));
            w.sets.resize(plainN);
            w.tracked.resize(trackedN);
        }
        auto work = [&, plainN, trackedN](LoadWorker& w) {
            CHECK(vkResetDescriptorPool(g.device, w.pool, 0));
            std::vector<VkDescriptorSetLayout> pl(plainN, loadSetLayout), tl(trackedN, trackedSetLayout);
            VkDescriptorSetAllocateInfo a{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            a.descriptorPool = w.pool;
            a.descriptorSetCount = plainN; a.pSetLayouts = pl.data();
            if (plainN) CHECK(vkAllocateDescriptorSets(g.device, &a, w.sets.data()));
            a.descriptorSetCount = trackedN; a.pSetLayouts = tl.data();
            if (trackedN) CHECK(vkAllocateDescriptorSets(g.device, &a, w.tracked.data()));
            const VkDescriptorBufferInfo bufferInfo{ubo.buffer, 0, 64};
            const VkDescriptorImageInfo imageInfo{VK_NULL_HANDLE, ssrOut.view, VK_IMAGE_LAYOUT_GENERAL};
            for (auto s : w.sets) vkUpdateDescriptorSetWithTemplate(g.device, s, uboTemplate, &bufferInfo);
            for (auto s : w.tracked) vkUpdateDescriptorSetWithTemplate(g.device, s, trackedTemplate, &imageInfo);
            for (int b = 0; b < kBuffersPerFrame; ++b)
            {
                VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
                bci.size = 256; bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
                VkBuffer buffer;
                CHECK(vkCreateBuffer(g.device, &bci, nullptr, &buffer));
                vkDestroyBuffer(g.device, buffer, nullptr);
            }
            CHECK(vkResetCommandPool(g.device, w.cmdPool, 0));
            VkCommandBufferBeginInfo wb{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            wb.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            CHECK(vkBeginCommandBuffer(w.cb, &wb));
            const uint32_t off = 0;
            for (auto s : w.sets)
            {
                vkCmdBindPipeline(w.cb, VK_PIPELINE_BIND_POINT_GRAPHICS, loadPipeline);
                for (int k = 0; k < kBindsPerSet; ++k) vkCmdBindDescriptorSets(w.cb, VK_PIPELINE_BIND_POINT_GRAPHICS, loadLayout, 0, 1, &s, 1, &off);
            }
            for (auto s : w.tracked)
                for (int k = 0; k < kBindsPerSet; ++k) vkCmdBindDescriptorSets(w.cb, VK_PIPELINE_BIND_POINT_COMPUTE, trackedLayout, 0, 1, &s, 0, nullptr);
            CHECK(vkEndCommandBuffer(w.cb));
        };
        for (auto& w : workers)
            loadThreadList.emplace_back([&, work, worker = &w]() {
                uint64_t seen = 0;
                for (;;)
                {
                    {
                        std::unique_lock<std::mutex> l(loadMutex);
                        loadStart.wait(l, [&] { return loadQuit || loadGeneration != seen; });
                        if (loadQuit) return;
                        seen = loadGeneration;
                    }
                    work(*worker);
                    std::lock_guard<std::mutex> l(loadMutex);
                    if (--loadRemaining == 0) loadDone.notify_one();
                }
            });
    }

    // ---- Uniform contents. Projection: reversed-Z infinite, clip = (x f/a, y f, near, -z).
    float proj[16] = {};
    proj[0] = kF / kAspect; proj[5] = kF; proj[14] = kNear; proj[11] = -1.0f;
    const Vec3 velocity = {0.0137f, 0.0041f, 0.0f};  // metres per frame; ~0.43, 0.13 render px at the wall

    // FAKE_START_DELAY=<seconds>: idle between device creation and the first frame, like X-Plane's
    // loading screen (tests the layer's status window).
    if (const char* delay = std::getenv("FAKE_START_DELAY")) std::this_thread::sleep_for(std::chrono::seconds(atoi(delay)));
    for (int f = 0; f < frames; ++f)
    {
        const Vec3 cam = {velocity.x * f, velocity.y * f, 0};
        g_f     = kF * std::pow(1.0f + zoom, float(f));
        proj[0] = g_f / kAspect; proj[5] = g_f;
        // CPU scene at render resolution (uploaded unless --raster draws it).
        auto* c16 = reinterpret_cast<uint16_t*>(staging.mapped);
        auto* d32 = reinterpret_cast<float*>(staging.mapped + kRW * kRH * 8);
        auto* v32 = reinterpret_cast<uint32_t*>(staging.mapped + kRW * kRH * 12);
        for (uint32_t j = 0; j < (rasterVert ? 0u : kRH); ++j)
            for (uint32_t i = 0; i < kRW; ++i)
            {
                float c[3];
                uint32_t flags;
                const float d = shade((i + 0.5f) / kRW, (j + 0.5f) / kRH, cam, c, flags);
                const size_t p = j * kRW + i;
                c16[p * 4 + 0] = half(c[0]); c16[p * 4 + 1] = half(c[1]); c16[p * 4 + 2] = half(c[2]); c16[p * 4 + 3] = half(1.0f);
                d32[p] = d;
                v32[p] = flags;
            }
        // Uniform ring slot for this frame.
        const uint32_t slot = f % 8;
        uint8_t* u = ubo.mapped + slot * uboStride;
        std::memset(u, 0, uboStride);
        float* gb = reinterpret_cast<float*>(u + 512);   // u_gbuffer_data
        gb[0] = kRW; gb[1] = kRH; gb[2] = 1.0f / kRW; gb[3] = 1.0f / kRH;
        gb[4] = kRW; gb[5] = kRH; gb[6] = 1.0f / kRW; gb[7] = 1.0f / kRH;
        gb[8] = 2 * kAspect / g_f; gb[9] = 2 / g_f; gb[10] = -kAspect / g_f; gb[11] = -1 / g_f;  // u_proj_coefficients[0]
        gb[16] = 0; gb[17] = -kNear; gb[18] = 1; gb[19] = 0;                                 // u_clip_info[0]: z = -near / d
        for (int e = 0; e < 2; ++e) for (int k = 0; k < 4; ++k) gb[28 + e * 16 + k * 5] = 1.0f;  // u_head_to_eye = identity
        float* ssr = reinterpret_cast<float*>(u + 1024); // u_ssr_data
        for (int e = 0; e < 2; ++e) std::memcpy(ssr + e * 16, proj, 64);
        // u_reproj: current view -> previous view. The camera moved by +velocity, so a world point's
        // view position last frame was (current + velocity).
        const float s = flip ? -1.0f : 1.0f;
        for (int e = 0; e < 2; ++e)
        {
            float* r = ssr + 32 + e * 16;
            r[0] = r[5] = r[10] = r[15] = 1.0f;
            r[12] = s * velocity.x; r[13] = s * velocity.y; r[14] = s * velocity.z;
            float* l = ssr + 64 + e * 16;  // u_local_reproj: the panel moves with the camera
            l[0] = l[5] = l[10] = l[15] = 1.0f;
        }
        reinterpret_cast<int32_t*>(u + 1024)[129] = kRW;  // u_hiz_width (offset 516)
        reinterpret_cast<int32_t*>(u + 1024)[130] = kRH;  // u_hiz_height (offset 520)
        // u_fsr_data: EASU constants as AMD's FsrEasuCon computes them.
        uint32_t* con = reinterpret_cast<uint32_t*>(u + 256);
        const float iw = float(kRW), ih = float(kRH), ow = float(kDW), oh = float(kDH);
        float k[16] = {iw / ow, ih / oh, 0.5f * iw / ow - 0.5f, 0.5f * ih / oh - 0.5f,
                       1 / iw, 1 / ih, 1 / iw, -1 / ih,
                       -1 / iw, 2 / ih, 1 / iw, 2 / ih,
                       0 / iw, 4 / ih, 0, 0};
        for (int q = 0; q < 16; ++q) con[q] = fbits(k[q]);
        const uint32_t ssrOffsets[2] = {uint32_t(slot * uboStride + 512), uint32_t(slot * uboStride + 1024)};
        const uint32_t easuOffset = uint32_t(slot * uboStride + 256);

        uint32_t imageIndex;
        CHECK(vkAcquireNextImageKHR(g.device, swapchain, UINT64_MAX, acquired, VK_NULL_HANDLE, &imageIndex));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        CHECK(vkBeginCommandBuffer(cb, &bi));
        const auto loadT0 = std::chrono::steady_clock::now();
        if (!workers.empty())
        {
            {
                std::lock_guard<std::mutex> l(loadMutex);
                ++loadGeneration;
                loadRemaining = loadThreads;
            }
            loadStart.notify_all();
        }
        const VkImageLayout RO = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, TD = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        if (f == 0)
        {
            VkClearColorValue zero{};
            VkImageSubresourceRange all{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            for (Img* im : {&mat, &nrm, &hiz, &ssrHistory, &brdf})
            {
                barrier(cb, im->image, VK_IMAGE_LAYOUT_UNDEFINED, TD);
                vkCmdClearColorImage(cb, im->image, TD, &zero, 1, &all);
                barrier(cb, im->image, TD, RO);
            }
            barrier(cb, ssrOut.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
            barrier(cb, easuOut.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        }
        if (rasterVert)
        {
            // Viewport set before the pass begins (the layer must re-issue it jittered inside).
            VkViewport viewport{0, 0, float(kRW), float(kRH), 0, 1};
            VkRect2D scissor{{0, 0}, {kRW, kRH}};
            vkCmdSetViewport(cb, 0, 1, &viewport);
            vkCmdSetScissor(cb, 0, 1, &scissor);
            VkClearValue clears[2] = {};  // colour 0; depth 0 (reversed-Z far = sky), stencil 0
            VkRenderPassBeginInfo rb{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
            rb.renderPass = gbufferPass; rb.framebuffer = gbufferFb; rb.renderArea = {{0, 0}, {kRW, kRH}};
            rb.clearValueCount = 2; rb.pClearValues = clears;
            vkCmdBeginRenderPass(cb, &rb, VK_SUBPASS_CONTENTS_INLINE);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, rasterPipeline);
            struct { float proj[16]; float cam[4]; float frust[4]; } push;
            std::memcpy(push.proj, proj, 64);
            push.cam[0] = cam.x; push.cam[1] = cam.y; push.cam[2] = cam.z; push.cam[3] = 0;
            push.frust[0] = kAspect / g_f; push.frust[1] = 1 / g_f; push.frust[2] = push.frust[3] = 0;
            vkCmdPushConstants(cb, rasterLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push), &push);
            vkCmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_AND_BACK, 0);  // wall: world
            vkCmdDraw(cb, 6, 1, 0, 0);
            vkCmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_AND_BACK, 4);  // panel: own aircraft
            vkCmdDraw(cb, 6, 1, 6, 0);
            if (load > 0 && workers.empty())
            {
                // Per-frame descriptor churn, then the draw-call stream (both timed: the layer's CPU cost).
                auto t0 = std::chrono::steady_clock::now();
                std::vector<VkDescriptorSetLayout> layouts(load, loadSetLayout);
                const int perPool = (load + loadPools - 1) / loadPools;
                for (int pi = 0; pi < loadPools; ++pi)
                {
                    CHECK(vkResetDescriptorPool(g.device, loadPoolList[pi], 0));
                    const int first = pi * perPool, n = std::min(perPool, load - first);
                    if (n <= 0) continue;
                    VkDescriptorSetAllocateInfo la{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
                    la.descriptorPool = loadPoolList[pi]; la.descriptorSetCount = uint32_t(n); la.pSetLayouts = layouts.data();
                    CHECK(vkAllocateDescriptorSets(g.device, &la, loadSets.data() + first));
                }
                VkDescriptorBufferInfo lbuf{ubo.buffer, 0, 64};
                for (int i = 0; i < load; ++i)
                {
                    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                    w.dstSet = loadSets[i]; w.dstBinding = 0; w.descriptorCount = 1;
                    w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC; w.pBufferInfo = &lbuf;
                    vkUpdateDescriptorSets(g.device, 1, &w, 0, nullptr);
                }
                auto t1 = std::chrono::steady_clock::now();
                VkViewport lv{0, 0, float(kRW), float(kRH), 0, 1};
                for (int i = 0; i < load; ++i)
                {
                    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, loadPipeline);
                    const uint32_t off = 0;
                    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, loadLayout, 0, 1, &loadSets[i], 1, &off);
                    vkCmdSetViewport(cb, 0, 1, &lv);
                    vkCmdDraw(cb, 0, 1, 0, 0);
                }
                auto t2 = std::chrono::steady_clock::now();
                if (f > 0)  // the first frame includes one-time costs
                {
                    loadAllocUs += std::chrono::duration<double, std::micro>(t1 - t0).count();
                    loadRecordUs += std::chrono::duration<double, std::micro>(t2 - t1).count();
                }
            }
            vkCmdEndRenderPass(cb);
        }
        else
        {
            for (Img* im : {&color, &pos, &vel}) barrier(cb, im->image, f == 0 ? VK_IMAGE_LAYOUT_UNDEFINED : RO, TD);
            VkBufferImageCopy copy{};
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageExtent = {kRW, kRH, 1};
            vkCmdCopyBufferToImage(cb, staging.buffer, color.image, TD, 1, &copy);
            copy.bufferOffset = kRW * kRH * 8;
            vkCmdCopyBufferToImage(cb, staging.buffer, pos.image, TD, 1, &copy);
            copy.bufferOffset = kRW * kRH * 12;
            vkCmdCopyBufferToImage(cb, staging.buffer, vel.image, TD, 1, &copy);
            for (Img* im : {&color, &pos, &vel}) barrier(cb, im->image, TD, RO);
        }

        // X-Plane's order: scene SSR, then FSR 1 EASU (then RCAS, not needed here).
        if (msaaSsr)
        {
            VkCommandBufferBeginInfo mbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            CHECK(vkBeginCommandBuffer(msaaCb, &mbi));
            vkCmdBindPipeline(msaaCb, VK_PIPELINE_BIND_POINT_COMPUTE, msaaPipeline);
            vkCmdDispatch(msaaCb, (kRW + 15) / 16, (kRH + 15) / 16, 1);
            CHECK(vkEndCommandBuffer(msaaCb));
        }
        else
        {
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, ssrPipeline);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, ssrLayout, 0, 1, &ssrSet, 2, ssrOffsets);
            vkCmdDispatch(cb, (kRW + 15) / 16, (kRH + 15) / 16, 1);
        }
        barrier(cb, ssrOut.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
        if (split)
        {
            CHECK(vkEndCommandBuffer(cb));
            cb = cbs[1];
            CHECK(vkBeginCommandBuffer(cb, &bi));
        }
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, easuPipeline);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, easuLayout, 0, 1, &easuSet, 1, &easuOffset);
        vkCmdDispatch(cb, (kDW + 15) / 16, (kDH + 15) / 16, 1);
        if (f == frames - 1)
        {
            barrier(cb, easuOut.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            VkBufferImageCopy rc{};
            rc.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            rc.imageExtent = {kDW, kDH, 1};
            vkCmdCopyImageToBuffer(cb, easuOut.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback.buffer, 1, &rc);
            barrier(cb, easuOut.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
        }
        barrier(cb, swapImages[imageIndex], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
        CHECK(vkEndCommandBuffer(cb));
        cb = cbs[0];
        if (!workers.empty())
        {
            std::unique_lock<std::mutex> l(loadMutex);
            loadDone.wait(l, [&] { return loadRemaining == 0; });
            if (f > 0) loadThreadsUs += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - loadT0).count();
        }
        const VkCommandBuffer order[2] = {cbs[reversed ? 1 : 0], cbs[reversed ? 0 : 1]};
        const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.waitSemaphoreCount = 1; si.pWaitSemaphores = &acquired; si.pWaitDstStageMask = &waitStage;
        si.commandBufferCount = split ? 2 : 1; si.pCommandBuffers = split ? order : &cbs[0];
        si.signalSemaphoreCount = 1; si.pSignalSemaphores = &rendered[imageIndex];
        CHECK(vkQueueSubmit(g.queue, 1, &si, fence));
        VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        pi.waitSemaphoreCount = 1; pi.pWaitSemaphores = &rendered[imageIndex];
        pi.swapchainCount = 1; pi.pSwapchains = &swapchain; pi.pImageIndices = &imageIndex;
        CHECK(vkQueuePresentKHR(g.queue, &pi));
        CHECK(vkWaitForFences(g.device, 1, &fence, VK_TRUE, UINT64_MAX));
        CHECK(vkResetFences(g.device, 1, &fence));
    }

    // ---- Compare the final frame with a supersampled ground truth.
    std::vector<float> result(kDW * kDH * 3), truth(kDW * kDH * 3);
    auto* r16 = reinterpret_cast<uint16_t*>(readback.mapped);
    for (uint32_t p = 0; p < kDW * kDH; ++p)
        for (int k = 0; k < 3; ++k) result[p * 3 + k] = fromHalf(r16[p * 4 + k]);
    const Vec3 cam = {velocity.x * (frames - 1), velocity.y * (frames - 1), 0};
    for (uint32_t y = 0; y < kDH; ++y)
        for (uint32_t x = 0; x < kDW; ++x)
        {
            float acc[3] = {0, 0, 0};
            for (int b = 0; b < 4; ++b)
                for (int a = 0; a < 4; ++a)
                {
                    float c[3];
                    uint32_t flags;
                    shade((x + (a + 0.5f) / 4) / kDW, (y + (b + 0.5f) / 4) / kDH, cam, c, flags);
                    for (int k = 0; k < 3; ++k) acc[k] += c[k] / 16;
                }
            for (int k = 0; k < 3; ++k) truth[(y * kDW + x) * 3 + k] = acc[k];
        }
    const uint32_t wallEnd = uint32_t(kDH * 0.68f), panelStart = uint32_t(kDH * kPanelRow) + 8;
    const double pw = psnr(result, truth, kDW, 32, wallEnd, 32), pp = psnr(result, truth, kDW, panelStart, kDH - 16, 32);
    if (!out.empty())
    {
        writePPM(out + "/output.ppm", result, kDW, kDH);
        writePPM(out + "/truth.ppm", truth, kDW, kDH);
    }
    if (load > 0 && frames > 1)
        printf("{\"load\":%d,\"load_alloc_us\":%.1f,\"load_record_us\":%.1f,\"load_threads\":%d,\"load_threads_us\":%.1f}\n", load,
               loadAllocUs / (frames - 1), loadRecordUs / (frames - 1), int(workers.size()), loadThreadsUs / (frames - 1));
    printf("{\"zoom\":%.4f,\"psnr_wall\":%.3f,\"psnr_panel\":%.3f,\"validation_errors\":%d,\"frames\":%d,\"flip\":%s,\"device\":\"%s\"}\n", zoom, pw, pp,
           g_validationErrors, frames, flip ? "true" : "false", g.props.deviceName);

    // Destroy everything this program created, so that any object the validation layer still
    // reports at vkDestroyDevice belongs to the FSR layer.
    vkDeviceWaitIdle(g.device);
    {
        std::lock_guard<std::mutex> l(loadMutex);
        loadQuit = true;
    }
    loadStart.notify_all();
    for (auto& t : loadThreadList) t.join();
    for (auto& w : workers)
    {
        vkDestroyCommandPool(g.device, w.cmdPool, nullptr);
        vkDestroyDescriptorPool(g.device, w.pool, nullptr);
    }
    if (uboTemplate) vkDestroyDescriptorUpdateTemplate(g.device, uboTemplate, nullptr);
    if (trackedTemplate) vkDestroyDescriptorUpdateTemplate(g.device, trackedTemplate, nullptr);
    if (trackedLayout) vkDestroyPipelineLayout(g.device, trackedLayout, nullptr);
    if (trackedSetLayout) vkDestroyDescriptorSetLayout(g.device, trackedSetLayout, nullptr);
    vkDestroySemaphore(g.device, acquired, nullptr);
    for (auto r : rendered) vkDestroySemaphore(g.device, r, nullptr);
    vkDestroyFence(g.device, fence, nullptr);
    vkDestroyCommandPool(g.device, cmdPool, nullptr);
    vkDestroyDescriptorPool(g.device, pool, nullptr);
    vkDestroySampler(g.device, sampler, nullptr);
    vkDestroySampler(g.device, nearest, nullptr);
    for (Buf* b : {&staging, &readback, &ubo}) { vkDestroyBuffer(g.device, b->buffer, nullptr); vkFreeMemory(g.device, b->memory, nullptr); }
    for (Img* im : {&color, &pos, &vel, &mat, &nrm, &hiz, &ssrHistory, &brdf, &ssrOut, &easuOut})
    {
        vkDestroyImageView(g.device, im->view, nullptr);
        vkDestroyImage(g.device, im->image, nullptr);
        vkFreeMemory(g.device, im->memory, nullptr);
    }
    vkDestroyPipeline(g.device, ssrPipeline, nullptr);
    if (msaaSsr)
    {
        vkDestroyPipeline(g.device, msaaPipeline, nullptr);
        vkDestroyPipelineLayout(g.device, msaaLayout, nullptr);
    }
    vkDestroyPipeline(g.device, easuPipeline, nullptr);
    vkDestroyPipelineLayout(g.device, ssrLayout, nullptr);
    vkDestroyPipelineLayout(g.device, easuLayout, nullptr);
    vkDestroyDescriptorSetLayout(g.device, ssrSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(g.device, easuSetLayout, nullptr);
    for (auto m : modules) vkDestroyShaderModule(g.device, m, nullptr);
    if (rasterVert)
    {
        vkDestroyPipeline(g.device, rasterPipeline, nullptr);
        if (loadPipeline) vkDestroyPipeline(g.device, loadPipeline, nullptr);
        if (loadLayout) vkDestroyPipelineLayout(g.device, loadLayout, nullptr);
        if (loadSetLayout) vkDestroyDescriptorSetLayout(g.device, loadSetLayout, nullptr);
        if (loadPool) vkDestroyDescriptorPool(g.device, loadPool, nullptr);
        for (auto lp : loadPoolList) vkDestroyDescriptorPool(g.device, lp, nullptr);
        vkDestroyPipelineLayout(g.device, rasterLayout, nullptr);
        vkDestroyFramebuffer(g.device, gbufferFb, nullptr);
        vkDestroyRenderPass(g.device, gbufferPass, nullptr);
        for (auto v : attachViews) if (v != depth.view) vkDestroyImageView(g.device, v, nullptr);
        vkDestroyImageView(g.device, depthPosView, nullptr);
        vkDestroyImageView(g.device, stencilVelView, nullptr);
        vkDestroyImageView(g.device, depth.view, nullptr);
        vkDestroyImage(g.device, depth.image, nullptr);
        vkFreeMemory(g.device, depth.memory, nullptr);
    }
    vkDestroySwapchainKHR(g.device, swapchain, nullptr);
    vkDestroyDevice(g.device, nullptr);  // the layer's engine is destroyed here
    vkDestroySurfaceKHR(g.instance, surface, nullptr);
    if (g.messenger)
        reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(g.instance, "vkDestroyDebugUtilsMessengerEXT"))(
            g.instance, g.messenger, nullptr);
    vkDestroyInstance(g.instance, nullptr);
    return 0;
}
