#include "engine.h"

#include "ffx_host_vk.h"
#include "layer_shaders.h"  // generated: kInputs2D, kInputsArray, kCopyIn, kCopyOut

#include <FidelityFX/host/backends/vk/ffx_vk.h>
#include <FidelityFX/host/ffx_fsr3upscaler.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace xpfsr {
namespace {

constexpr uint32_t kFrames           = 8;     // ring depth: descriptor pools, telemetry, deferred deletion
constexpr uint32_t kSetsPerFrame     = 16;
constexpr uint32_t kTelemetryStride  = 512;
constexpr uint32_t kTelemetryLatency = 4;     // frames before a record is read back
constexpr float    kNear             = 0.05f; // layer depth convention: near / distance (X-Plane units, metres)

// Mirrors the Telemetry block in inputs.comp / copy_in.comp (std430).
struct Telemetry {
    uint32_t tag, skyPixels, localPixels, pad0;
    float    proj[16], reproj[16], localReproj[16];
    float    clipInfo[4], projCoef[4], viewport[4], gbufferSize[4], centre[4];
    uint32_t easuConst[20];
};
static_assert(sizeof(Telemetry) == 368, "Telemetry must match the shader block");

struct PushInputs { int32_t w, h, eye; float nearPlane; uint32_t tag, slot, havePrevious; };
struct PushCopy { int32_t w, h, layer; uint32_t pad; };

std::string num(double v)
{
    std::ostringstream o;
    o.precision(9);
    o << v;
    return o.str();
}

std::string mat(const float* m)
{
    std::string s = "[";
    for (int i = 0; i < 16; ++i)
        s += (i ? "," : "") + num(m[i]);
    return s + "]";
}

}  // namespace

struct Engine::Impl {
    DeviceSetup& setup;
    VkDevice     device;

#define FN(name) PFN_##name name = nullptr
    FN(vkCreateImage); FN(vkDestroyImage); FN(vkGetImageMemoryRequirements); FN(vkAllocateMemory); FN(vkFreeMemory);
    FN(vkBindImageMemory); FN(vkCreateImageView); FN(vkDestroyImageView); FN(vkCreateBuffer); FN(vkDestroyBuffer);
    FN(vkGetBufferMemoryRequirements); FN(vkBindBufferMemory); FN(vkMapMemory); FN(vkCreateShaderModule);
    FN(vkDestroyShaderModule); FN(vkCreateDescriptorSetLayout); FN(vkDestroyDescriptorSetLayout); FN(vkCreatePipelineLayout);
    FN(vkDestroyPipelineLayout); FN(vkCreateComputePipelines); FN(vkDestroyPipeline); FN(vkCreateDescriptorPool);
    FN(vkDestroyDescriptorPool); FN(vkResetDescriptorPool); FN(vkAllocateDescriptorSets); FN(vkUpdateDescriptorSets);
    FN(vkCmdBindPipeline); FN(vkCmdBindDescriptorSets); FN(vkCmdPushConstants); FN(vkCmdDispatch); FN(vkCmdPipelineBarrier);
    FN(vkGetPhysicalDeviceMemoryProperties);
    FN(vkCreateQueryPool); FN(vkDestroyQueryPool); FN(vkCmdResetQueryPool); FN(vkCmdWriteTimestamp); FN(vkGetQueryPoolResults);
#undef FN

    VkPhysicalDeviceMemoryProperties memory{};

    struct Pass {
        VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
        VkPipelineLayout      layout    = VK_NULL_HANDLE;
        VkPipeline            pipeline  = VK_NULL_HANDLE;
    } inputs2D, inputsArray, copyIn, copyOut;

    VkDescriptorPool pools[kFrames]{};
    uint64_t         poolEpoch[kFrames];
    VkBuffer         projHistoryBuffer = VK_NULL_HANDLE;  // inputs.comp ProjHistory: u_proj of the last two SSR passes
    VkDeviceMemory   projHistoryMemory = VK_NULL_HANDLE;
    bool             projHistoryValid  = false;
    VkBuffer         telemetryBuffer = VK_NULL_HANDLE;
    VkDeviceMemory   telemetryMemory = VK_NULL_HANDLE;
    uint8_t*         telemetryMapped = nullptr;

    struct Image {
        VkImage           image  = VK_NULL_HANDLE;
        VkDeviceMemory    memory = VK_NULL_HANDLE;
        VkImageView       view   = VK_NULL_HANDLE;
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        VkImageLayout     layout = VK_IMAGE_LAYOUT_UNDEFINED;
        explicit operator bool() const { return image != VK_NULL_HANDLE; }
    };
    Image depth, motion;  // written at SSR, read by FSR

    struct Fsr {
        std::unique_ptr<uint8_t, decltype(&free)> scratch{nullptr, &free};
        std::unique_ptr<FfxFsr3UpscalerContext>  context;
        VkExtent2D render{}, display{};
        Image      color, output, dilatedDepth, dilatedMotion, prevNearestDepth;
        bool       reset = true;
    };
    std::unique_ptr<Fsr> fsr;

    // Deferred destruction: run once the GPU can no longer be using the objects.
    std::vector<std::pair<uint64_t, std::function<void()>>> graveyard;

    uint64_t        presents = 0;
    uint64_t        inputsSeq = 0, consumedSeq = 0;  // newest inputs recorded; newest read by FSR
    uint64_t        inputsRecording = 0;             // the recording that wrote inputsSeq
    VkExtent2D      inputsExtent{};
    // Submission order (onSubmitted): the newest inputs written by a submitted command buffer.
    uint64_t        submittedSeq = 0, submittedCall = 0, submittedRecording = 0;
    VkQueue         submittedQueue = VK_NULL_HANDLE;
    uint64_t        readsChecked = 0, readsInOrder = 0, readsMissing = 0, readsOverwritten = 0, readsCrossQueue = 0;
    uint64_t        fsrRuns = 0, sameRecordingReads = 0, telemetryValid = 0;
    // GPU timestamps: per frame slot, pairs for [0] inputs pass, [1] FSR 3.1 (copy in .. copy out),
    // [2] X-Plane's EASU when it runs unchanged; read back kTelemetryLatency frames later.
    enum { kTsInputs, kTsFsr, kTsEasu, kTsKinds };
    VkQueryPool     queries = VK_NULL_HANDLE;
    float           timestampPeriodNs = 1.0f;
    uint64_t        tsFrame[kFrames][kTsKinds];
    double          gpuMsSum[kTsKinds] = {}, abGpuMsSum[kTsKinds] = {};
    uint64_t        gpuCount[kTsKinds] = {}, abGpuCount[kTsKinds] = {};
    // A/B (setup.abSeconds): phase 0 = FSR 3.1, phase 1 = X-Plane's FSR 1 (EASU unchanged, no jitter,
    // no inputs pass). Starts at the first EASU dispatch; the first 2 s of a phase are not counted.
    int             abPhase = 0;
    std::chrono::steady_clock::time_point abStart{}, lastPresentTime{};
    double          abFrameMsSum = 0;
    uint64_t        abFrames = 0;
    std::chrono::steady_clock::time_point lastSummary{};
    float           fovY = 0.0f;  // radians, from telemetry; 0 until known
    int32_t         phaseCount = 0;              // jitter sequence length, from the first FSR context
    uint64_t        jitterAppliedFrame = UINT64_MAX;
    uint64_t        jitteredFrames = 0;
    std::chrono::steady_clock::time_point lastFsr{};
    uint64_t        ssrCount = 0, easuCount = 0, replaced = 0;
    std::string     lastReason;
    uint64_t        statusEasu = 0, statusReplaced = 0;  // counts at the previous status
    std::chrono::steady_clock::time_point statusTime = std::chrono::steady_clock::now();
    bool            statusIdle = true;  // no EASU since the previous status: the next one restarts the window
    std::string     statusReason;                        // newest refusal since the previous status
    uint64_t        msaaPresent = UINT64_MAX;            // presents at the newest MSAA scene SSR

    explicit Impl(DeviceSetup& s) : setup(s), device(s.device)
    {
        std::fill(std::begin(poolEpoch), std::end(poolEpoch), UINT64_MAX);
        for (auto& f : tsFrame) std::fill(std::begin(f), std::end(f), UINT64_MAX);
    }

    bool fsr1Phase() const { return setup.abSeconds > 0 && abPhase == 1; }

    void stamp(VkCommandBuffer cmd, int kind, bool begin)
    {
        const uint32_t base = static_cast<uint32_t>(presents % kFrames) * (2 * kTsKinds) + 2 * kind;
        if (begin)
        {
            vkCmdResetQueryPool(cmd, queries, base, 2);
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, base);
            tsFrame[presents % kFrames][kind] = UINT64_MAX;
        }
        else
        {
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, base + 1);
            tsFrame[presents % kFrames][kind] = presents;
        }
    }

    void log(const std::string& e, const std::string& f) { if (setup.log) setup.log(e, f); }

    template <class T> void load(T& f, const char* name)
    {
        f = reinterpret_cast<T>(setup.gdpa(device, name));
        if (!f) throw std::runtime_error(std::string("missing device function ") + name);
    }

    void loadFunctions()
    {
#define LOAD(name) load(name, #name)
        LOAD(vkCreateImage); LOAD(vkDestroyImage); LOAD(vkGetImageMemoryRequirements); LOAD(vkAllocateMemory);
        LOAD(vkFreeMemory); LOAD(vkBindImageMemory); LOAD(vkCreateImageView); LOAD(vkDestroyImageView); LOAD(vkCreateBuffer);
        LOAD(vkDestroyBuffer); LOAD(vkGetBufferMemoryRequirements); LOAD(vkBindBufferMemory); LOAD(vkMapMemory);
        LOAD(vkCreateShaderModule); LOAD(vkDestroyShaderModule); LOAD(vkCreateDescriptorSetLayout);
        LOAD(vkDestroyDescriptorSetLayout); LOAD(vkCreatePipelineLayout); LOAD(vkDestroyPipelineLayout);
        LOAD(vkCreateComputePipelines); LOAD(vkDestroyPipeline); LOAD(vkCreateDescriptorPool); LOAD(vkDestroyDescriptorPool);
        LOAD(vkResetDescriptorPool); LOAD(vkAllocateDescriptorSets); LOAD(vkUpdateDescriptorSets); LOAD(vkCmdBindPipeline);
        LOAD(vkCmdBindDescriptorSets); LOAD(vkCmdPushConstants); LOAD(vkCmdDispatch); LOAD(vkCmdPipelineBarrier);
        LOAD(vkCreateQueryPool); LOAD(vkDestroyQueryPool); LOAD(vkCmdResetQueryPool); LOAD(vkCmdWriteTimestamp);
        LOAD(vkGetQueryPoolResults);
#undef LOAD
        vkGetPhysicalDeviceMemoryProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
            setup.gipa(setup.instance, "vkGetPhysicalDeviceMemoryProperties"));
        vkGetPhysicalDeviceMemoryProperties(setup.physical, &memory);
        VkPhysicalDeviceProperties props{};
        reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(setup.gipa(setup.instance, "vkGetPhysicalDeviceProperties"))(setup.physical, &props);
        timestampPeriodNs = props.limits.timestampPeriod;

        // The FidelityFX backend's direct Vulkan calls (ffx_host_vk.h) go to the next layer too.
        FfxHostVkFunctions host;
        auto gi = [&](const char* n) { return setup.gipa(setup.instance, n); };
        host.GetPhysicalDeviceProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(gi("vkGetPhysicalDeviceProperties"));
        host.GetPhysicalDeviceProperties2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(gi("vkGetPhysicalDeviceProperties2"));
        host.GetPhysicalDeviceFeatures = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures>(gi("vkGetPhysicalDeviceFeatures"));
        host.GetPhysicalDeviceFeatures2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(gi("vkGetPhysicalDeviceFeatures2"));
        host.GetPhysicalDeviceMemoryProperties = vkGetPhysicalDeviceMemoryProperties;
        host.EnumerateDeviceExtensionProperties =
            reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(gi("vkEnumerateDeviceExtensionProperties"));
        host.GetDeviceProcAddr = setup.gdpa;
        host.CreateBuffer      = vkCreateBuffer;
        FfxHostVkEnabled enabled;
        enabled.deviceExtensions = setup.enabledExtensions;
        enabled.shaderFloat16    = setup.shaderFloat16;
        enabled.shaderInt16      = setup.shaderInt16;
        ffxHostVkSet(host, enabled);
    }

    uint32_t memoryType(uint32_t bits, VkMemoryPropertyFlags want)
    {
        for (uint32_t i = 0; i < memory.memoryTypeCount; ++i)
            if ((bits & (1u << i)) && (memory.memoryTypes[i].propertyFlags & want) == want)
                return i;
        throw std::runtime_error("no suitable memory type");
    }

    void check(VkResult r, const char* what)
    {
        if (r != VK_SUCCESS)
            throw std::runtime_error(std::string(what) + " failed: " + std::to_string(r));
    }

    Pass makePass(const uint32_t* code, size_t bytes, std::vector<VkDescriptorType> types, uint32_t pushBytes)
    {
        Pass p;
        std::vector<VkDescriptorSetLayoutBinding> b(types.size());
        for (uint32_t i = 0; i < types.size(); ++i)
            b[i] = {i, types[i], 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo sl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        sl.bindingCount = static_cast<uint32_t>(b.size());
        sl.pBindings    = b.data();
        check(vkCreateDescriptorSetLayout(device, &sl, nullptr, &p.setLayout), "vkCreateDescriptorSetLayout");
        VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, pushBytes};
        VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pl.setLayoutCount         = 1;
        pl.pSetLayouts            = &p.setLayout;
        pl.pushConstantRangeCount = 1;
        pl.pPushConstantRanges    = &pr;
        check(vkCreatePipelineLayout(device, &pl, nullptr, &p.layout), "vkCreatePipelineLayout");
        VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        sm.codeSize = bytes;
        sm.pCode    = code;
        VkShaderModule module;
        check(vkCreateShaderModule(device, &sm, nullptr, &module), "vkCreateShaderModule");
        VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cp.stage  = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, module, "main", nullptr};
        cp.layout = p.layout;
        VkResult r = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp, nullptr, &p.pipeline);
        vkDestroyShaderModule(device, module, nullptr);
        check(r, "vkCreateComputePipelines");
        return p;
    }

    void destroyPass(Pass& p)
    {
        if (p.pipeline) vkDestroyPipeline(device, p.pipeline, nullptr);
        if (p.layout) vkDestroyPipelineLayout(device, p.layout, nullptr);
        if (p.setLayout) vkDestroyDescriptorSetLayout(device, p.setLayout, nullptr);
        p = {};
    }

    void createStatic()
    {
        const auto SI = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, ST = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        const auto UB = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, SB = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        inputs2D    = makePass(kInputs2D, sizeof(kInputs2D), {SI, SI, UB, UB, ST, ST, SB, SB}, sizeof(PushInputs));
        inputsArray = makePass(kInputsArray, sizeof(kInputsArray), {SI, SI, UB, UB, ST, ST, SB, SB}, sizeof(PushInputs));
        copyIn      = makePass(kCopyIn, sizeof(kCopyIn), {SI, ST, UB, SB}, sizeof(PushCopy));
        copyOut     = makePass(kCopyOut, sizeof(kCopyOut), {SI, ST}, sizeof(PushCopy));

        VkDescriptorPoolSize sizes[] = {{SI, 4 * kSetsPerFrame}, {ST, 4 * kSetsPerFrame}, {UB, 4 * kSetsPerFrame}, {SB, 3 * kSetsPerFrame}};
        VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dp.maxSets       = kSetsPerFrame;
        dp.poolSizeCount = 4;
        dp.pPoolSizes    = sizes;
        for (auto& p : pools)
            check(vkCreateDescriptorPool(device, &dp, nullptr, &p), "vkCreateDescriptorPool");

        {
            VkBufferCreateInfo pb{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            pb.size  = 2 * 16 * sizeof(float);
            pb.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            check(vkCreateBuffer(device, &pb, nullptr, &projHistoryBuffer), "vkCreateBuffer");
            VkMemoryRequirements pr;
            vkGetBufferMemoryRequirements(device, projHistoryBuffer, &pr);
            VkMemoryAllocateInfo pa{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            pa.allocationSize  = pr.size;
            pa.memoryTypeIndex = memoryType(pr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            check(vkAllocateMemory(device, &pa, nullptr, &projHistoryMemory), "vkAllocateMemory");
            check(vkBindBufferMemory(device, projHistoryBuffer, projHistoryMemory, 0), "vkBindBufferMemory");
        }
        VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qi.queryType  = VK_QUERY_TYPE_TIMESTAMP;
        qi.queryCount = kFrames * 2 * kTsKinds;
        check(vkCreateQueryPool(device, &qi, nullptr, &queries), "vkCreateQueryPool");
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size  = kFrames * kTelemetryStride;
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        check(vkCreateBuffer(device, &bi, nullptr, &telemetryBuffer), "vkCreateBuffer");
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device, telemetryBuffer, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize  = req.size;
        ai.memoryTypeIndex = memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        check(vkAllocateMemory(device, &ai, nullptr, &telemetryMemory), "vkAllocateMemory");
        check(vkBindBufferMemory(device, telemetryBuffer, telemetryMemory, 0), "vkBindBufferMemory");
        void* mapped = nullptr;
        check(vkMapMemory(device, telemetryMemory, 0, VK_WHOLE_SIZE, 0, &mapped), "vkMapMemory");
        telemetryMapped = static_cast<uint8_t*>(mapped);
        std::memset(telemetryMapped, 0, kFrames * kTelemetryStride);
    }

    Image makeImage(VkFormat format, VkExtent2D extent, VkImageUsageFlags usage)
    {
        Image im;
        im.info.imageType   = VK_IMAGE_TYPE_2D;
        im.info.format      = format;
        im.info.extent      = {extent.width, extent.height, 1};
        im.info.mipLevels   = 1;
        im.info.arrayLayers = 1;
        im.info.samples     = VK_SAMPLE_COUNT_1_BIT;
        im.info.tiling      = VK_IMAGE_TILING_OPTIMAL;
        im.info.usage       = usage;
        check(vkCreateImage(device, &im.info, nullptr, &im.image), "vkCreateImage");
        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(device, im.image, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize  = req.size;
        ai.memoryTypeIndex = memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        check(vkAllocateMemory(device, &ai, nullptr, &im.memory), "vkAllocateMemory");  // dedicated: never aliased
        check(vkBindImageMemory(device, im.image, im.memory, 0), "vkBindImageMemory");
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image            = im.image;
        vi.viewType         = VK_IMAGE_VIEW_TYPE_2D;
        vi.format           = format;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        check(vkCreateImageView(device, &vi, nullptr, &im.view), "vkCreateImageView");
        return im;
    }

    void bury(Image& im)
    {
        if (!im) return;
        Image dead = im;
        im = Image{};
        graveyard.emplace_back(presents, [this, dead]() {
            vkDestroyImageView(device, dead.view, nullptr);
            vkDestroyImage(device, dead.image, nullptr);
            vkFreeMemory(device, dead.memory, nullptr);
        });
    }

    void destroyImageNow(Image& im)
    {
        if (!im) return;
        vkDestroyImageView(device, im.view, nullptr);
        vkDestroyImage(device, im.image, nullptr);
        vkFreeMemory(device, im.memory, nullptr);
        im = Image{};
    }

    void barrier(VkCommandBuffer cmd, Image& im, VkImageLayout to, VkAccessFlags src, VkAccessFlags dst)
    {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.srcAccessMask       = src;
        b.dstAccessMask       = dst;
        b.oldLayout           = im.layout;
        b.newLayout           = to;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = im.image;
        b.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr,
                             1, &b);
        im.layout = to;
    }

    // The descriptor pool and telemetry record of the current frame (reset on first use).
    uint32_t frameSlot()
    {
        const uint32_t slot = static_cast<uint32_t>(presents % kFrames);
        if (poolEpoch[slot] != presents)
        {
            vkResetDescriptorPool(device, pools[slot], 0);
            std::memset(telemetryMapped + slot * kTelemetryStride, 0, kTelemetryStride);
            poolEpoch[slot] = presents;
        }
        return slot;
    }

    VkDescriptorSet allocate(const Pass& p)
    {
        const uint32_t slot = frameSlot();
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool     = pools[slot];
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &p.setLayout;
        VkDescriptorSet set   = VK_NULL_HANDLE;
        if (vkAllocateDescriptorSets(device, &ai, &set) != VK_SUCCESS)
            return VK_NULL_HANDLE;
        return set;
    }

    struct Writer {
        std::vector<VkWriteDescriptorSet>   writes;
        std::vector<VkDescriptorImageInfo>  images;
        std::vector<VkDescriptorBufferInfo> buffers;
        VkDescriptorSet                     set;
        explicit Writer(VkDescriptorSet s) : set(s) { images.reserve(8); buffers.reserve(8); }
        void image(uint32_t binding, VkDescriptorType type, VkImageView view, VkImageLayout layout)
        {
            images.push_back({VK_NULL_HANDLE, view, layout});
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = set; w.dstBinding = binding; w.descriptorCount = 1; w.descriptorType = type; w.pImageInfo = &images.back();
            writes.push_back(w);
        }
        void buffer(uint32_t binding, VkDescriptorType type, VkDescriptorBufferInfo info)
        {
            buffers.push_back(info);
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = set; w.dstBinding = binding; w.descriptorCount = 1; w.descriptorType = type; w.pBufferInfo = &buffers.back();
            writes.push_back(w);
        }
    };

    void submit(Writer& w) { vkUpdateDescriptorSets(device, static_cast<uint32_t>(w.writes.size()), w.writes.data(), 0, nullptr); }

    VkDescriptorBufferInfo telemetrySlot() { return {telemetryBuffer, frameSlot() * VkDeviceSize(kTelemetryStride), sizeof(Telemetry)}; }

    void restore(VkCommandBuffer cmd, const ComputeState& s)
    {
        if (!s.pipeline) return;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipeline);
        for (const auto& set : s.sets)
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.layout, set.index, 1, &set.set,
                                    static_cast<uint32_t>(set.dynamicOffsets.size()), set.dynamicOffsets.data());
    }

    // ------------------------------------------------------------------------------------------
    uint64_t ssr(VkCommandBuffer cmd, uint64_t recording, const SsrBinding& b, const ComputeState& state)
    {
        ++ssrCount;
        if (b.layers != 1)
        {
            if (ssrCount <= 3) log("ssr_skipped", "\"reason\":\"layers\",\"layers\":" + std::to_string(b.layers));
            return 0;
        }
        if (fsr1Phase()) return 0;  // A/B: X-Plane's FSR 1 phase costs nothing extra
        if (!depth || depth.info.extent.width != b.extent.width || depth.info.extent.height != b.extent.height)
        {
            bury(depth);
            bury(motion);
            const VkImageUsageFlags usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            depth  = makeImage(VK_FORMAT_R32_SFLOAT, b.extent, usage);
            motion = makeImage(VK_FORMAT_R16G16_SFLOAT, b.extent, usage);
            log("inputs_images", "\"width\":" + std::to_string(b.extent.width) + ",\"height\":" + std::to_string(b.extent.height));
        }
        const Pass&     pass = b.array ? inputsArray : inputs2D;
        VkDescriptorSet set  = allocate(pass);
        if (!set)
        {
            log("ssr_skipped", "\"reason\":\"descriptor_pool_exhausted\"");
            return 0;
        }
        Writer w(set);
        w.image(0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, b.pos, b.posLayout);
        w.image(1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, b.vel, b.velLayout);
        w.buffer(2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, b.gbufferData);
        w.buffer(3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, b.ssrData);
        w.image(4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, depth.view, VK_IMAGE_LAYOUT_GENERAL);
        w.image(5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, motion.view, VK_IMAGE_LAYOUT_GENERAL);
        w.buffer(6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, telemetrySlot());
        w.buffer(7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, {projHistoryBuffer, 0, VK_WHOLE_SIZE});
        submit(w);

        stamp(cmd, kTsInputs, true);
        // The previous inputs pass wrote ProjHistory (read now) and read the slot written now.
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0,
                             nullptr);
        // Last frame's FSR read these images: execution dependency only (write-after-read).
        barrier(cmd, depth, VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT);
        barrier(cmd, motion, VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pass.layout, 0, 1, &set, 0, nullptr);
        PushInputs pc{static_cast<int32_t>(b.extent.width), static_cast<int32_t>(b.extent.height), 0, kNear,
                      static_cast<uint32_t>(presents), static_cast<uint32_t>((inputsSeq + 1) & 1), projHistoryValid ? 1u : 0u};
        projHistoryValid = true;
        vkCmdPushConstants(cmd, pass.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, (b.extent.width + 7) / 8, (b.extent.height + 7) / 8, 1);
        barrier(cmd, depth, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        barrier(cmd, motion, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        stamp(cmd, kTsInputs, false);
        restore(cmd, state);

        inputsRecording = recording;
        inputsExtent    = b.extent;
        ++inputsSeq;
        if (ssrCount <= 3)
            log("ssr_inputs_recorded", "\"recording\":" + std::to_string(recording) + ",\"seq\":" + std::to_string(inputsSeq) +
                                           ",\"width\":" + std::to_string(b.extent.width) + ",\"height\":" + std::to_string(b.extent.height) +
                                           ",\"array\":" + (b.array ? "true" : "false"));
        return inputsSeq;
    }

    void ensureFsr(VkExtent2D render, VkExtent2D display)
    {
        if (fsr && fsr->render.width == render.width && fsr->render.height == render.height && fsr->display.width == display.width &&
            fsr->display.height == display.height)
            return;
        if (fsr)
        {
            // The old context's resources may still be in flight: destroy it a ring later.
            std::shared_ptr<Fsr> old(fsr.release());
            for (Image* im : {&old->color, &old->output, &old->dilatedDepth, &old->dilatedMotion, &old->prevNearestDepth})
                bury(*im);
            graveyard.emplace_back(presents, [old]() { ffxFsr3UpscalerContextDestroy(old->context.get()); });
        }
        auto f    = std::make_unique<Fsr>();
        f->render = render;
        f->display = display;
        const size_t scratchSize = ffxGetScratchMemorySizeVK(setup.physical, FFX_FSR3UPSCALER_CONTEXT_COUNT);
        f->scratch.reset(static_cast<uint8_t*>(aligned_alloc(64, (scratchSize + 63) / 64 * 64)));
        std::memset(f->scratch.get(), 0, scratchSize);  // ffxGetInterfaceVK reads refCount before initialising
        VkDeviceContext dc{device, setup.physical, setup.gdpa};
        FfxFsr3UpscalerContextDescription cd{};
        FfxErrorCode e = ffxGetInterfaceVK(&cd.backendInterface, ffxGetDeviceVK(&dc), f->scratch.get(), scratchSize,
                                           FFX_FSR3UPSCALER_CONTEXT_COUNT);
        if (e != FFX_OK) throw std::runtime_error("ffxGetInterfaceVK failed: " + std::to_string(e));
        cd.flags          = FFX_FSR3UPSCALER_ENABLE_DEPTH_INVERTED | FFX_FSR3UPSCALER_ENABLE_DEPTH_INFINITE;
        cd.maxRenderSize  = {render.width, render.height};
        cd.maxUpscaleSize = {display.width, display.height};
        cd.fpMessage      = nullptr;
        f->context        = std::make_unique<FfxFsr3UpscalerContext>();
        e                 = ffxFsr3UpscalerContextCreate(f->context.get(), &cd);
        if (e != FFX_OK) throw std::runtime_error("ffxFsr3UpscalerContextCreate failed: " + std::to_string(e));
        // What the backend chose its shader permutations from (fp16: the 16-bit variants).
        FfxDeviceCapabilities caps{};
        cd.backendInterface.fpGetDeviceCapabilities(&cd.backendInterface, &caps);
        const VkImageUsageFlags rw = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        f->color            = makeImage(VK_FORMAT_R16G16B16A16_SFLOAT, render, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        f->output           = makeImage(VK_FORMAT_R16G16B16A16_SFLOAT, display, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        f->dilatedDepth     = makeImage(VK_FORMAT_R32_SFLOAT, render, rw);
        f->dilatedMotion    = makeImage(VK_FORMAT_R16G16_SFLOAT, render, rw);
        f->prevNearestDepth = makeImage(VK_FORMAT_R32_UINT, render, rw);
        fsr                 = std::move(f);
        phaseCount          = ffxFsr3UpscalerGetJitterPhaseCount(static_cast<int32_t>(render.width), static_cast<int32_t>(display.width));
        log("fsr_context", "\"render\":[" + std::to_string(render.width) + "," + std::to_string(render.height) + "],\"display\":[" +
                               std::to_string(display.width) + "," + std::to_string(display.height) + "],\"scratch_bytes\":" +
                               std::to_string(scratchSize) + ",\"fp16_shaders\":" + (caps.fp16Supported ? "true" : "false"));
    }

    // Why X-Plane's FSR 1 ran instead, in words a user can act on.
    static std::string plainReason(const std::string& r)
    {
        if (r == "layers") return "X-Plane is rendering for VR (two eyes). VR is not supported, so X-Plane's own FSR 1 is used.";
        if (r == "ab_fsr1_phase") return "A/B comparison (XP_FSR3_AB): this phase shows X-Plane's own FSR 1.";
        if (r == "observe_mode") return "observe mode (XP_FSR3=observe): FSR 3.1 runs, but the picture shown is X-Plane's FSR 1.";
        if (r == "msaa")
            return "X-Plane's MSAA anti-aliasing is on.\n"
                   "FSR 3.1 does its own anti-aliasing and needs MSAA off: turn MSAA off in X-Plane's graphics settings.";
        if (r == "no_new_ssr_inputs")
            return "X-Plane's screen-space reflection pass, which supplies the depth FSR 3.1 needs, did not run before the upscale.";
        if (r == "downscale") return "X-Plane's FSR setting is not upscaling.";
        if (r.empty()) return "unknown.";
        return "internal check '" + r + "' failed (details in the fsr3-*.jsonl log).";
    }

    void writeStatus(double fsrGpuMs, std::chrono::steady_clock::time_point now)
    {
        // At least 5 s of frames per status, so a count is never one frame's worth.
        const long seconds = long(std::chrono::duration_cast<std::chrono::seconds>(now - statusTime).count());
        if (!setup.status || seconds < 5) return;
        statusTime = now;
        const std::string window = " in the last " + std::to_string(seconds) + " s";
        const uint64_t easu = easuCount - statusEasu, done = replaced - statusReplaced;
        statusEasu     = easuCount;
        statusReplaced = replaced;
        const std::string sizes =
            fsr ? std::to_string(fsr->render.width) + "x" + std::to_string(fsr->render.height) + " -> " + std::to_string(fsr->display.width) +
                      "x" + std::to_string(fsr->display.height)
                : std::string("?");
        statusIdle = easu == 0;
        // GPU timestamps arrive a few frames late: say nothing rather than "0.00 ms" until they do.
        char gpuText[64] = "";
        if (fsrGpuMs > 0) std::snprintf(gpuText, sizeof gpuText, ", %.2f ms of GPU time per frame", fsrGpuMs);
        auto frames = [](uint64_t n) { return std::to_string(n) + (n == 1 ? " frame" : " frames"); };
        std::string text;
        if (easu == 0)
            text = "WAITING: X-Plane's FSR upscale is not running, so there is nothing to replace.\n"
                   "To use FSR 3.1, set Settings > Graphics > Rendering Resolution (FSR Supersampling) to anything but Off.\n";
        else if (done == easu)
            text = "ACTIVE: FSR 3.1 upscaled all " + frames(done) + window + ", " + sizes + gpuText + ".\n";
        else if (done > 0)
            text = "PARTLY ACTIVE: FSR 3.1 upscaled " + std::to_string(done) + " of " + frames(easu) + window + " (" +
                   sizes + ").\nThe others used X-Plane's FSR 1 because: " + plainReason(statusReason) + "\n";
        else
            text = "NOT USED: X-Plane's own FSR 1 upscaled all " + frames(easu) + window + " because: " +
                   plainReason(statusReason) + "\n";
        statusReason.clear();
        setup.status(text);
    }

    bool refuse(const std::string& reason, const std::string& extra = "")
    {
        if (reason != lastReason || easuCount <= 3 || easuCount % 600 == 0)
            log("easu_unchanged", "\"reason\":\"" + reason + "\",\"easu_dispatches\":" + std::to_string(easuCount) + extra);
        lastReason = statusReason = reason;
        return false;
    }

    static FfxResource resource(Image& im, const wchar_t* name, FfxResourceStates state, FfxResourceUsage usage)
    {
        return ffxGetResourceVK(reinterpret_cast<void*>(im.image), ffxGetImageResourceDescriptionVK(im.image, im.info, usage), name, state);
    }

    bool easu(VkCommandBuffer cmd, uint64_t recording, const EasuBinding& b, const ComputeState& state, uint64_t& readSeq)
    {
        // The first upscale after an idle stretch (startup, loading, FSR switched off) starts a new
        // status window, so the loading screen is not counted as time FSR 3.1 was running.
        if (statusIdle)
        {
            statusIdle     = false;
            statusTime     = std::chrono::steady_clock::now();
            statusEasu     = easuCount;
            statusReplaced = replaced;
        }
        ++easuCount;
        readSeq = 0;
        if (setup.abSeconds > 0 && abStart.time_since_epoch().count() == 0) abStart = std::chrono::steady_clock::now();
        if (b.layers != 1) return refuse("layers");
        if (fsr1Phase()) return refuse("ab_fsr1_phase");
        // Each SSR's inputs feed at most one FSR dispatch; whether they reach the GPU first is
        // checked at submission (onSubmitted).
        if (inputsSeq == 0 || inputsSeq == consumedSeq)
            return refuse(msaaPresent != UINT64_MAX && presents - msaaPresent <= 3 ? "msaa" : "no_new_ssr_inputs");
        if (b.inputExtent.width != inputsExtent.width || b.inputExtent.height != inputsExtent.height)
            return refuse("size_mismatch", ",\"input\":[" + std::to_string(b.inputExtent.width) + "," + std::to_string(b.inputExtent.height) +
                                               "],\"gbuffer\":[" + std::to_string(inputsExtent.width) + "," +
                                               std::to_string(inputsExtent.height) + "]");
        if (b.outputExtent.width < b.inputExtent.width || b.outputExtent.height < b.inputExtent.height) return refuse("downscale");

        ensureFsr(b.inputExtent, b.outputExtent);
        Fsr& f = *fsr;

        // 1. EASU's input -> layer colour image.
        VkDescriptorSet set = allocate(copyIn);
        if (!set) return refuse("descriptor_pool_exhausted");
        stamp(cmd, kTsFsr, true);
        struct StampEnd {
            Impl* self; VkCommandBuffer cmd;
            ~StampEnd() { self->stamp(cmd, kTsFsr, false); }
        } stampEnd{this, cmd};
        {
            Writer w(set);
            w.image(0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, b.input, b.inputLayout);
            w.image(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, f.color.view, VK_IMAGE_LAYOUT_GENERAL);
            w.buffer(2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, b.constants);
            w.buffer(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, telemetrySlot());
            submit(w);
        }
        barrier(cmd, f.color, VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, copyIn.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, copyIn.layout, 0, 1, &set, 0, nullptr);
        PushCopy pin{static_cast<int32_t>(b.inputExtent.width), static_cast<int32_t>(b.inputExtent.height), 0, 0};
        vkCmdPushConstants(cmd, copyIn.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pin), &pin);
        vkCmdDispatch(cmd, (b.inputExtent.width + 7) / 8, (b.inputExtent.height + 7) / 8, 1);
        barrier(cmd, f.color, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        // 2. FSR 3.1. Its own resources start UNDEFINED; the backend returns each to the declared state.
        for (Image* im : {&f.output, &f.dilatedDepth, &f.dilatedMotion, &f.prevNearestDepth})
            if (im->layout != VK_IMAGE_LAYOUT_GENERAL)
                barrier(cmd, *im, VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        const auto now   = std::chrono::steady_clock::now();
        const float dtMs = lastFsr.time_since_epoch().count() == 0
                               ? 16.7f
                               : std::clamp(std::chrono::duration<float, std::milli>(now - lastFsr).count(), 1.0f, 100.0f);
        lastFsr = now;
        FfxFsr3UpscalerDispatchDescription dd{};
        dd.commandList                  = ffxGetCommandListVK(cmd);
        dd.color                        = resource(f.color, L"xp_color", FFX_RESOURCE_STATE_COMPUTE_READ, FFX_RESOURCE_USAGE_READ_ONLY);
        dd.depth                        = resource(depth, L"xp_depth", FFX_RESOURCE_STATE_COMPUTE_READ, FFX_RESOURCE_USAGE_READ_ONLY);
        dd.motionVectors                = resource(motion, L"xp_motion", FFX_RESOURCE_STATE_COMPUTE_READ, FFX_RESOURCE_USAGE_READ_ONLY);
        dd.output                       = resource(f.output, L"xp_output", FFX_RESOURCE_STATE_UNORDERED_ACCESS, FFX_RESOURCE_USAGE_UAV);
        dd.dilatedDepth                 = resource(f.dilatedDepth, L"xp_dilated_depth", FFX_RESOURCE_STATE_UNORDERED_ACCESS, FFX_RESOURCE_USAGE_UAV);
        dd.dilatedMotionVectors         = resource(f.dilatedMotion, L"xp_dilated_motion", FFX_RESOURCE_STATE_UNORDERED_ACCESS, FFX_RESOURCE_USAGE_UAV);
        dd.reconstructedPrevNearestDepth = resource(f.prevNearestDepth, L"xp_prev_depth", FFX_RESOURCE_STATE_UNORDERED_ACCESS, FFX_RESOURCE_USAGE_UAV);
        // The viewport offset this frame's main-scene passes were recorded with (layer.cpp), if any.
        float jx = 0.0f, jy = 0.0f;
        const bool jittered = jitterAppliedFrame == presents && currentJitter(jx, jy);
        dd.jitterOffset                 = {jittered ? jx : 0.0f, jittered ? jy : 0.0f};
        jitteredFrames += jittered ? 1 : 0;
        dd.motionVectorScale            = {static_cast<float>(b.inputExtent.width), static_cast<float>(b.inputExtent.height)};
        dd.renderSize                   = {b.inputExtent.width, b.inputExtent.height};
        dd.upscaleSize                  = {b.outputExtent.width, b.outputExtent.height};
        dd.enableSharpening             = false;  // X-Plane's own RCAS pass follows
        dd.frameTimeDelta               = dtMs;
        dd.preExposure                  = 1.0f;
        dd.reset                        = f.reset;
        dd.cameraNear                   = kNear;
        dd.cameraFar                    = 1.0e6f;
        dd.cameraFovAngleVertical       = fovY > 0.0f ? fovY : 1.0f;
        dd.viewSpaceToMetersFactor      = 1.0f;
        const FfxErrorCode e            = ffxFsr3UpscalerContextDispatch(f.context.get(), &dd);
        f.reset                         = false;
        if (e != FFX_OK)
        {
            restore(cmd, state);
            return refuse("ffx_dispatch_error", ",\"code\":" + std::to_string(e));
        }
        consumedSeq = readSeq = inputsSeq;
        ++fsrRuns;
        sameRecordingReads += recording == inputsRecording;

        if (setup.mode != Mode::Replace)
        {
            restore(cmd, state);
            return refuse("observe_mode");  // X-Plane's EASU runs unchanged; FSR wrote only layer images
        }

        // 3. FSR output -> EASU's output image, as EASU would write it.
        set = allocate(copyOut);
        if (!set)
        {
            restore(cmd, state);
            return refuse("descriptor_pool_exhausted");
        }
        {
            Writer w(set);
            w.image(0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, f.output.view, VK_IMAGE_LAYOUT_GENERAL);
            w.image(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, b.output, b.outputLayout);
            submit(w);
        }
        barrier(cmd, f.output, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, copyOut.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, copyOut.layout, 0, 1, &set, 0, nullptr);
        PushCopy pout{static_cast<int32_t>(b.outputExtent.width), static_cast<int32_t>(b.outputExtent.height), 0, 0};
        vkCmdPushConstants(cmd, copyOut.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pout), &pout);
        vkCmdDispatch(cmd, (b.outputExtent.width + 7) / 8, (b.outputExtent.height + 7) / 8, 1);
        restore(cmd, state);
        if (++replaced <= 3 || replaced % 600 == 0)
            log("easu_replaced", "\"count\":" + std::to_string(replaced) + ",\"frame_ms\":" + num(dtMs) + ",\"fov_y\":" + num(fovY) +
                                     ",\"jitter\":[" + num(dd.jitterOffset.x) + "," + num(dd.jitterOffset.y) + "],\"jittered_frames\":" +
                                     std::to_string(jitteredFrames));
        lastReason.clear();
        return true;
    }

    void submitted(VkQueue queue, uint64_t call, uint64_t recording, const std::vector<InputsEvent>& events)
    {
        for (const InputsEvent& e : events)
        {
            if (e.write)
            {
                submittedSeq       = e.seq;
                submittedQueue     = queue;
                submittedCall      = call;
                submittedRecording = recording;
                continue;
            }
            ++readsChecked;
            const char* order = "in_order";
            if (e.seq > submittedSeq) order = "missing", ++readsMissing;            // writer not submitted yet
            else if (e.seq < submittedSeq) order = "overwritten", ++readsOverwritten;  // a newer SSR went first
            else if (queue != submittedQueue) order = "cross_queue", ++readsCrossQueue;
            else ++readsInOrder;
            const uint64_t bad = readsMissing + readsOverwritten + readsCrossQueue;
            if (readsChecked <= 3 || (std::strcmp(order, "in_order") && bad <= 5))
                log("inputs_order", "\"order\":\"" + std::string(order) + "\",\"read_seq\":" + std::to_string(e.seq) +
                                        ",\"submitted_seq\":" + std::to_string(submittedSeq) + ",\"same_recording\":" +
                                        (submittedRecording == recording ? "true" : "false") + ",\"same_submit_call\":" +
                                        (submittedCall == call ? "true" : "false"));
        }
    }

    bool currentJitter(float& x, float& y)
    {
        if (setup.mode != Mode::Replace || phaseCount <= 0 || fsr1Phase()) return false;
        return ffxFsr3UpscalerGetJitterOffset(&x, &y, static_cast<int32_t>(presents % uint64_t(phaseCount)), phaseCount) == FFX_OK;
    }

    void readTimestamps(uint64_t frame)
    {
        const uint32_t slot = static_cast<uint32_t>(frame % kFrames);
        for (int k = 0; k < kTsKinds; ++k)
        {
            if (tsFrame[slot][k] != frame) continue;
            uint64_t v[2];
            if (vkGetQueryPoolResults(device, queries, slot * 2 * kTsKinds + 2 * k, 2, sizeof v, v, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) !=
                VK_SUCCESS)
                continue;
            const double ms = double(v[1] - v[0]) * timestampPeriodNs / 1.0e6;
            if (ms < 0 || ms > 1000) continue;
            gpuMsSum[k] += ms, ++gpuCount[k];
            abGpuMsSum[k] += ms, ++abGpuCount[k];
        }
    }

    void abTick(std::chrono::steady_clock::time_point now)
    {
        if (setup.abSeconds <= 0 || abStart.time_since_epoch().count() == 0) return;
        const double inPhase = std::chrono::duration<double>(now - abStart).count();
        if (inPhase >= 2.0 && lastPresentTime.time_since_epoch().count() != 0)
        {
            abFrameMsSum += std::chrono::duration<double, std::milli>(now - lastPresentTime).count();
            ++abFrames;
        }
        else if (inPhase < 2.0)
        {
            std::fill(std::begin(abGpuMsSum), std::end(abGpuMsSum), 0.0);
            std::fill(std::begin(abGpuCount), std::end(abGpuCount), 0);
        }
        if (inPhase < setup.abSeconds) return;
        auto mean = [&](int k) { return abGpuCount[k] ? abGpuMsSum[k] / abGpuCount[k] : 0.0; };
        const double frameMs = abFrames ? abFrameMsSum / abFrames : 0.0;
        log("ab_phase", std::string("\"phase\":\"") + (abPhase == 0 ? "fsr3" : "fsr1") + "\",\"frames\":" + std::to_string(abFrames) +
                            ",\"mean_frame_ms\":" + num(frameMs) + ",\"fps\":" + num(frameMs > 0 ? 1000.0 / frameMs : 0.0) +
                            ",\"gpu_inputs_ms\":" + num(mean(kTsInputs)) + ",\"gpu_fsr_ms\":" + num(mean(kTsFsr)) +
                            ",\"gpu_easu_ms\":" + num(mean(kTsEasu)));
        abPhase      = 1 - abPhase;
        abStart      = now;
        abFrameMsSum = 0;
        abFrames     = 0;
        std::fill(std::begin(abGpuMsSum), std::end(abGpuMsSum), 0.0);
        std::fill(std::begin(abGpuCount), std::end(abGpuCount), 0);
        if (abPhase == 0 && fsr) fsr->reset = true;  // history from before the FSR 1 phase is stale
    }

    void present()
    {
        const auto now = std::chrono::steady_clock::now();
        abTick(now);
        lastPresentTime = now;
        ++presents;
        if (presents > kTelemetryLatency) readTimestamps(presents - kTelemetryLatency);
        // Telemetry recorded kTelemetryLatency frames ago: long finished on the GPU.
        if (presents > kTelemetryLatency)
        {
            const uint64_t frame = presents - kTelemetryLatency;
            const uint32_t slot  = static_cast<uint32_t>(frame % kFrames);
            if (poolEpoch[slot] == frame)
            {
                Telemetry t;
                std::memcpy(&t, telemetryMapped + slot * kTelemetryStride, sizeof(t));
                if (t.tag == frame && std::isfinite(t.proj[5]) && std::fabs(t.proj[5]) > 1e-3f)
                {
                    const float fov = 2.0f * std::atan(1.0f / std::fabs(t.proj[5]));
                    if (fov > 0.05f && fov < 3.0f)
                        fovY = fov;
                    if (++telemetryValid <= 5 || telemetryValid % 600 == 0)
                    {
                        std::string easuConst = "[";
                        for (int i = 0; i < 20; ++i)
                            easuConst += (i ? "," : "") + std::to_string(t.easuConst[i]);
                        log("telemetry",
                            "\"frame\":" + std::to_string(frame) + ",\"fov_y\":" + num(fovY) + ",\"sky_pixels\":" +
                                std::to_string(t.skyPixels) + ",\"local_pixels\":" + std::to_string(t.localPixels) + ",\"proj\":" + mat(t.proj) +
                                ",\"reproj\":" + mat(t.reproj) + ",\"local_reproj\":" + mat(t.localReproj) + ",\"clip_info\":[" +
                                num(t.clipInfo[0]) + "," + num(t.clipInfo[1]) + "," + num(t.clipInfo[2]) + "," + num(t.clipInfo[3]) +
                                "],\"viewport\":[" + num(t.viewport[0]) + "," + num(t.viewport[1]) + "," + num(t.viewport[2]) + "," +
                                num(t.viewport[3]) + "],\"gbuffer_size\":[" + num(t.gbufferSize[0]) + "," + num(t.gbufferSize[1]) +
                                "],\"centre\":[" + num(t.centre[0]) + "," + num(t.centre[1]) + "," + num(t.centre[2]) + "," +
                                num(t.centre[3]) + "],\"easu_const\":" + easuConst + "]");
                    }
                }
            }
        }
        // Every 10 s while the pipeline is active: what ran, what was refused, the submit order, and
        // the GPU time of the layer's passes and of X-Plane's own EASU when it ran.
        if (ssrCount + easuCount > 0 && now - lastSummary >= std::chrono::seconds(10))
        {
            lastSummary = now;
            auto gpu = [&](int k) { return gpuCount[k] ? gpuMsSum[k] / gpuCount[k] : 0.0; };
            const std::string gpuFields = ",\"gpu_inputs_ms\":" + num(gpu(kTsInputs)) + ",\"gpu_fsr_ms\":" + num(gpu(kTsFsr)) +
                                          ",\"gpu_easu_ms\":" + num(gpu(kTsEasu));
            const double fsrGpuMs = gpu(kTsInputs) + gpu(kTsFsr);
            std::fill(std::begin(gpuMsSum), std::end(gpuMsSum), 0.0);
            std::fill(std::begin(gpuCount), std::end(gpuCount), 0);
            log("fsr_summary", "\"presents\":" + std::to_string(presents) + ",\"ssr\":" + std::to_string(ssrCount) + ",\"easu\":" +
                                   std::to_string(easuCount) + ",\"fsr_runs\":" + std::to_string(fsrRuns) + ",\"replaced\":" +
                                   std::to_string(replaced) + ",\"same_recording_reads\":" + std::to_string(sameRecordingReads) +
                                   ",\"reads_checked\":" + std::to_string(readsChecked) + ",\"in_order\":" + std::to_string(readsInOrder) +
                                   ",\"missing\":" + std::to_string(readsMissing) + ",\"overwritten\":" + std::to_string(readsOverwritten) +
                                   ",\"cross_queue\":" + std::to_string(readsCrossQueue) + ",\"last_refusal\":\"" + lastReason + "\"" + gpuFields);
            writeStatus(fsrGpuMs, now);
        }
        for (auto it = graveyard.begin(); it != graveyard.end();)
        {
            if (it->first + kFrames <= presents)
            {
                it->second();
                it = graveyard.erase(it);
            }
            else
                ++it;
        }
    }

    void destroyAll()
    {
        // vkDestroyDevice follows: the application has made the device idle.
        for (auto& g : graveyard) g.second();
        graveyard.clear();
        if (fsr)
        {
            ffxFsr3UpscalerContextDestroy(fsr->context.get());
            for (Image* im : {&fsr->color, &fsr->output, &fsr->dilatedDepth, &fsr->dilatedMotion, &fsr->prevNearestDepth})
                destroyImageNow(*im);
            fsr.reset();
        }
        destroyImageNow(depth);
        destroyImageNow(motion);
        for (auto& p : pools) if (p) vkDestroyDescriptorPool(device, p, nullptr);
        if (queries) vkDestroyQueryPool(device, queries, nullptr);
        if (projHistoryBuffer) vkDestroyBuffer(device, projHistoryBuffer, nullptr);
        if (projHistoryMemory) vkFreeMemory(device, projHistoryMemory, nullptr);
        if (telemetryBuffer) vkDestroyBuffer(device, telemetryBuffer, nullptr);
        if (telemetryMemory) vkFreeMemory(device, telemetryMemory, nullptr);
        for (Pass* p : {&inputs2D, &inputsArray, &copyIn, &copyOut}) destroyPass(*p);
    }
};

Engine::Engine(const DeviceSetup& setup) : setup_(setup), impl_(std::make_unique<Impl>(setup_))
{
    impl_->loadFunctions();
    impl_->createStatic();
    setup_.log("engine", std::string("\"mode\":\"") + (setup_.mode == Mode::Replace ? "replace" : "observe") + "\"");
}

Engine::~Engine()
{
    std::lock_guard<std::mutex> l(lock_);
    impl_->destroyAll();
}

uint64_t Engine::onSsr(VkCommandBuffer cmd, uint64_t recording, const SsrBinding& b, const ComputeState& restore)
{
    std::lock_guard<std::mutex> l(lock_);
    return impl_->ssr(cmd, recording, b, restore);
}

bool Engine::onEasu(VkCommandBuffer cmd, uint64_t recording, const EasuBinding& b, const ComputeState& restore, uint64_t& readSeq)
{
    std::lock_guard<std::mutex> l(lock_);
    return impl_->easu(cmd, recording, b, restore, readSeq);
}

void Engine::onSubmitted(VkQueue queue, uint64_t submitCall, uint64_t recording, const std::vector<InputsEvent>& events)
{
    std::lock_guard<std::mutex> l(lock_);
    impl_->submitted(queue, submitCall, recording, events);
}

bool Engine::jitter(float& x, float& y)
{
    std::lock_guard<std::mutex> l(lock_);
    return impl_->currentJitter(x, y);
}

void Engine::noteJitterApplied()
{
    std::lock_guard<std::mutex> l(lock_);
    impl_->jitterAppliedFrame = impl_->presents;
}

void Engine::markEasu(VkCommandBuffer cmd, bool begin)
{
    std::lock_guard<std::mutex> l(lock_);
    impl_->stamp(cmd, Impl::kTsEasu, begin);
}

void Engine::onMsaaScene()
{
    std::lock_guard<std::mutex> l(lock_);
    if (impl_->msaaPresent == UINT64_MAX) impl_->log("msaa_scene", "\"presents\":" + std::to_string(impl_->presents));
    impl_->msaaPresent = impl_->presents;
}

void Engine::onPresent()
{
    std::lock_guard<std::mutex> l(lock_);
    impl_->present();
}

}  // namespace xpfsr
