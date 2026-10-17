// VK_LAYER_XPLANE_fsr3: replaces X-Plane 12's FSR 1 EASU pass with AMD FSR 3.1 temporal upscaling.
//
// This file tracks descriptors, pipelines and command buffers and finds X-Plane's SSR and EASU
// dispatches; the FSR 3.1 work is in engine.cpp. Mode comes from XP_FSR3: "replace", "observe" (run everything into layer-owned
// images but leave X-Plane's picture unchanged), "timing" (intercept only vkQueuePresentKHR and log
// frame times: the baseline against which the layer's own cost is measured), anything else passes
// through. A DISABLED file in XP_FSR3_OUTPUT also passes through. Only X-Plane-x86_64 and the test
// harness (fake_xplane) are instrumented.
#include "engine.h"
#include "shader_classify.h"

#include <vulkan/vk_layer.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <unistd.h>
#include <x86intrin.h>
#include <atomic>

#define EXPORT extern "C" __attribute__((visibility("default"))) VKAPI_ATTR

namespace {
std::recursive_mutex guard;

#define LAYER_NEXT_FUNCTIONS(X) \
    X(vkDestroyDevice) \
    X(vkCreateImage) \
    X(vkDestroyImage) \
    X(vkCreateImageView) \
    X(vkDestroyImageView) \
    X(vkCreateBuffer) \
    X(vkDestroyBuffer) \
    X(vkCreateShaderModule) \
    X(vkDestroyShaderModule) \
    X(vkCreateComputePipelines) \
    X(vkDestroyPipeline) \
    X(vkCreatePipelineLayout) \
    X(vkDestroyPipelineLayout) \
    X(vkCreateDescriptorSetLayout) \
    X(vkDestroyDescriptorSetLayout) \
    X(vkAllocateDescriptorSets) \
    X(vkFreeDescriptorSets) \
    X(vkResetDescriptorPool) \
    X(vkDestroyDescriptorPool) \
    X(vkUpdateDescriptorSets) \
    X(vkCreateDescriptorUpdateTemplate) \
    X(vkCreateDescriptorUpdateTemplateKHR) \
    X(vkDestroyDescriptorUpdateTemplate) \
    X(vkDestroyDescriptorUpdateTemplateKHR) \
    X(vkUpdateDescriptorSetWithTemplate) \
    X(vkUpdateDescriptorSetWithTemplateKHR) \
    X(vkAllocateCommandBuffers) \
    X(vkFreeCommandBuffers) \
    X(vkDestroyCommandPool) \
    X(vkResetCommandPool) \
    X(vkBeginCommandBuffer) \
    X(vkResetCommandBuffer) \
    X(vkCmdBindPipeline) \
    X(vkCmdBindDescriptorSets) \
    X(vkCmdDispatch) \
    X(vkQueueSubmit) \
    X(vkQueueSubmit2) \
    X(vkQueueSubmit2KHR) \
    X(vkQueuePresentKHR) \
    X(vkCreateRenderPass) \
    X(vkCreateRenderPass2) \
    X(vkCreateRenderPass2KHR) \
    X(vkDestroyRenderPass) \
    X(vkCreateFramebuffer) \
    X(vkDestroyFramebuffer) \
    X(vkCreateGraphicsPipelines) \
    X(vkCmdBeginRenderPass) \
    X(vkCmdBeginRenderPass2) \
    X(vkCmdBeginRenderPass2KHR) \
    X(vkCmdEndRenderPass) \
    X(vkCmdEndRenderPass2) \
    X(vkCmdEndRenderPass2KHR) \
    X(vkCmdBeginRendering) \
    X(vkCmdBeginRenderingKHR) \
    X(vkCmdEndRendering) \
    X(vkCmdEndRenderingKHR) \
    X(vkCmdSetViewport) \
    X(vkCmdSetViewportWithCount) \
    X(vkCmdSetViewportWithCountEXT) \
    X(vkCmdBindDescriptorSets2) \
    X(vkCmdBindDescriptorSets2KHR) \
    X(vkCmdPushDescriptorSet) \
    X(vkCmdPushDescriptorSetKHR) \
    X(vkCmdPushDescriptorSetWithTemplate) \
    X(vkCmdPushDescriptorSetWithTemplateKHR) \
    X(vkCmdBindDescriptorBuffersEXT) \
    X(vkCmdSetDescriptorBufferOffsetsEXT) \
    X(vkCmdBindShadersEXT)

// The next layer's functions, resolved once per device (no per-call lookup).
enum NextId : int {
#define X(n) F_##n,
    LAYER_NEXT_FUNCTIONS(X)
#undef X
    F_COUNT
};
const char* const kNextNames[F_COUNT] = {
#define X(n) #n,
    LAYER_NEXT_FUNCTIONS(X)
#undef X
};

// CPU profile (XP_FSR3_PROFILE=1): per hook, calls and the layer's own time, with the time spent
// in the next layer and driver excluded; plus waits for `guard`. Thread-local, summed every 10 s.
bool profiling()
{
    static const bool p = [] { const char* v = std::getenv("XP_FSR3_PROFILE"); return v && !std::strcmp(v, "1"); }();
    return p;
}
// Written only by the owning thread, read by the presenting thread: relaxed atomics, bumped with a
// plain load and store (no locked instruction).
struct ProfBlock {
    std::atomic<uint64_t> calls[F_COUNT] = {}, self[F_COUNT] = {};
    std::atomic<uint64_t> lockWaits{0}, lockWaitCycles{0};
};
inline void bump(std::atomic<uint64_t>& a, uint64_t n) { a.store(a.load(std::memory_order_relaxed) + n, std::memory_order_relaxed); }
std::mutex              profRegistryLock;
std::vector<ProfBlock*> profRegistry;  // blocks of threads that ever ran a hook (never freed)
__attribute__((tls_model("initial-exec"))) thread_local ProfBlock* t_prof       = nullptr;
__attribute__((tls_model("initial-exec"))) thread_local uint64_t   t_downstream = 0;
inline ProfBlock& prof()
{
    if (!t_prof)
    {
        t_prof = new ProfBlock();
        std::lock_guard<std::mutex> l(profRegistryLock);
        profRegistry.push_back(t_prof);
    }
    return *t_prof;
}
struct HookScope {
    int      id;
    uint64_t t0 = 0, d0 = 0;
    explicit HookScope(int i) : id(i)
    {
        if (profiling()) { t0 = __rdtsc(); d0 = t_downstream; }
    }
    ~HookScope()
    {
        if (!t0) return;
        auto& p = prof();
        bump(p.calls[id], 1);
        bump(p.self[id], __rdtsc() - t0 - (t_downstream - d0));
    }
};
#define HOOK_SCOPE(name) HookScope hookScope_(F_##name)
template <class PFN> struct NextCall {
    PFN f;
    template <class... A> auto operator()(A... a) const
    {
        struct Timer {
            uint64_t s = profiling() ? __rdtsc() : 0;
            ~Timer() { if (s) t_downstream += __rdtsc() - s; }
        } timer;
        return f(a...);
    }
};
#define NEXT(dev, name) (NextCall<PFN_##name>{reinterpret_cast<PFN_##name>((dev)->next[F_##name])})

// A lock that records how often and how long a thread waited for it (cpu_profile).
template <class M> struct TimedUnique {
    M& m;
    explicit TimedUnique(M& mutex) : m(mutex)
    {
        if (m.try_lock()) return;
        const uint64_t t = profiling() ? __rdtsc() : 0;
        m.lock();
        if (t) { auto& p = prof(); bump(p.lockWaits, 1); bump(p.lockWaitCycles, __rdtsc() - t); }
    }
    ~TimedUnique() { m.unlock(); }
    TimedUnique(const TimedUnique&) = delete;
};

// `guard`, recording how often and how long a thread waited for it.
struct GuardLock {
    GuardLock()
    {
        if (guard.try_lock()) return;
        const uint64_t t = profiling() ? __rdtsc() : 0;
        guard.lock();
        if (t) { auto& p = prof(); bump(p.lockWaits, 1); bump(p.lockWaitCycles, __rdtsc() - t); }
    }
    ~GuardLock() { guard.unlock(); }
    GuardLock(const GuardLock&) = delete;
};

std::string outputDir()
{
    const char* d = std::getenv("XP_FSR3_OUTPUT");
    return d && *d ? d : "/tmp/xplane-fsr3";
}

bool targeted()
{
    static const bool active = [] {
        std::error_code ec;
        if (std::filesystem::exists(std::filesystem::path(outputDir()) / "DISABLED", ec)) return false;
        auto name = std::filesystem::read_symlink("/proc/self/exe", ec).filename().string();
        return !ec && (name == "X-Plane-x86_64" || name == "fake_xplane");
    }();
    return active;
}

xpfsr::Mode mode()
{
    static const xpfsr::Mode m = [] {
        const char* v = std::getenv("XP_FSR3");
        if (!targeted() || !v) return xpfsr::Mode::Passthrough;
        if (!std::strcmp(v, "replace")) return xpfsr::Mode::Replace;
        if (!std::strcmp(v, "observe")) return xpfsr::Mode::Observe;
        return xpfsr::Mode::Passthrough;
    }();
    return m;
}

// XP_FSR3=timing: inert except for frame timing (mode() stays Passthrough; no engine, no tracking).
bool timingOnly()
{
    static const bool t = [] {
        const char* v = std::getenv("XP_FSR3");
        return targeted() && v && !std::strcmp(v, "timing");
    }();
    return t;
}

void event(const std::string& type, const std::string& fields)
{
    static std::mutex      m;
    static std::ofstream   file;
    std::lock_guard<std::mutex> l(m);
    if (!file.is_open())
    {
        std::error_code ec;
        std::filesystem::create_directories(outputDir(), ec);
        file.open(std::filesystem::path(outputDir()) / ("fsr3-" + std::to_string(getpid()) + ".jsonl"), std::ios::app);
    }
    file << "{\"event\":\"" << type << "\"" << (fields.empty() ? "" : ",") << fields << "}\n";
    file.flush();
}

template <class T> uint64_t id(T h) { return (uint64_t)(uintptr_t)h; }
template <class T> void* key(T h) { return h ? *reinterpret_cast<void**>(h) : nullptr; }


struct Instance {
    VkInstance                     handle;
    PFN_vkGetInstanceProcAddr      gipa;
    PFN_GetPhysicalDeviceProcAddr  gpdpa;
};
struct Image { VkExtent3D extent; uint32_t layers; VkImageUsageFlags usage; VkFormat format; };
struct View { VkImage image; VkImageViewType type; VkImageSubresourceRange range; };
struct Pipeline { ShaderRole role = ShaderRole::None; VkPipelineLayout layout{}; };
struct Slot { VkDescriptorType type; VkDescriptorImageInfo image{}; VkDescriptorBufferInfo buffer{}; };
using Slots = std::map<std::pair<uint32_t, uint32_t>, Slot>;
// track: the layout has a storage image, as X-Plane's SSR (i_output) and EASU (i_output_texture)
// layouts do; only such sets get their descriptor writes recorded. dynamicCount: its dynamic
// uniform/storage buffer descriptors, the ones that take dynamic offsets when bound.
struct DescriptorLayout {
    std::map<uint32_t, VkDescriptorSetLayoutBinding> bindings;
    bool     ordinary = false, track = false;
    uint32_t dynamicCount = 0;
};
using LayoutPtr = std::shared_ptr<const DescriptorLayout>;
// setLayouts are captured at creation: a set layout may be destroyed once the pipeline layout exists.
struct PipelineLayoutInfo { uint32_t sets = 0, pushRanges = 0; std::vector<LayoutPtr> setLayouts; };
struct Set { VkDescriptorPool pool; LayoutPtr layout; Slots slots; bool complete = true; };
struct UpdateTemplate { std::vector<VkDescriptorUpdateTemplateEntry> entries; VkDescriptorUpdateTemplateType type; };
// One set number's compute vkCmdBindDescriptorSets arguments, as passed. Decoded only at an SSR or
// EASU dispatch (boundIssue), so a bind costs no table lookup beyond finding the command buffer.
constexpr uint32_t kMaxBoundSets = 8, kMaxBindOffsets = 16;
struct BoundSet {
    VkDescriptorSet  set    = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    uint32_t         first = 0, count = 0;  // the call's firstSet and descriptorSetCount
    uint32_t         offsetCount = 0;       // the call's dynamic offsets, all its sets (kept up to kMaxBindOffsets)
    std::array<uint32_t, kMaxBindOffsets> offsets{};
};
struct Command {
    bool                     known = true;     // false after a binding command this layer does not decode
    bool                     oneTime = false;  // ONE_TIME_SUBMIT and not SIMULTANEOUS_USE
    VkPipeline               pipeline{};
    ShaderRole               role = ShaderRole::None;  // of the bound compute pipeline
    std::array<BoundSet, kMaxBoundSets> bound{};       // compute set numbers 0..7
    uint64_t                 recording = 0;
    std::vector<xpfsr::InputsEvent> inputs;    // layer input images written/read, in recording order
    // Viewport jitter (main-scene render passes only).
    bool                     mainPass = false;  // inside a render pass on the main scene depth
    bool                     jittered = false;
    float                    jx = 0, jy = 0;
    std::vector<VkViewport>  viewports;  // as the application last set them (unjittered)
    uint32_t                 firstViewport = 0;
    bool                     viewportKnown = false, viewportWithCount = false;
    std::string              census;     // main_pass_census fields for the pass being recorded
    uint32_t                 passBinds = 0;
};
struct RenderPassInfo { std::vector<uint32_t> colors; std::vector<uint32_t> depths; };

// A hash table in 64 independently locked shards, so that X-Plane's worker threads rarely meet in
// it. A shard lock is a leaf: nothing else is locked while one is held.
template <class V> class Sharded {
public:
    // f(V&) under the shard lock; false if absent.
    template <class F> bool with(uint64_t k, F&& f)
    {
        Shard&  s = shard(k);
        Locked  l(s.m);
        auto    it = s.map.find(k);
        if (it == s.map.end()) return false;
        f(it->second);
        return true;
    }
    bool get(uint64_t k, V& out) { return with(k, [&](V& v) { out = v; }); }
    bool contains(uint64_t k) { return with(k, [](V&) {}); }
    // f(V&) on the entry, value-initialised first if absent.
    template <class F> void upsert(uint64_t k, F&& f)
    {
        Shard& s = shard(k);
        Locked l(s.m);
        f(s.map[k]);
    }
    void put(uint64_t k, V v) { upsert(k, [&](V& x) { x = std::move(v); }); }
    bool erase(uint64_t k)
    {
        Shard& s = shard(k);
        Locked l(s.m);
        return s.map.erase(k) != 0;
    }
    // Removes the entry, moving its value out.
    bool take(uint64_t k, V& out)
    {
        Shard& s = shard(k);
        Locked l(s.m);
        auto   it = s.map.find(k);
        if (it == s.map.end()) return false;
        out = std::move(it->second);
        s.map.erase(it);
        return true;
    }
    // Removes the entry if pred(V&).
    template <class F> void eraseIf(uint64_t k, F&& pred)
    {
        Shard& s = shard(k);
        Locked l(s.m);
        auto   it = s.map.find(k);
        if (it != s.map.end() && pred(it->second)) s.map.erase(it);
    }

private:
    using Locked = TimedUnique<std::mutex>;
    struct alignas(64) Shard {
        std::mutex                     m;
        std::unordered_map<uint64_t, V> map;
    };
    Shard& shard(uint64_t k) { return shards_[(k * UINT64_C(0x9E3779B97F4A7C15)) >> 58]; }
    Shard  shards_[64];
};

// Templates and set layouts are few and read by every worker thread, so their shards and reference
// counts would be hot (fake_xplane --load-threads 24: ~3 us per template update). Each thread keeps
// what it looked up; the cache is dropped when the device destroys any object of that kind (`epoch`,
// bumped before the destroy, so a reused handle is never served stale). Hits return a pointer into
// the thread's own map: no shared write.
template <class V> struct ThreadCache {
    uint64_t                        owner = 0;  // Device::serial
    uint64_t                        epoch = 0;
    std::unordered_map<uint64_t, V> map;
};
template <class V>
const V* cachedFind(Sharded<V>& table, const std::atomic<uint64_t>& epoch, ThreadCache<V>& cache, uint64_t owner, uint64_t k)
{
    const uint64_t e = epoch.load(std::memory_order_acquire);
    if (cache.owner != owner || cache.epoch != e)
    {
        cache.map.clear();
        cache.owner = owner;
        cache.epoch = e;
    }
    auto it = cache.map.find(k);
    if (it != cache.map.end()) return &it->second;
    V v;
    if (!table.get(k, v)) return nullptr;
    return &cache.map.emplace(k, std::move(v)).first->second;
}

// Locking. The hot paths (recording on X-Plane's 24 worker threads, descriptor and object churn)
// take only shard locks. A command buffer's Command is found through `commands` and then used
// without a lock: Vulkan requires a command buffer and its pool to be externally synchronized, so
// only the thread recording it touches it. A descriptor set is likewise externally synchronized for
// updates, but its slots are read at other threads' dispatches, so they change under its shard lock.
// `guard` (taken before any shard lock, never while holding one) covers the rest, all rare: the
// instance and device maps, the engine pointer, SSR/EASU dispatches, main-scene render passes,
// submissions that carry layer inputs, and presents.
// Why: first live A/B profile (one lock for everything), 24 worker threads waited ~7 ms/frame;
// reader/writer locks nested under that lock were worse (fake_xplane --load 1230 --load-threads 24:
// 4.3 ms of worker time per frame vs 1.8 ms; layer absent 0.1 ms).
std::atomic<uint64_t> deviceSerials{0};
struct Device {
    VkDevice                 handle;
    PFN_vkGetDeviceProcAddr  gdpa;
    const uint64_t           serial = ++deviceSerials;  // thread caches: never matches a later device at the same address
    std::unique_ptr<xpfsr::Engine> engine;      // under guard
    std::atomic<bool>        engineOn{false};   // engine != null, a hint for the hot paths
    Sharded<Image>           images;            // render targets and storage images only
    Sharded<View>            views;             // of tracked images only
    Sharded<VkDeviceSize>    buffers;           // uniform buffers only (SSR/EASU constants)
    Sharded<ShaderRole>      shaders;           // SSR and EASU modules only
    Sharded<Pipeline>        pipelines;         // SSR and EASU pipelines only
    Sharded<PipelineLayoutInfo> pipelineLayouts;
    Sharded<LayoutPtr>       layouts;
    std::atomic<uint64_t>    layoutEpoch{0}, templateEpoch{0};  // ThreadCache
    Sharded<Set>             sets;              // tracked sets only
    Sharded<std::shared_ptr<const UpdateTemplate>> templates;
    Sharded<std::vector<uint64_t>> poolSets;    // descriptor pool -> tracked sets allocated from it
    Sharded<std::unique_ptr<Command>> commands;
    std::atomic<uint64_t>    commandEpoch{0};   // bumped before a command buffer's entry is erased (t_command)
    Sharded<std::vector<uint64_t>> pools;       // command pool -> its command buffers
    Sharded<RenderPassInfo>  renderPasses;
    Sharded<std::vector<VkImageView>> framebuffers;  // empty: imageless
    Sharded<bool>            staticViewport;    // graphics pipelines with the viewport baked in
    Sharded<bool>            gbufferImages, mainDepthImages;
    int                      censusBudget = 0;  // under guard: main passes still to describe (main_pass_census)
    std::atomic<uint64_t>    staticViewportBinds{0}, jitteredPasses{0}, nextRecording{0}, submitCalls{0};
    // Observation for the first live run (all modes): can main-scene passes be jittered at all?
    std::atomic<uint64_t>    mainPasses{0}, mainViewportSets{0}, mainStaticBinds{0}, mainDynamicBinds{0};
    // Presents, under guard.
    uint64_t presents = 0;
    std::chrono::steady_clock::time_point lastPresent{};
    double frameMsSum = 0;
    uint64_t frameCount = 0;
    double   frameMsMax = 0;
    std::chrono::steady_clock::time_point lastSummary{};
    PFN_vkVoidFunction next[F_COUNT] = {};  // the next layer's functions (LAYER_NEXT_FUNCTIONS)
    uint64_t profLastCycles = 0;
    std::chrono::steady_clock::time_point profLast{};
    std::vector<uint64_t> profCalls, profSelf;
    uint64_t profLockWaits = 0, profLockWaitCycles = 0, profPresents = 0;
};

thread_local ThreadCache<LayoutPtr>                             t_layouts;
thread_local ThreadCache<std::shared_ptr<const UpdateTemplate>> t_templates;

std::unordered_map<void*, std::shared_ptr<Instance>> instances;
std::unordered_map<void*, std::shared_ptr<Device>>   devices;
template <class T> auto instance(T h) { GuardLock l; return instances.at(key(h)); }
struct DeviceSlot { std::atomic<void*> key{nullptr}; std::atomic<Device*> device{nullptr}; };
DeviceSlot deviceSlots[8];  // lock-free lookup for the hot path; `devices` owns the objects
template <class T> Device* device(T h)
{
    void* k = key(h);
    for (auto& s : deviceSlots)
        if (s.key.load(std::memory_order_acquire) == k) return s.device.load(std::memory_order_relaxed);
    GuardLock l;
    return devices.at(k).get();
}

// The command buffer's state, created if the layer has not seen the command buffer (see Device for
// why it is used without a lock). A thread records many commands into one command buffer in a row,
// so each thread remembers the last one it looked up; any erase (commandEpoch) forgets it.
struct CommandCache { uint64_t device = 0, epoch = 0; VkCommandBuffer cb = VK_NULL_HANDLE; Command* command = nullptr; };
thread_local CommandCache t_command;
Command* findCommand(Device& d, VkCommandBuffer cb, bool create = false)
{
    const uint64_t e = d.commandEpoch.load(std::memory_order_acquire);
    if (t_command.device == d.serial && t_command.cb == cb && t_command.epoch == e) return t_command.command;
    Command* c = nullptr;
    if (create)
        d.commands.upsert(id(cb), [&](std::unique_ptr<Command>& p) {
            if (!p) p = std::make_unique<Command>();
            c = p.get();
        });
    else
        d.commands.with(id(cb), [&](std::unique_ptr<Command>& p) { c = p.get(); });
    if (c) t_command = {d.serial, e, cb, c};
    return c;
}
Command& command(Device& d, VkCommandBuffer cb) { return *findCommand(d, cb, true); }

// status.txt in XP_FSR3_OUTPUT: the user-facing state, replaced atomically (written next to it, renamed).
// A shader rejected by shader_classify (an X-Plane whose interface this version was not written for)
// is explained at the top of every later status, which would otherwise only say FSR is not running.
std::mutex  statusMutex;
std::string unsupportedNote;  // under statusMutex
std::string statusGpu;        // under statusMutex: the device X-Plane created, named in every status
void writeStatus(const std::string& body)
{
    std::lock_guard<std::mutex> l(statusMutex);
    if (!targeted()) return;
    const std::string text = unsupportedNote + body;
    const std::filesystem::path dir = outputDir();
    std::error_code             ec;
    std::filesystem::create_directories(dir, ec);
    const std::time_t wall = std::time(nullptr);
    std::tm           tm{};
    localtime_r(&wall, &tm);
    char clock[32];
    std::strftime(clock, sizeof clock, "%Y-%m-%d %H:%M:%S", &tm);
    {
        std::ofstream f(dir / "status.txt.new", std::ios::trunc);
        f << "X-Plane FSR 3.1 retrofit " XP_FSR3_VERSION ", " << clock << (statusGpu.empty() ? "" : ", GPU: " + statusGpu) << "\n" << text;
    }
    std::filesystem::rename(dir / "status.txt.new", dir / "status.txt", ec);
}

void noteUnsupportedShader(const char* reason)
{
    {
        std::lock_guard<std::mutex> l(statusMutex);
        if (!unsupportedNote.empty()) return;
        const bool easu = std::strncmp(reason, "easu", 4) == 0;
        unsupportedNote = std::string("UNSUPPORTED X-Plane version: its ") + (easu ? "FSR upscale" : "screen-space reflection") +
                          " shader is not the one this version of the mod was written for (" + reason +
                          ").\nX-Plane's own FSR 1 is used. Look for a newer version of the mod.\n\n";
    }
    writeStatus("");
}

void disableEngine(Device& d, const std::string& why)  // under guard
{
    writeStatus("STOPPED after an error (" + why + ").\nX-Plane's own FSR 1 is used for the rest of this session.\n");
    event("engine_disabled", "\"reason\":\"" + why + "\"");
    d.engineOn.store(false, std::memory_order_relaxed);
    d.engine.reset();
}

// ---------------------------------------------------------------------------------------------
// Descriptor tracking. Called with the set's shard lock held.
void update(Set& set, const VkWriteDescriptorSet& w)
{
    uint32_t binding = w.dstBinding, element = w.dstArrayElement;
    for (uint32_t i = 0; i < w.descriptorCount; ++i)
    {
        auto bi = set.layout->bindings.find(binding);
        while (bi != set.layout->bindings.end() && element >= bi->second.descriptorCount)
        {
            element -= bi->second.descriptorCount;
            ++bi;
        }
        if (bi == set.layout->bindings.end()) { set.complete = false; return; }
        binding = bi->first;
        Slot s{};
        s.type = w.descriptorType;
        switch (s.type)
        {
        case VK_DESCRIPTOR_TYPE_SAMPLER: case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
            if (w.pImageInfo) s.image = w.pImageInfo[i];
            break;
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC: case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
            if (w.pBufferInfo) s.buffer = w.pBufferInfo[i];
            break;
        default: set.complete = false; break;
        }
        set.slots[{binding, element}] = s;
        ++element;
    }
}

// The source set is copied out under its own shard lock, then the destination is written under its
// lock (never two shard locks at once). An untracked source leaves the destination incomplete.
void copySlots(Device& d, const VkCopyDescriptorSet& c)
{
    if (!d.sets.contains(id(c.dstSet))) return;
    Set        src;
    const bool haveSource = d.sets.get(id(c.srcSet), src);
    d.sets.with(id(c.dstSet), [&](Set& dst) {
        if (!haveSource) { dst.complete = false; return; }
        std::vector<Slot> slots;
        auto*    layout  = src.layout.get();
        uint32_t binding = c.srcBinding, element = c.srcArrayElement;
        for (uint32_t i = 0; i < c.descriptorCount; ++i)
        {
            auto bi = layout->bindings.find(binding);
            while (bi != layout->bindings.end() && element >= bi->second.descriptorCount) { element -= bi->second.descriptorCount; ++bi; }
            if (bi == layout->bindings.end()) { dst.complete = false; return; }
            binding = bi->first;
            auto it = src.slots.find({binding, element++});
            if (it == src.slots.end()) { dst.complete = false; return; }
            slots.push_back(it->second);
        }
        layout  = dst.layout.get();
        binding = c.dstBinding;
        element = c.dstArrayElement;
        for (const auto& s : slots)
        {
            auto bi = layout->bindings.find(binding);
            while (bi != layout->bindings.end() && element >= bi->second.descriptorCount) { element -= bi->second.descriptorCount; ++bi; }
            if (bi == layout->bindings.end()) { dst.complete = false; return; }
            binding = bi->first;
            dst.slots[{binding, element++}] = s;
        }
        dst.complete &= src.complete;
    });
}

void templateUpdate(Set& set, const UpdateTemplate* t, const void* data)
{
    if (!t || t->type != VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET)
    {
        set.complete = false;
        return;
    }
    for (const auto& e : t->entries)
        for (uint32_t i = 0; i < e.descriptorCount; ++i)
        {
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstBinding      = e.dstBinding;
            w.dstArrayElement = e.dstArrayElement + i;
            w.descriptorCount = 1;
            w.descriptorType  = e.descriptorType;
            auto p            = static_cast<const char*>(data) + e.offset + i * e.stride;
            VkDescriptorImageInfo  image{};
            VkDescriptorBufferInfo buffer{};
            switch (e.descriptorType)
            {
            case VK_DESCRIPTOR_TYPE_SAMPLER: case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE: case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
            case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
                std::memcpy(&image, p, sizeof(image)); w.pImageInfo = &image; break;
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC: case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
                std::memcpy(&buffer, p, sizeof(buffer)); w.pBufferInfo = &buffer; break;
            default: break;
            }
            update(set, w);
        }
}

// ---------------------------------------------------------------------------------------------
// Interception: what X-Plane has bound at an SSR or EASU dispatch, copied out of the tables.
struct Bound {
    Command*            command = nullptr;
    Pipeline            pipeline;
    Set                 set0;
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> offsets0;  // set 0's dynamic offsets by (binding, element)
    xpfsr::ComputeState state;                                   // to restore after the layer's passes
};

// Set `number`'s dynamic offsets, from the bind call that bound it. A call's offsets run in set
// order, then binding, then array element, so the call's earlier sets are skipped using the set
// layouts of the pipeline layout the call named; their total must match what the call passed.
bool decodeOffsets(Device& d, const BoundSet& b, uint32_t number, std::vector<uint32_t>& raw, LayoutPtr& layout)
{
    PipelineLayoutInfo pl;
    if (b.offsetCount > kMaxBindOffsets || !d.pipelineLayouts.get(id(b.layout), pl) || b.first + b.count > pl.setLayouts.size())
        return false;
    uint32_t start = 0, total = 0;
    for (uint32_t k = b.first; k < b.first + b.count; ++k)
    {
        if (!pl.setLayouts[k]) return false;
        if (k < number) start += pl.setLayouts[k]->dynamicCount;
        total += pl.setLayouts[k]->dynamicCount;
    }
    if (total != b.offsetCount) return false;
    layout = pl.setLayouts[number];
    raw.assign(b.offsets.begin() + start, b.offsets.begin() + start + layout->dynamicCount);
    return true;
}

std::string boundIssue(Device& d, VkCommandBuffer cmd, ShaderRole want, Bound& out)
{
    Command* c = findCommand(d, cmd);
    if (!c) return "untracked_command_buffer";
    if (!c->oneTime) return "command_buffer_not_one_time_submit";
    if (!c->known) return "untracked_binding_command";
    if (!d.pipelines.get(id(c->pipeline), out.pipeline) || out.pipeline.role != want) return "role";
    PipelineLayoutInfo li;
    if (!d.pipelineLayouts.get(id(out.pipeline.layout), li)) return "unknown_pipeline_layout";
    if (li.pushRanges != 0) return "push_constants";  // the layer's passes would disturb them
    if (li.sets > kMaxBoundSets) return "too_many_sets";
    uint32_t boundCount = 0;
    for (const auto& b : c->bound) boundCount += b.set != VK_NULL_HANDLE;
    if (boundCount != li.sets) return "sets_not_all_bound";
    out.state.pipeline = c->pipeline;
    out.state.layout   = out.pipeline.layout;
    for (uint32_t n = 0; n < kMaxBoundSets; ++n)
    {
        const BoundSet& b = c->bound[n];
        if (!b.set) continue;
        std::vector<uint32_t> raw;
        LayoutPtr             layout;
        if (!decodeOffsets(d, b, n, raw, layout)) return "dynamic_offsets_unknown";
        if (!d.sets.contains(id(b.set))) return "unknown_descriptor_set";
        if (n == 0)
        {
            size_t i = 0;
            for (const auto& [binding, description] : layout->bindings)
                if (description.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ||
                    description.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)
                    for (uint32_t element = 0; element < description.descriptorCount; ++element) out.offsets0[{binding, element}] = raw[i++];
        }
        out.state.sets.push_back({n, b.set, std::move(raw)});
    }
    if (!c->bound[0].set) return "set0_unbound";
    if (!d.sets.get(id(c->bound[0].set), out.set0)) return "unknown_descriptor_set";
    if (!out.set0.layout->track || !out.set0.complete || !out.set0.layout->ordinary) return "set0_incomplete";
    out.command = c;
    return {};
}

// The effective range of a uniform-buffer slot, at least `need` bytes, or false.
bool uniform(Device& d, const Bound& b, uint32_t binding, VkDeviceSize need, VkDescriptorBufferInfo& out)
{
    auto it = b.set0.slots.find({binding, 0});
    if (it == b.set0.slots.end()) return false;
    const Slot& s = it->second;
    out           = s.buffer;
    if (s.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC)
    {
        auto o = b.offsets0.find({binding, 0});
        if (o == b.offsets0.end()) return false;
        out.offset += o->second;
    }
    else if (s.type != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
        return false;
    VkDeviceSize size = 0;
    if (!d.buffers.get(id(out.buffer), size) || out.offset >= size) return false;
    VkDeviceSize range = s.buffer.range == VK_WHOLE_SIZE ? size - out.offset : s.buffer.range;
    range              = std::min(range, size - out.offset);
    if (range < need) return false;
    out.range = std::min<VkDeviceSize>(range, 65536);
    return true;
}

bool imageSlot(Device& d, const Bound& b, uint32_t binding, VkImageViewType type, VkDescriptorImageInfo& info, VkExtent2D& extent,
               uint32_t& layers)
{
    auto it = b.set0.slots.find({binding, 0});
    if (it == b.set0.slots.end()) return false;
    View v;
    if (!d.views.get(id(it->second.image.imageView), v) || v.type != type || v.range.baseMipLevel != 0) return false;
    Image im;
    if (!d.images.get(id(v.image), im)) return false;
    info   = it->second.image;
    extent = {im.extent.width, im.extent.height};
    layers = v.range.layerCount == VK_REMAINING_ARRAY_LAYERS ? im.layers - v.range.baseArrayLayer : v.range.layerCount;
    return true;
}

void onSsrDispatch(Device& d, VkCommandBuffer cmd, ShaderRole role, uint32_t z)  // under guard
{
    Bound       b;
    std::string issue = boundIssue(d, cmd, role, b);
    static int  logged = 0;
    xpfsr::SsrBinding s;
    s.array  = role == ShaderRole::SsrSceneArray;
    s.layers = z;
    const VkImageViewType type = s.array ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    VkDescriptorImageInfo pos{}, vel{};
    VkExtent2D            posExtent{}, velExtent{};
    uint32_t              posLayers = 0, velLayers = 0;
    if (issue.empty() && !imageSlot(d, b, 3, type, pos, posExtent, posLayers)) issue = "gbuffer_pos";
    if (issue.empty() && !imageSlot(d, b, 4, type, vel, velExtent, velLayers)) issue = "gbuffer_vel";
    if (issue.empty() && (posExtent.width != velExtent.width || posExtent.height != velExtent.height)) issue = "pos_vel_size";
    if (issue.empty() && !uniform(d, b, 11, 240, s.gbufferData)) issue = "u_gbuffer_data";
    if (issue.empty() && !uniform(d, b, 12, 384, s.ssrData)) issue = "u_ssr_data";
    if (!issue.empty())
    {
        if (++logged <= 5) event("ssr_unusable", "\"reason\":\"" + issue + "\"");
        return;
    }
    s.pos = pos.imageView; s.posLayout = pos.imageLayout;
    s.vel = vel.imageView; s.velLayout = vel.imageLayout;
    // The render passes that write these are the main G-buffer passes; their depth buffer is the
    // main scene depth (viewport jitter, below).
    // First live run: gbuffer_pos and gbuffer_vel looked like the depth and stencil views of one
    // depth/stencil image, which is then the main scene depth itself.
    for (VkImageView v : {pos.imageView, vel.imageView})
    {
        View view;
        if (!d.views.get(id(v), view)) continue;
        const uint64_t image = id(view.image);
        if (d.gbufferImages.contains(image)) continue;
        d.gbufferImages.put(image, true);
        Image      info;
        const bool known        = d.images.get(image, info);
        const bool depthStencil = known && (info.usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
        event("gbuffer_image", "\"image\":" + std::to_string(image) + ",\"binding\":\"" + (v == pos.imageView ? "pos" : "vel") +
                                   "\",\"format\":" + std::to_string(known ? info.format : 0) +
                                   ",\"usage\":" + std::to_string(known ? info.usage : 0) +
                                   ",\"aspect\":" + std::to_string(view.range.aspectMask) +
                                   ",\"depth_stencil\":" + (depthStencil ? "true" : "false"));
        if (depthStencil && !d.mainDepthImages.contains(image))
        {
            d.mainDepthImages.put(image, true);
            d.censusBudget = 40;
            event("main_depth", "\"image\":" + std::to_string(image) + ",\"via\":\"gbuffer_is_depth\"");
        }
    }
    s.extent = posExtent;
    if (const uint64_t seq = d.engine->onSsr(cmd, b.command->recording, s, b.state)) b.command->inputs.push_back({true, seq});
}

bool onEasuDispatch(Device& d, VkCommandBuffer cmd, uint32_t z)  // under guard
{
    Bound       b;
    std::string issue = boundIssue(d, cmd, ShaderRole::Easu, b);
    xpfsr::EasuBinding e;
    e.layers = z;
    VkDescriptorImageInfo in{}, out{};
    uint32_t              inLayers = 0, outLayers = 0;
    if (issue.empty() && !imageSlot(d, b, 1, VK_IMAGE_VIEW_TYPE_2D_ARRAY, in, e.inputExtent, inLayers)) issue = "easu_input";
    if (issue.empty() && !imageSlot(d, b, 2, VK_IMAGE_VIEW_TYPE_2D_ARRAY, out, e.outputExtent, outLayers)) issue = "easu_output";
    if (issue.empty() && !uniform(d, b, 3, 80, e.constants)) issue = "u_fsr_data";
    if (!issue.empty())
    {
        static int logged = 0;
        if (++logged <= 5) event("easu_unusable", "\"reason\":\"" + issue + "\"");
        return false;
    }
    e.input = in.imageView; e.inputLayout = in.imageLayout;
    e.output = out.imageView; e.outputLayout = out.imageLayout;
    uint64_t   readSeq  = 0;
    const bool replaced = d.engine->onEasu(cmd, b.command->recording, e, b.state, readSeq);
    if (readSeq) b.command->inputs.push_back({false, readSeq});
    return replaced;
}
}  // namespace

// =============================================================================================
EXPORT PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance, const char*);
EXPORT PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice, const char*);

EXPORT VkResult VKAPI_CALL vkCreateInstance(const VkInstanceCreateInfo* ci, const VkAllocationCallbacks* a, VkInstance* out)
{
    auto link = reinterpret_cast<const VkLayerInstanceCreateInfo*>(ci->pNext);
    while (link && (link->sType != VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO || link->function != VK_LAYER_LINK_INFO))
        link = reinterpret_cast<const VkLayerInstanceCreateInfo*>(link->pNext);
    if (!link || !link->u.pLayerInfo) return VK_ERROR_INITIALIZATION_FAILED;
    auto gipa  = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    auto gpdpa = link->u.pLayerInfo->pfnNextGetPhysicalDeviceProcAddr;
    const_cast<VkLayerInstanceCreateInfo*>(link)->u.pLayerInfo = link->u.pLayerInfo->pNext;
    VkResult r = reinterpret_cast<PFN_vkCreateInstance>(gipa(VK_NULL_HANDLE, "vkCreateInstance"))(ci, a, out);
    if (r == VK_SUCCESS)
    {
        GuardLock l;
        instances[key(*out)] = std::make_shared<Instance>(Instance{*out, gipa, gpdpa});
    }
    return r;
}

EXPORT void VKAPI_CALL vkDestroyInstance(VkInstance h, const VkAllocationCallbacks* a)
{
    auto i = instance(h);
    auto k = key(h);
    reinterpret_cast<PFN_vkDestroyInstance>(i->gipa(h, "vkDestroyInstance"))(h, a);
    GuardLock l;
    instances.erase(k);
}

EXPORT VkResult VKAPI_CALL vkCreateDevice(VkPhysicalDevice physical, const VkDeviceCreateInfo* ci, const VkAllocationCallbacks* a, VkDevice* out)
{
    auto i    = instance(physical);
    auto link = reinterpret_cast<const VkLayerDeviceCreateInfo*>(ci->pNext);
    while (link && (link->sType != VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO || link->function != VK_LAYER_LINK_INFO))
        link = reinterpret_cast<const VkLayerDeviceCreateInfo*>(link->pNext);
    if (!link || !link->u.pLayerInfo) return VK_ERROR_INITIALIZATION_FAILED;
    auto gipa = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    auto gdpa = link->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    const_cast<VkLayerDeviceCreateInfo*>(link)->u.pLayerInfo = link->u.pLayerInfo->pNext;

    // FSR 3.1's fp16 shaders need shaderFloat16 and shaderInt16, which X-Plane does not enable (its
    // log: "shader float16 : 0, shader int16 : 0"), so FSR runs its fp32 shaders. XP_FSR3_FP16=1
    // enables both when the GPU supports them. Opt-in: on an RX 9070 XT at 4233x1666 to 5504x2166
    // (fake_xplane) the fp16 shaders took 2.231 and 2.237 ms against fp32's 2.252 and
    // 2.221 ms, no gain for a change to X-Plane's device. Feature structures in X-Plane's chain are
    // set for the call and restored after it.
    VkDeviceCreateInfo                         info = *ci;
    VkPhysicalDeviceFeatures                   features{};
    VkPhysicalDeviceShaderFloat16Int8Features  added16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    std::vector<std::pair<VkBool32*, VkBool32>> restore;
    bool fp16 = false;
    if (mode() != xpfsr::Mode::Passthrough && std::getenv("XP_FSR3_FP16") && !std::strcmp(std::getenv("XP_FSR3_FP16"), "1"))
    {
        VkPhysicalDeviceShaderFloat16Int8Features has16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
        VkPhysicalDeviceFeatures2                 has{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &has16};
        if (auto get = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(i->gipa(i->handle, "vkGetPhysicalDeviceFeatures2")))
            get(physical, &has);
        fp16 = has16.shaderFloat16 && has.features.shaderInt16;
    }
    // The FidelityFX backend looks for fp16 only when VK_KHR_shader_float16_int8 is among the enabled
    // extensions, and X-Plane enables the features as Vulkan 1.2 core without naming it (X-Plane 12.4's
    // device: 24 extensions, not this one). Name it too, which also makes the prepended
    // Float16Int8 structure valid whatever X-Plane's API version.
    std::vector<const char*> deviceExtensions(ci->ppEnabledExtensionNames, ci->ppEnabledExtensionNames + ci->enabledExtensionCount);
    if (fp16 && std::none_of(deviceExtensions.begin(), deviceExtensions.end(),
                             [](const char* e) { return !std::strcmp(e, VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME); }))
    {
        auto     enumerate = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(i->gipa(i->handle, "vkEnumerateDeviceExtensionProperties"));
        uint32_t n         = 0;
        enumerate(physical, nullptr, &n, nullptr);
        std::vector<VkExtensionProperties> available(n);
        enumerate(physical, nullptr, &n, available.data());
        fp16 = std::any_of(available.begin(), available.begin() + n,
                           [](const VkExtensionProperties& e) { return !std::strcmp(e.extensionName, VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME); });
        if (fp16)
        {
            deviceExtensions.push_back(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
            info.enabledExtensionCount   = static_cast<uint32_t>(deviceExtensions.size());
            info.ppEnabledExtensionNames = deviceExtensions.data();
        }
    }
    if (fp16)
    {
        VkPhysicalDeviceFeatures2*                 f2  = nullptr;
        VkPhysicalDeviceVulkan12Features*          v12 = nullptr;
        VkPhysicalDeviceShaderFloat16Int8Features* f16 = nullptr;
        for (auto n = static_cast<const VkBaseInStructure*>(ci->pNext); n; n = n->pNext)
        {
            auto m = const_cast<VkBaseInStructure*>(n);
            if (n->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2) f2 = reinterpret_cast<VkPhysicalDeviceFeatures2*>(m);
            if (n->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES) v12 = reinterpret_cast<VkPhysicalDeviceVulkan12Features*>(m);
            if (n->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES)
                f16 = reinterpret_cast<VkPhysicalDeviceShaderFloat16Int8Features*>(m);
        }
        auto enable = [&](VkBool32* flag) { restore.push_back({flag, *flag}); *flag = VK_TRUE; };
        if (f2) enable(&f2->features.shaderInt16);
        else
        {
            if (ci->pEnabledFeatures) features = *ci->pEnabledFeatures;
            features.shaderInt16  = VK_TRUE;
            info.pEnabledFeatures = &features;
        }
        if (v12) enable(&v12->shaderFloat16);
        else if (f16) enable(&f16->shaderFloat16);
        else
        {
            added16.shaderFloat16 = VK_TRUE;
            added16.pNext         = const_cast<void*>(info.pNext);
            info.pNext            = &added16;
        }
    }
    VkResult r = reinterpret_cast<PFN_vkCreateDevice>(gipa(i->handle, "vkCreateDevice"))(physical, &info, a, out);
    for (auto& [flag, value] : restore) *flag = value;
    if (r != VK_SUCCESS) return r;

    auto d    = std::make_shared<Device>();
    d->handle = *out;
    d->gdpa   = gdpa;
    for (int f = 0; f < F_COUNT; ++f) d->next[f] = gdpa(*out, kNextNames[f]);
    if (!d->next[F_vkCmdSetViewportWithCount]) d->next[F_vkCmdSetViewportWithCount] = d->next[F_vkCmdSetViewportWithCountEXT];
    if (mode() != xpfsr::Mode::Passthrough)
    {
        xpfsr::DeviceSetup setup;
        setup.instance = i->handle;
        setup.physical = physical;
        setup.device   = *out;
        setup.gipa     = i->gipa;
        setup.gdpa     = gdpa;
        setup.mode     = mode();
        setup.log      = event;
        setup.status   = writeStatus;
        setup.shaderFloat16 = setup.shaderInt16 = fp16;  // enabled above
        // from_chars, not atoi: glibc 2.38+ headers turn atoi into __isoc23_strtol, which would raise the
        // layer's glibc floor above X-Plane's own (2.34).
        if (const char* ab = std::getenv("XP_FSR3_AB"))
        {
            int seconds = 0;
            std::from_chars(ab, ab + std::strlen(ab), seconds);
            setup.abSeconds = std::max(0, seconds);
        }
        for (uint32_t e = 0; e < info.enabledExtensionCount; ++e) setup.enabledExtensions.push_back(info.ppEnabledExtensionNames[e]);
        if (ci->pEnabledFeatures) setup.shaderInt16 |= ci->pEnabledFeatures->shaderInt16;
        for (auto n = static_cast<const VkBaseInStructure*>(ci->pNext); n; n = n->pNext)
        {
            if (n->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2)
                setup.shaderInt16 |= reinterpret_cast<const VkPhysicalDeviceFeatures2*>(n)->features.shaderInt16;
            if (n->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES)
                setup.shaderFloat16 |= reinterpret_cast<const VkPhysicalDeviceVulkan12Features*>(n)->shaderFloat16;
            if (n->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES)
                setup.shaderFloat16 |= reinterpret_cast<const VkPhysicalDeviceShaderFloat16Int8Features*>(n)->shaderFloat16;
        }
        VkPhysicalDeviceProperties props{};
        reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(i->gipa(i->handle, "vkGetPhysicalDeviceProperties"))(physical, &props);
        std::string extensions;
        for (auto& e : setup.enabledExtensions) extensions += (extensions.empty() ? "\"" : ",\"") + e + "\"";
        event("device", std::string("\"version\":\"" XP_FSR3_VERSION "\",\"name\":\"") + props.deviceName + "\",\"fp16\":" + (setup.shaderFloat16 && setup.shaderInt16 ? "true" : "false") +
                            ",\"fp16_enabled_by_layer\":" + (fp16 ? "true" : "false") +
                            ",\"extensions\":[" + extensions + "]");
        {
            std::lock_guard<std::mutex> l(statusMutex);
            statusGpu = props.deviceName;
        }
        try
        {
            d->engine = std::make_unique<xpfsr::Engine>(setup);
            d->engineOn = true;
            writeStatus(std::string("LOADED on ") + props.deviceName + ", waiting for X-Plane's FSR upscale.\n"
                        "If this does not change within a minute of a flight loading, set Settings > Graphics >\n"
                        "Rendering Resolution (FSR Supersampling) to anything but Off.\n");
        }
        catch (const std::exception& e)
        {
            writeStatus(std::string("OFF: FSR 3.1 could not start on ") + props.deviceName + " (" + e.what() +
                        ").\nX-Plane's own FSR 1 is used.\n");
            event("engine_disabled", std::string("\"reason\":\"create: ") + e.what() + "\"");
        }
    }
    GuardLock l;
    devices[key(*out)] = d;
    for (auto& s : deviceSlots)
        if (!s.key.load(std::memory_order_relaxed))
        {
            s.device.store(d.get(), std::memory_order_relaxed);
            s.key.store(key(*out), std::memory_order_release);
            break;
        }
    return r;
}

EXPORT void VKAPI_CALL vkDestroyDevice(VkDevice h, const VkAllocationCallbacks* a)
{
    HOOK_SCOPE(vkDestroyDevice); auto d = device(h);
    auto k = key(h);
    d->engine.reset();
    NEXT(d, vkDestroyDevice)(h, a);
    GuardLock l;
    for (auto& s : deviceSlots)
        if (s.key.load(std::memory_order_relaxed) == k)
        {
            s.key.store(nullptr, std::memory_order_release);
            s.device.store(nullptr, std::memory_order_relaxed);
        }
    devices.erase(k);
}

// Destroy and free hooks drop the layer's entry before the call: once the driver has released a
// handle, another thread may be handed the same value and register it.
EXPORT VkResult VKAPI_CALL vkCreateImage(VkDevice h, const VkImageCreateInfo* ci, const VkAllocationCallbacks* a, VkImage* out)
{
    HOOK_SCOPE(vkCreateImage); auto d = device(h);
    auto r = NEXT(d, vkCreateImage)(h, ci, a, out);
    const VkImageUsageFlags renderTarget =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    if (r == VK_SUCCESS && (ci->usage & renderTarget)) d->images.put(id(*out), {ci->extent, ci->arrayLayers, ci->usage, ci->format});
    return r;
}
EXPORT void VKAPI_CALL vkDestroyImage(VkDevice h, VkImage im, const VkAllocationCallbacks* a)
{
    HOOK_SCOPE(vkDestroyImage); auto d = device(h);
    if (d->images.erase(id(im)))  // else never tracked (textures)
    {
        // A later image may reuse the handle (first live run: a stale entry matched an unrelated pass).
        d->gbufferImages.erase(id(im));
        d->mainDepthImages.erase(id(im));
    }
    NEXT(d, vkDestroyImage)(h, im, a);
}
EXPORT VkResult VKAPI_CALL vkCreateImageView(VkDevice h, const VkImageViewCreateInfo* ci, const VkAllocationCallbacks* a, VkImageView* out)
{
    HOOK_SCOPE(vkCreateImageView); auto d = device(h);
    auto r = NEXT(d, vkCreateImageView)(h, ci, a, out);
    if (r == VK_SUCCESS && d->images.contains(id(ci->image)))  // views of untracked images are not needed
        d->views.put(id(*out), {ci->image, ci->viewType, ci->subresourceRange});
    return r;
}
EXPORT void VKAPI_CALL vkDestroyImageView(VkDevice h, VkImageView v, const VkAllocationCallbacks* a)
{
    HOOK_SCOPE(vkDestroyImageView); auto d = device(h);
    d->views.erase(id(v));
    NEXT(d, vkDestroyImageView)(h, v, a);
}
EXPORT VkResult VKAPI_CALL vkCreateBuffer(VkDevice h, const VkBufferCreateInfo* ci, const VkAllocationCallbacks* a, VkBuffer* out)
{
    HOOK_SCOPE(vkCreateBuffer); auto d = device(h);
    auto r = NEXT(d, vkCreateBuffer)(h, ci, a, out);
    if (r == VK_SUCCESS && (ci->usage & VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)) d->buffers.put(id(*out), ci->size);  // SSR/EASU constants only
    return r;
}
EXPORT void VKAPI_CALL vkDestroyBuffer(VkDevice h, VkBuffer b, const VkAllocationCallbacks* a)
{
    HOOK_SCOPE(vkDestroyBuffer); auto d = device(h);
    d->buffers.erase(id(b));
    NEXT(d, vkDestroyBuffer)(h, b, a);
}
EXPORT VkResult VKAPI_CALL vkCreateShaderModule(VkDevice h, const VkShaderModuleCreateInfo* ci, const VkAllocationCallbacks* a, VkShaderModule* out)
{
    HOOK_SCOPE(vkCreateShaderModule); auto d = device(h);
    auto r = NEXT(d, vkCreateShaderModule)(h, ci, a, out);
    if (r == VK_SUCCESS)
    {
        // By interface, not by hash, so any X-Plane build with the same interface works
        // (shader_classify.h). A near miss means X-Plane changed what the layer relies on.
        const char*      reason = nullptr;
        const ShaderRole role   = classifyShader(ci->pCode, ci->codeSize / 4, &reason);
        if (role != ShaderRole::None) d->shaders.put(id(*out), role);
        if (reason) noteUnsupportedShader(reason);
        if (role != ShaderRole::None || reason)
            event("shader", std::string("\"role\":\"") + roleName(role) + "\",\"bytes\":" + std::to_string(ci->codeSize) +
                                (reason ? std::string(",\"rejected\":\"") + reason + "\"" : std::string()));
    }
    return r;
}
EXPORT void VKAPI_CALL vkDestroyShaderModule(VkDevice h, VkShaderModule s, const VkAllocationCallbacks* a)
{
    HOOK_SCOPE(vkDestroyShaderModule); auto d = device(h);
    d->shaders.erase(id(s));
    NEXT(d, vkDestroyShaderModule)(h, s, a);
}
EXPORT VkResult VKAPI_CALL vkCreateComputePipelines(VkDevice h, VkPipelineCache cache, uint32_t n, const VkComputePipelineCreateInfo* ci,
                                                    const VkAllocationCallbacks* a, VkPipeline* out)
{
    HOOK_SCOPE(vkCreateComputePipelines); auto d = device(h);
    auto r = NEXT(d, vkCreateComputePipelines)(h, cache, n, ci, a, out);
    for (uint32_t i = 0; i < n; ++i)
    {
        ShaderRole role = ShaderRole::None;
        if (out[i] && d->shaders.get(id(ci[i].stage.module), role))
        {
            d->pipelines.put(id(out[i]), {role, ci[i].layout});
            event("pipeline", "\"role\":" + std::to_string(int(role)));
        }
    }
    return r;
}
EXPORT void VKAPI_CALL vkDestroyPipeline(VkDevice h, VkPipeline p, const VkAllocationCallbacks* a)
{
    HOOK_SCOPE(vkDestroyPipeline); auto d = device(h);
    d->pipelines.erase(id(p));
    d->staticViewport.erase(id(p));
    NEXT(d, vkDestroyPipeline)(h, p, a);
}
EXPORT VkResult VKAPI_CALL vkCreatePipelineLayout(VkDevice h, const VkPipelineLayoutCreateInfo* ci, const VkAllocationCallbacks* a, VkPipelineLayout* out)
{
    HOOK_SCOPE(vkCreatePipelineLayout); auto d = device(h);
    auto r = NEXT(d, vkCreatePipelineLayout)(h, ci, a, out);
    if (r == VK_SUCCESS)
    {
        PipelineLayoutInfo info{ci->setLayoutCount, ci->pushConstantRangeCount, {}};
        for (uint32_t k = 0; k < ci->setLayoutCount; ++k)
        {
            LayoutPtr layout;  // stays null if unknown: offsets through it are then never decoded
            d->layouts.get(id(ci->pSetLayouts[k]), layout);
            info.setLayouts.push_back(std::move(layout));
        }
        d->pipelineLayouts.put(id(*out), std::move(info));
    }
    return r;
}
EXPORT void VKAPI_CALL vkDestroyPipelineLayout(VkDevice h, VkPipelineLayout p, const VkAllocationCallbacks* a)
{
    HOOK_SCOPE(vkDestroyPipelineLayout); auto d = device(h);
    d->pipelineLayouts.erase(id(p));
    NEXT(d, vkDestroyPipelineLayout)(h, p, a);
}
EXPORT VkResult VKAPI_CALL vkCreateDescriptorSetLayout(VkDevice h, const VkDescriptorSetLayoutCreateInfo* ci, const VkAllocationCallbacks* a,
                                                       VkDescriptorSetLayout* out)
{
    HOOK_SCOPE(vkCreateDescriptorSetLayout); auto d = device(h);
    auto r = NEXT(d, vkCreateDescriptorSetLayout)(h, ci, a, out);
    if (r == VK_SUCCESS)
    {
        DescriptorLayout layout;
        layout.ordinary = ci->flags == 0 && !ci->pNext;
        for (uint32_t j = 0; j < ci->bindingCount; ++j)
        {
            auto b               = ci->pBindings[j];
            b.pImmutableSamplers = nullptr;
            layout.bindings[b.binding] = b;
            layout.track |= b.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            if (b.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC || b.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)
                layout.dynamicCount += b.descriptorCount;
        }
        d->layouts.put(id(*out), std::make_shared<const DescriptorLayout>(std::move(layout)));
    }
    return r;
}
EXPORT void VKAPI_CALL vkDestroyDescriptorSetLayout(VkDevice h, VkDescriptorSetLayout s, const VkAllocationCallbacks* a)
{
    HOOK_SCOPE(vkDestroyDescriptorSetLayout); auto d = device(h);
    d->layoutEpoch.fetch_add(1, std::memory_order_release);
    d->layouts.erase(id(s));
    NEXT(d, vkDestroyDescriptorSetLayout)(h, s, a);
}
EXPORT VkResult VKAPI_CALL vkAllocateDescriptorSets(VkDevice h, const VkDescriptorSetAllocateInfo* ci, VkDescriptorSet* out)
{
    HOOK_SCOPE(vkAllocateDescriptorSets); auto d = device(h);
    auto r = NEXT(d, vkAllocateDescriptorSets)(h, ci, out);
    if (r != VK_SUCCESS) return r;
    // Only sets whose layout could be SSR's or EASU's are tracked.
    std::vector<uint64_t> tracked;
    for (uint32_t j = 0; j < ci->descriptorSetCount; ++j)
    {
        const LayoutPtr* layout = cachedFind(d->layouts, d->layoutEpoch, t_layouts, d->serial, id(ci->pSetLayouts[j]));
        if (!layout || !*layout || !(*layout)->track) continue;
        d->sets.put(id(out[j]), Set{ci->descriptorPool, *layout, {}, true});
        tracked.push_back(id(out[j]));
    }
    if (!tracked.empty())
        d->poolSets.upsert(id(ci->descriptorPool), [&](std::vector<uint64_t>& v) { v.insert(v.end(), tracked.begin(), tracked.end()); });
    return r;
}
EXPORT VkResult VKAPI_CALL vkFreeDescriptorSets(VkDevice h, VkDescriptorPool pool, uint32_t n, const VkDescriptorSet* sets)
{
    HOOK_SCOPE(vkFreeDescriptorSets); auto d = device(h);
    for (uint32_t j = 0; j < n; ++j) d->sets.eraseIf(id(sets[j]), [&](Set& s) { return s.pool == pool; });
    return NEXT(d, vkFreeDescriptorSets)(h, pool, n, sets);
}
// Only the sets allocated from this pool: a reset used to scan every tracked set of every pool.
static void erasePool(Device& d, VkDescriptorPool pool)
{
    std::vector<uint64_t> ids;
    if (!d.poolSets.take(id(pool), ids)) return;  // the common case: no tracked sets in this pool
    for (uint64_t s : ids) d.sets.eraseIf(s, [&](Set& x) { return x.pool == pool; });  // a freed handle may be reused by another pool
}
EXPORT VkResult VKAPI_CALL vkResetDescriptorPool(VkDevice h, VkDescriptorPool pool, VkDescriptorPoolResetFlags flags)
{
    HOOK_SCOPE(vkResetDescriptorPool); auto d = device(h);
    erasePool(*d, pool);
    return NEXT(d, vkResetDescriptorPool)(h, pool, flags);
}
EXPORT void VKAPI_CALL vkDestroyDescriptorPool(VkDevice h, VkDescriptorPool pool, const VkAllocationCallbacks* a)
{
    HOOK_SCOPE(vkDestroyDescriptorPool); auto d = device(h);
    erasePool(*d, pool);
    NEXT(d, vkDestroyDescriptorPool)(h, pool, a);
}
EXPORT void VKAPI_CALL vkUpdateDescriptorSets(VkDevice h, uint32_t n, const VkWriteDescriptorSet* writes, uint32_t count, const VkCopyDescriptorSet* copies)
{
    HOOK_SCOPE(vkUpdateDescriptorSets); auto d = device(h);
    NEXT(d, vkUpdateDescriptorSets)(h, n, writes, count, copies);
    for (uint32_t j = 0; j < n; ++j) d->sets.with(id(writes[j].dstSet), [&](Set& s) { update(s, writes[j]); });
    for (uint32_t j = 0; j < count; ++j) copySlots(*d, copies[j]);
}
#define TEMPLATE_HOOKS(suffix)                                                                                                         \
    EXPORT VkResult VKAPI_CALL vkCreateDescriptorUpdateTemplate##suffix(VkDevice h, const VkDescriptorUpdateTemplateCreateInfo* ci,  \
                                                                        const VkAllocationCallbacks* a, VkDescriptorUpdateTemplate* out) \
    {                                                                                                                                  \
        HOOK_SCOPE(vkCreateDescriptorUpdateTemplate##suffix); auto d = device(h);                                                      \
        auto r = NEXT(d, vkCreateDescriptorUpdateTemplate##suffix)(h, ci, a, out);                                                     \
        if (r == VK_SUCCESS)                                                                                                           \
            d->templates.put(id(*out), std::make_shared<const UpdateTemplate>(UpdateTemplate{                                          \
                                           std::vector<VkDescriptorUpdateTemplateEntry>(ci->pDescriptorUpdateEntries,                  \
                                                                                        ci->pDescriptorUpdateEntries +                 \
                                                                                            ci->descriptorUpdateEntryCount),           \
                                           ci->templateType}));                                                                        \
        return r;                                                                                                                      \
    }                                                                                                                                  \
    EXPORT void VKAPI_CALL vkDestroyDescriptorUpdateTemplate##suffix(VkDevice h, VkDescriptorUpdateTemplate t, const VkAllocationCallbacks* a) \
    {                                                                                                                                  \
        HOOK_SCOPE(vkDestroyDescriptorUpdateTemplate##suffix); auto d = device(h);                                                     \
        d->templateEpoch.fetch_add(1, std::memory_order_release);                                                                      \
        d->templates.erase(id(t));                                                                                                     \
        NEXT(d, vkDestroyDescriptorUpdateTemplate##suffix)(h, t, a);                                                                   \
    }                                                                                                                                  \
    EXPORT void VKAPI_CALL vkUpdateDescriptorSetWithTemplate##suffix(VkDevice h, VkDescriptorSet set, VkDescriptorUpdateTemplate t, const void* data) \
    {                                                                                                                                  \
        HOOK_SCOPE(vkUpdateDescriptorSetWithTemplate##suffix); auto d = device(h);                                                     \
        NEXT(d, vkUpdateDescriptorSetWithTemplate##suffix)(h, set, t, data);                                                           \
        if (!d->sets.contains(id(set))) return; /* untracked: the common case */                                                       \
        auto tp = cachedFind(d->templates, d->templateEpoch, t_templates, d->serial, id(t));                                                   \
        d->sets.with(id(set), [&](Set& s) { templateUpdate(s, tp ? tp->get() : nullptr, data); });                                     \
    }
TEMPLATE_HOOKS()
TEMPLATE_HOOKS(KHR)
#undef TEMPLATE_HOOKS

EXPORT VkResult VKAPI_CALL vkAllocateCommandBuffers(VkDevice h, const VkCommandBufferAllocateInfo* ci, VkCommandBuffer* out)
{
    HOOK_SCOPE(vkAllocateCommandBuffers); auto d = device(h);
    auto r = NEXT(d, vkAllocateCommandBuffers)(h, ci, out);
    if (r == VK_SUCCESS)
    {
        // Reset in place, never replaced: a thread's t_command may point at an existing entry.
        for (uint32_t j = 0; j < ci->commandBufferCount; ++j)
            d->commands.upsert(id(out[j]), [](std::unique_ptr<Command>& p) {
                if (p) *p = Command{};
                else p = std::make_unique<Command>();
            });
        d->pools.upsert(id(ci->commandPool), [&](std::vector<uint64_t>& v) {
            for (uint32_t j = 0; j < ci->commandBufferCount; ++j) v.push_back(id(out[j]));
        });
    }
    return r;
}
EXPORT void VKAPI_CALL vkFreeCommandBuffers(VkDevice h, VkCommandPool pool, uint32_t n, const VkCommandBuffer* cmds)
{
    HOOK_SCOPE(vkFreeCommandBuffers); auto d = device(h);
    d->commandEpoch.fetch_add(1, std::memory_order_release);
    for (uint32_t j = 0; j < n; ++j) d->commands.erase(id(cmds[j]));
    d->pools.with(id(pool), [&](std::vector<uint64_t>& ids) {
        for (uint32_t j = 0; j < n; ++j) ids.erase(std::remove(ids.begin(), ids.end(), id(cmds[j])), ids.end());
    });
    NEXT(d, vkFreeCommandBuffers)(h, pool, n, cmds);
}
EXPORT void VKAPI_CALL vkDestroyCommandPool(VkDevice h, VkCommandPool pool, const VkAllocationCallbacks* a)
{
    HOOK_SCOPE(vkDestroyCommandPool); auto d = device(h);
    std::vector<uint64_t> ids;
    d->pools.take(id(pool), ids);
    d->commandEpoch.fetch_add(1, std::memory_order_release);
    for (auto x : ids) d->commands.erase(x);
    NEXT(d, vkDestroyCommandPool)(h, pool, a);
}
EXPORT VkResult VKAPI_CALL vkResetCommandPool(VkDevice h, VkCommandPool pool, VkCommandPoolResetFlags flags)
{
    HOOK_SCOPE(vkResetCommandPool); auto d = device(h);
    auto r = NEXT(d, vkResetCommandPool)(h, pool, flags);
    if (r == VK_SUCCESS)
    {
        std::vector<uint64_t> ids;
        d->pools.get(id(pool), ids);
        for (auto x : ids) d->commands.with(x, [](std::unique_ptr<Command>& c) { if (c) *c = Command{}; });
    }
    return r;
}
EXPORT VkResult VKAPI_CALL vkBeginCommandBuffer(VkCommandBuffer cmd, const VkCommandBufferBeginInfo* ci)
{
    HOOK_SCOPE(vkBeginCommandBuffer); auto d = device(cmd);
    auto r = NEXT(d, vkBeginCommandBuffer)(cmd, ci);
    if (r == VK_SUCCESS)
    {
        Command& c  = command(*d, cmd);
        c           = Command{};
        c.recording = d->nextRecording.fetch_add(1, std::memory_order_relaxed) + 1;
        // FidelityFX rotates its descriptor sets and constants per dispatch, so a recording must be
        // submitted once. X-Plane's FSR recording uses ONE_TIME_SUBMIT (observed in X-Plane 12.4).
        c.oneTime = (ci->flags & VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT) && !(ci->flags & VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT);
    }
    return r;
}
EXPORT VkResult VKAPI_CALL vkResetCommandBuffer(VkCommandBuffer cmd, VkCommandBufferResetFlags flags)
{
    HOOK_SCOPE(vkResetCommandBuffer); auto d = device(cmd);
    auto r = NEXT(d, vkResetCommandBuffer)(cmd, flags);
    if (r == VK_SUCCESS) command(*d, cmd) = Command{};
    return r;
}
EXPORT void VKAPI_CALL vkCmdBindPipeline(VkCommandBuffer cmd, VkPipelineBindPoint point, VkPipeline pipeline)
{
    HOOK_SCOPE(vkCmdBindPipeline); auto d = device(cmd);
    NEXT(d, vkCmdBindPipeline)(cmd, point, pipeline);
    if (point == VK_PIPELINE_BIND_POINT_COMPUTE)
    {
        Command& c = command(*d, cmd);
        Pipeline p;
        c.pipeline = pipeline;
        c.role     = d->pipelines.get(id(pipeline), p) ? p.role : ShaderRole::None;
    }
    else if (point == VK_PIPELINE_BIND_POINT_GRAPHICS && d->engineOn.load(std::memory_order_relaxed))
    {
        Command& c = command(*d, cmd);
        if (!c.mainPass) return;
        const bool isStatic = d->staticViewport.contains(id(pipeline));
        ++(isStatic ? d->mainStaticBinds : d->mainDynamicBinds);
        ++c.passBinds;
        if (isStatic && c.jittered)
            if (const uint64_t count = ++d->staticViewportBinds; count <= 3)
                event("static_viewport_in_jittered_pass", "\"count\":" + std::to_string(count));
    }
}
EXPORT void VKAPI_CALL vkCmdBindDescriptorSets(VkCommandBuffer cmd, VkPipelineBindPoint point, VkPipelineLayout layout, uint32_t first, uint32_t n,
                                               const VkDescriptorSet* sets, uint32_t ndyn, const uint32_t* offsets)
{
    HOOK_SCOPE(vkCmdBindDescriptorSets); auto d = device(cmd);
    NEXT(d, vkCmdBindDescriptorSets)(cmd, point, layout, first, n, sets, ndyn, offsets);
    if (point != VK_PIPELINE_BIND_POINT_COMPUTE) return;
    Command& c = command(*d, cmd);
    for (uint32_t j = 0; j < n; ++j)
    {
        const uint32_t number = first + j;
        if (number >= kMaxBoundSets) continue;  // beyond any SSR or EASU layout (boundIssue: too_many_sets)
        BoundSet& b   = c.bound[number];
        b.set         = sets[j];
        b.layout      = layout;
        b.first       = first;
        b.count       = n;
        b.offsetCount = ndyn;
        if (ndyn <= kMaxBindOffsets && offsets) std::copy(offsets, offsets + ndyn, b.offsets.begin());
    }
}

EXPORT void VKAPI_CALL vkCmdDispatch(VkCommandBuffer cmd, uint32_t x, uint32_t y, uint32_t z)
{
    HOOK_SCOPE(vkCmdDispatch); auto d = device(cmd);
    ShaderRole role = ShaderRole::None;
    if (x && y && z && d->engineOn.load(std::memory_order_relaxed))
        if (Command* c = findCommand(*d, cmd)) role = c->role;
    if (role == ShaderRole::Easu)
    {
        GuardLock l;
        if (d->engine)
        {
            try
            {
                if (onEasuDispatch(*d, cmd, z)) return;  // replaced: X-Plane's EASU is not recorded
            }
            catch (const std::exception& e)
            {
                disableEngine(*d, std::string("easu: ") + e.what());
            }
        }
        // X-Plane's EASU running unchanged: time it on the GPU (the FSR 1 side of the comparison).
        if (d->engine) d->engine->markEasu(cmd, true);
    }
    NEXT(d, vkCmdDispatch)(cmd, x, y, z);
    if (role == ShaderRole::Easu)
    {
        GuardLock l;
        if (d->engine) d->engine->markEasu(cmd, false);
    }
    if (role == ShaderRole::SsrScene2D || role == ShaderRole::SsrSceneArray)
    {
        GuardLock l;
        if (d->engine)
        {
            try
            {
                onSsrDispatch(*d, cmd, role, z);
            }
            catch (const std::exception& e)
            {
                disableEngine(*d, std::string("ssr: ") + e.what());
            }
        }
    }
}

// cpu_profile: what the layer itself cost per frame since the last report (all threads), from the
// thread-local counters. TSC ticks are converted with the rate measured over the same interval.
static void profileReport(Device& d, std::chrono::steady_clock::time_point now)
{
    const uint64_t cycles = __rdtsc();
    if (d.profCalls.empty()) d.profCalls.assign(F_COUNT, 0), d.profSelf.assign(F_COUNT, 0);
    std::vector<uint64_t> calls(F_COUNT, 0), self(F_COUNT, 0);
    uint64_t waits = 0, waitCycles = 0;
    size_t   threads = 0;
    {
        std::lock_guard<std::mutex> l(profRegistryLock);
        threads = profRegistry.size();
        for (const ProfBlock* b : profRegistry)
        {
            for (int f = 0; f < F_COUNT; ++f) calls[f] += b->calls[f].load(std::memory_order_relaxed), self[f] += b->self[f].load(std::memory_order_relaxed);
            waits += b->lockWaits.load(std::memory_order_relaxed), waitCycles += b->lockWaitCycles.load(std::memory_order_relaxed);
        }
    }
    if (d.profLastCycles)
    {
        const double ns     = std::chrono::duration<double, std::nano>(now - d.profLast).count();
        const double perUs  = (cycles - d.profLastCycles) / (ns / 1000.0);  // TSC ticks per microsecond
        const double frames = double(std::max<uint64_t>(1, d.presents - d.profPresents));
        std::vector<std::pair<double, int>> order;
        double total = 0;
        for (int f = 0; f < F_COUNT; ++f)
        {
            const double us = (self[f] - d.profSelf[f]) / perUs / frames;
            total += us;
            if (calls[f] != d.profCalls[f]) order.push_back({us, f});
        }
        std::sort(order.rbegin(), order.rend());
        std::string top;
        for (size_t i = 0; i < order.size() && i < 14; ++i)
        {
            const int f = order[i].second;
            top += std::string(top.empty() ? "" : ",") + "[\"" + kNextNames[f] + "\"," +
                   std::to_string((calls[f] - d.profCalls[f]) / frames) + "," + std::to_string(order[i].first) + "]";
        }
        event("cpu_profile", "\"frames\":" + std::to_string(uint64_t(frames)) + ",\"layer_us_per_frame\":" + std::to_string(total) +
                                 ",\"lock_waits_per_frame\":" + std::to_string((waits - d.profLockWaits) / frames) +
                                 ",\"lock_wait_us_per_frame\":" + std::to_string((waitCycles - d.profLockWaitCycles) / perUs / frames) +
                                 ",\"threads\":" + std::to_string(threads) + ",\"top_calls_us\":[" + top + "]");
    }
    d.profLastCycles = cycles, d.profLast = now, d.profPresents = d.presents;
    d.profCalls = calls, d.profSelf = self, d.profLockWaits = waits, d.profLockWaitCycles = waitCycles;
}

// Submission order of the command buffers that wrote or read the layer's input images (engine
// onSubmitted), reported after the driver accepted the submission. What to report is copied before
// the call: once submitted, a command buffer may complete and be re-recorded by another thread.
struct Submitted { uint64_t recording; std::vector<xpfsr::InputsEvent> inputs; };
static void collect(Device& d, VkCommandBuffer cb, std::vector<Submitted>& out)
{
    if (const Command* c = findCommand(d, cb); c && !c->inputs.empty()) out.push_back({c->recording, c->inputs});
}
static void report(Device& d, VkQueue q, const std::vector<Submitted>& submitted)
{
    const uint64_t call = ++d.submitCalls;
    if (submitted.empty()) return;
    GuardLock l;
    for (const auto& s : submitted)
    {
        if (!d.engine) return;
        try { d.engine->onSubmitted(q, call, s.recording, s.inputs); }
        catch (const std::exception& e) { disableEngine(d, std::string("submit: ") + e.what()); }
    }
}
EXPORT VkResult VKAPI_CALL vkQueueSubmit(VkQueue q, uint32_t n, const VkSubmitInfo* s, VkFence fence)
{
    HOOK_SCOPE(vkQueueSubmit); auto d = device(q);
    std::vector<Submitted> submitted;
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t j = 0; j < s[i].commandBufferCount; ++j) collect(*d, s[i].pCommandBuffers[j], submitted);
    auto r = NEXT(d, vkQueueSubmit)(q, n, s, fence);
    if (r == VK_SUCCESS) report(*d, q, submitted);
    return r;
}
static VkResult submit2(int which, VkQueue q, uint32_t n, const VkSubmitInfo2* s, VkFence fence)
{
    HookScope hookScope_(which);
    auto d = device(q);
    std::vector<Submitted> submitted;
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t j = 0; j < s[i].commandBufferInfoCount; ++j) collect(*d, s[i].pCommandBufferInfos[j].commandBuffer, submitted);
    auto r = NextCall<PFN_vkQueueSubmit2>{reinterpret_cast<PFN_vkQueueSubmit2>(d->next[which])}(q, n, s, fence);
    if (r == VK_SUCCESS) report(*d, q, submitted);
    return r;
}
EXPORT VkResult VKAPI_CALL vkQueueSubmit2(VkQueue q, uint32_t n, const VkSubmitInfo2* s, VkFence f) { return submit2(F_vkQueueSubmit2, q, n, s, f); }
EXPORT VkResult VKAPI_CALL vkQueueSubmit2KHR(VkQueue q, uint32_t n, const VkSubmitInfo2* s, VkFence f) { return submit2(F_vkQueueSubmit2KHR, q, n, s, f); }

EXPORT VkResult VKAPI_CALL vkQueuePresentKHR(VkQueue q, const VkPresentInfoKHR* ci)
{
    HOOK_SCOPE(vkQueuePresentKHR); auto d = device(q);
    auto r = NEXT(d, vkQueuePresentKHR)(q, ci);
    GuardLock l;
    const auto now = std::chrono::steady_clock::now();
    if (d->presents++ > 0)
    {
        const double ms = std::chrono::duration<double, std::milli>(now - d->lastPresent).count();
        d->frameMsSum += ms;
        d->frameMsMax = std::max(d->frameMsMax, ms);
        ++d->frameCount;
    }
    d->lastPresent = now;
    // Every 10 s (and at present 300), stamped with the wall clock so notes like "switched FSR off
    // at 18:05" can be matched to the numbers.
    if ((d->engine || timingOnly()) && d->presents >= 300 && (d->presents == 300 || now - d->lastSummary >= std::chrono::seconds(10)))
    {
        d->lastSummary = now;
        const std::time_t wall = std::time(nullptr);
        char clock[16];
        std::strftime(clock, sizeof clock, "%H:%M:%S", std::localtime(&wall));
        const double mean = d->frameCount ? d->frameMsSum / d->frameCount : 0.0;
        event("frame_summary", "\"clock\":\"" + std::string(clock) + "\",\"presents\":" + std::to_string(d->presents) +
                                   ",\"fps\":" + std::to_string(mean > 0 ? 1000.0 / mean : 0.0) + ",\"mean_frame_ms\":" + std::to_string(mean) +
                                   ",\"max_frame_ms\":" + std::to_string(d->frameMsMax) + ",\"main_passes\":" +
                                   std::to_string(d->mainPasses.load()) + ",\"main_viewport_sets\":" + std::to_string(d->mainViewportSets.load()) +
                                   ",\"main_dynamic_viewport_binds\":" + std::to_string(d->mainDynamicBinds.load()) +
                                   ",\"main_static_viewport_binds\":" + std::to_string(d->mainStaticBinds.load()) +
                                   ",\"jittered_passes\":" + std::to_string(d->jitteredPasses.load()));
        d->frameMsSum = 0;
        d->frameMsMax = 0;
        d->frameCount = 0;
        if (profiling()) profileReport(*d, now);
    }
    if (d->engine)
    {
        try { d->engine->onPresent(); }
        catch (const std::exception& e) { disableEngine(*d, std::string("present: ") + e.what()); }
    }
    return r;
}

// ---------------------------------------------------------------------------------------------
// Camera jitter by viewport offset. Shifting VkViewport.x/y by (jx, jy) < 0.5 px moves every
// rasterised fragment by exactly that amount: the same as a projection jitter. Applied inside
// render passes whose depth attachment is the main scene depth (learned from the passes that
// write gbuffer_pos / gbuffer_vel), never to shadow, panel-texture or post-processing passes.
static void setViewports(Device& d, VkCommandBuffer cb, Command& c, bool jitter)
{
    if (!c.viewportKnown) return;
    std::vector<VkViewport> v = c.viewports;
    if (jitter)
        for (auto& vp : v) { vp.x += c.jx; vp.y += c.jy; }
    if (c.viewportWithCount)
        NEXT(&d, vkCmdSetViewportWithCount)(cb, uint32_t(v.size()), v.data());
    else
        NEXT(&d, vkCmdSetViewport)(cb, c.firstViewport, uint32_t(v.size()), v.data());
}

static void beginPass(Device& d, VkCommandBuffer cb, const std::vector<VkImageView>& colors, VkImageView depth)
{
    Command& c = command(d, cb);
    c.jittered = false;
    c.mainPass = false;
    if (!d.engineOn.load(std::memory_order_relaxed) || !depth) return;
    View dv;
    if (!d.views.get(id(depth), dv)) return;
    const uint64_t depthImage = id(dv.image);
    bool main = d.mainDepthImages.contains(depthImage);
    for (VkImageView v : colors)
    {
        if (main) break;
        View cv;
        if (!d.views.get(id(v), cv) || !d.gbufferImages.contains(id(cv.image))) continue;
        main = true;
        GuardLock l;
        if (!d.mainDepthImages.contains(depthImage))
        {
            d.mainDepthImages.put(depthImage, true);
            d.censusBudget = 40;
            event("main_depth", "\"image\":" + std::to_string(depthImage) + ",\"via\":\"gbuffer_color_attachment\"");
        }
    }
    if (!main) return;
    // A main-scene pass: a few per frame, so the bookkeeping takes guard.
    GuardLock l;
    if (!d.engine) return;
    c.mainPass = true;
    ++d.mainPasses;
    if (d.censusBudget > 0)
    {
        // Which passes count as main-scene passes (all get jittered in replace mode).
        --d.censusBudget;
        auto describe = [&](uint64_t image) {
            Image info;
            return d.images.get(image, info) ? "[" + std::to_string(info.format) + "," + std::to_string(info.extent.width) + "," +
                                                   std::to_string(info.extent.height) + "]"
                                             : std::string("[0,0,0]");
        };
        std::string colorList;
        for (VkImageView v : colors)
        {
            View cv;
            colorList += (colorList.empty() ? "" : ",") + (d.views.get(id(v), cv) ? describe(id(cv.image)) : std::string("[0,0,0]"));
        }
        c.census    = "\"recording\":" + std::to_string(c.recording) + ",\"colors\":[" + colorList + "],\"depth\":" + describe(depthImage);
        c.passBinds = 0;
    }
    if (!d.engine->jitter(c.jx, c.jy)) return;
    c.jittered = true;
    d.engine->noteJitterApplied();
    const uint64_t passes = ++d.jitteredPasses;
    if (passes <= 3 || passes % 2000 == 0)
        event("jitter_pass", "\"count\":" + std::to_string(passes) + ",\"viewport_known\":" + (c.viewportKnown ? "true" : "false") +
                                 ",\"static_viewport_binds\":" + std::to_string(d.staticViewportBinds.load()));
    setViewports(d, cb, c, true);
}

static void endPass(Device& d, VkCommandBuffer cb)
{
    Command& c = command(d, cb);
    if (c.jittered) setViewports(d, cb, c, false);
    if (!c.census.empty()) event("main_pass_census", c.census + ",\"pipeline_binds\":" + std::to_string(c.passBinds));
    c.census.clear();
    c.jittered = false;
    c.mainPass = false;
}

static std::vector<VkImageView> passViews(Device& d, const VkRenderPassBeginInfo* bi)
{
    for (auto n = static_cast<const VkBaseInStructure*>(bi->pNext); n; n = n->pNext)
        if (n->sType == VK_STRUCTURE_TYPE_RENDER_PASS_ATTACHMENT_BEGIN_INFO)
        {
            auto a = reinterpret_cast<const VkRenderPassAttachmentBeginInfo*>(n);
            return {a->pAttachments, a->pAttachments + a->attachmentCount};
        }
    std::vector<VkImageView> views;
    d.framebuffers.get(id(bi->framebuffer), views);
    return views;
}

static void beginClassicPass(Device& d, VkCommandBuffer cb, const VkRenderPassBeginInfo* bi)
{
    std::vector<VkImageView> colors;
    VkImageView              depth = VK_NULL_HANDLE;
    if (d.engineOn.load(std::memory_order_relaxed))
    {
        RenderPassInfo rp;
        if (!d.renderPasses.get(id(bi->renderPass), rp)) return;
        auto views = passViews(d, bi);
        for (uint32_t i : rp.colors) if (i < views.size()) colors.push_back(views[i]);
        for (uint32_t i : rp.depths) if (i < views.size()) depth = views[i];
    }
    beginPass(d, cb, colors, depth);
}

template <class Info> static RenderPassInfo renderPassInfo(const Info* ci)
{
    RenderPassInfo r;
    for (uint32_t s = 0; s < ci->subpassCount; ++s)
    {
        const auto& sp = ci->pSubpasses[s];
        for (uint32_t k = 0; k < sp.colorAttachmentCount; ++k)
            if (sp.pColorAttachments[k].attachment != VK_ATTACHMENT_UNUSED) r.colors.push_back(sp.pColorAttachments[k].attachment);
        if (sp.pDepthStencilAttachment && sp.pDepthStencilAttachment->attachment != VK_ATTACHMENT_UNUSED)
            r.depths.push_back(sp.pDepthStencilAttachment->attachment);
    }
    return r;
}

EXPORT VkResult VKAPI_CALL vkCreateRenderPass(VkDevice h, const VkRenderPassCreateInfo* ci, const VkAllocationCallbacks* a, VkRenderPass* out)
{
    HOOK_SCOPE(vkCreateRenderPass); auto d = device(h);
    auto r = NEXT(d, vkCreateRenderPass)(h, ci, a, out);
    if (r == VK_SUCCESS) d->renderPasses.put(id(*out), renderPassInfo(ci));
    return r;
}
#define RENDER_PASS_2(name)                                                                                          \
    EXPORT VkResult VKAPI_CALL name(VkDevice h, const VkRenderPassCreateInfo2* ci, const VkAllocationCallbacks* a, VkRenderPass* out) \
    {                                                                                                                \
        HOOK_SCOPE(name); auto d = device(h);                                                                        \
        auto r = NEXT(d, name)(h, ci, a, out);                                                                       \
        if (r == VK_SUCCESS) d->renderPasses.put(id(*out), renderPassInfo(ci));                                      \
        return r;                                                                                                    \
    }
RENDER_PASS_2(vkCreateRenderPass2)
RENDER_PASS_2(vkCreateRenderPass2KHR)
#undef RENDER_PASS_2
EXPORT void VKAPI_CALL vkDestroyRenderPass(VkDevice h, VkRenderPass rp, const VkAllocationCallbacks* a)
{
    HOOK_SCOPE(vkDestroyRenderPass); auto d = device(h);
    d->renderPasses.erase(id(rp));
    NEXT(d, vkDestroyRenderPass)(h, rp, a);
}
EXPORT VkResult VKAPI_CALL vkCreateFramebuffer(VkDevice h, const VkFramebufferCreateInfo* ci, const VkAllocationCallbacks* a, VkFramebuffer* out)
{
    HOOK_SCOPE(vkCreateFramebuffer); auto d = device(h);
    auto r = NEXT(d, vkCreateFramebuffer)(h, ci, a, out);
    if (r == VK_SUCCESS)
        d->framebuffers.put(id(*out), (ci->flags & VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT) || !ci->pAttachments
                                          ? std::vector<VkImageView>{}
                                          : std::vector<VkImageView>(ci->pAttachments, ci->pAttachments + ci->attachmentCount));
    return r;
}
EXPORT void VKAPI_CALL vkDestroyFramebuffer(VkDevice h, VkFramebuffer fb, const VkAllocationCallbacks* a)
{
    HOOK_SCOPE(vkDestroyFramebuffer); auto d = device(h);
    d->framebuffers.erase(id(fb));
    NEXT(d, vkDestroyFramebuffer)(h, fb, a);
}
EXPORT VkResult VKAPI_CALL vkCreateGraphicsPipelines(VkDevice h, VkPipelineCache cache, uint32_t n, const VkGraphicsPipelineCreateInfo* ci,
                                                     const VkAllocationCallbacks* a, VkPipeline* out)
{
    HOOK_SCOPE(vkCreateGraphicsPipelines); auto d = device(h);
    auto r = NEXT(d, vkCreateGraphicsPipelines)(h, cache, n, ci, a, out);
    for (uint32_t i = 0; i < n; ++i)
    {
        if (!out[i]) continue;
        bool dynamic = false;
        if (ci[i].pDynamicState)
            for (uint32_t k = 0; k < ci[i].pDynamicState->dynamicStateCount; ++k)
            {
                auto st = ci[i].pDynamicState->pDynamicStates[k];
                dynamic |= st == VK_DYNAMIC_STATE_VIEWPORT || st == VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT;
            }
        if (!dynamic) d->staticViewport.put(id(out[i]), true);
    }
    return r;
}
EXPORT void VKAPI_CALL vkCmdBeginRenderPass(VkCommandBuffer cb, const VkRenderPassBeginInfo* bi, VkSubpassContents contents)
{
    HOOK_SCOPE(vkCmdBeginRenderPass); auto d = device(cb);
    NEXT(d, vkCmdBeginRenderPass)(cb, bi, contents);
    beginClassicPass(*d, cb, bi);
}
#define BEGIN_PASS_2(name)                                                                                       \
    EXPORT void VKAPI_CALL name(VkCommandBuffer cb, const VkRenderPassBeginInfo* bi, const VkSubpassBeginInfo* si) \
    {                                                                                                            \
        HOOK_SCOPE(name); auto d = device(cb);                                                                   \
        NEXT(d, name)(cb, bi, si);                                                                               \
        beginClassicPass(*d, cb, bi);                                                                            \
    }
BEGIN_PASS_2(vkCmdBeginRenderPass2)
BEGIN_PASS_2(vkCmdBeginRenderPass2KHR)
#undef BEGIN_PASS_2
EXPORT void VKAPI_CALL vkCmdEndRenderPass(VkCommandBuffer cb)
{
    HOOK_SCOPE(vkCmdEndRenderPass); auto d = device(cb);
    endPass(*d, cb);
    NEXT(d, vkCmdEndRenderPass)(cb);
}
#define END_PASS_2(name)                                                       \
    EXPORT void VKAPI_CALL name(VkCommandBuffer cb, const VkSubpassEndInfo* ei) \
    {                                                                          \
        HOOK_SCOPE(name); auto d = device(cb);                                 \
        endPass(*d, cb);                                                       \
        NEXT(d, name)(cb, ei);                                                 \
    }
END_PASS_2(vkCmdEndRenderPass2)
END_PASS_2(vkCmdEndRenderPass2KHR)
#undef END_PASS_2
#define BEGIN_RENDERING(name)                                                              \
    EXPORT void VKAPI_CALL name(VkCommandBuffer cb, const VkRenderingInfo* ri)             \
    {                                                                                      \
        HOOK_SCOPE(name); auto d = device(cb);                                             \
        NEXT(d, name)(cb, ri);                                                             \
        std::vector<VkImageView> colors;                                                   \
        for (uint32_t k = 0; k < ri->colorAttachmentCount; ++k)                            \
            colors.push_back(ri->pColorAttachments[k].imageView);                          \
        beginPass(*d, cb, colors, ri->pDepthAttachment ? ri->pDepthAttachment->imageView : VK_NULL_HANDLE); \
    }
BEGIN_RENDERING(vkCmdBeginRendering)
BEGIN_RENDERING(vkCmdBeginRenderingKHR)
#undef BEGIN_RENDERING
#define END_RENDERING(name)                                \
    EXPORT void VKAPI_CALL name(VkCommandBuffer cb)        \
    {                                                      \
        HOOK_SCOPE(name); auto d = device(cb);             \
        endPass(*d, cb);                                   \
        NEXT(d, name)(cb);                                 \
    }
END_RENDERING(vkCmdEndRendering)
END_RENDERING(vkCmdEndRenderingKHR)
#undef END_RENDERING
EXPORT void VKAPI_CALL vkCmdSetViewport(VkCommandBuffer cb, uint32_t first, uint32_t count, const VkViewport* v)
{
    HOOK_SCOPE(vkCmdSetViewport); auto d = device(cb);
    Command& c = command(*d, cb);
    c.viewports.assign(v, v + count);
    c.firstViewport     = first;
    c.viewportKnown     = true;
    c.viewportWithCount = false;
    if (c.mainPass) ++d->mainViewportSets;
    if (c.jittered) setViewports(*d, cb, c, true);
    else NEXT(d, vkCmdSetViewport)(cb, first, count, v);
}
#define SET_VIEWPORT_WITH_COUNT(name)                                                       \
    EXPORT void VKAPI_CALL name(VkCommandBuffer cb, uint32_t count, const VkViewport* v)    \
    {                                                                                       \
        HOOK_SCOPE(name); auto d = device(cb);                                              \
        Command& c = command(*d, cb);                                                       \
        c.viewports.assign(v, v + count);                                                   \
        c.firstViewport     = 0;                                                            \
        c.viewportKnown     = true;                                                         \
        c.viewportWithCount = true;                                                         \
        if (c.mainPass) ++d->mainViewportSets;                                              \
        if (c.jittered) setViewports(*d, cb, c, true);                                      \
        else NEXT(d, name)(cb, count, v);                                                   \
    }
SET_VIEWPORT_WITH_COUNT(vkCmdSetViewportWithCount)
SET_VIEWPORT_WITH_COUNT(vkCmdSetViewportWithCountEXT)
#undef SET_VIEWPORT_WITH_COUNT

// Binding paths this layer does not decode: forget the command buffer's compute bindings, so a later
// dispatch is never handled from stale state (the engine then leaves EASU unchanged). Render-pass
// and viewport state stays: clearing it inside a jittered pass would leave the jittered viewport set
// after the pass.
static void forgetBindings(Device& d, VkCommandBuffer cb)
{
    Command& c = command(d, cb);
    c.known    = false;
    c.pipeline = VK_NULL_HANDLE;
    c.role     = ShaderRole::None;
    c.bound    = {};
}
#define UNTRACKED(name, signature, arguments)                         \
    EXPORT void VKAPI_CALL name signature                             \
    {                                                                 \
        auto d = device(cb);                                          \
        NEXT(d, name) arguments;                                      \
        forgetBindings(*d, cb);                                       \
    }
UNTRACKED(vkCmdBindDescriptorSets2, (VkCommandBuffer cb, const VkBindDescriptorSetsInfo* i), (cb, i))
UNTRACKED(vkCmdBindDescriptorSets2KHR, (VkCommandBuffer cb, const VkBindDescriptorSetsInfo* i), (cb, i))
UNTRACKED(vkCmdPushDescriptorSet, (VkCommandBuffer cb, VkPipelineBindPoint p, VkPipelineLayout l, uint32_t s, uint32_t n, const VkWriteDescriptorSet* w),
          (cb, p, l, s, n, w))
UNTRACKED(vkCmdPushDescriptorSetKHR, (VkCommandBuffer cb, VkPipelineBindPoint p, VkPipelineLayout l, uint32_t s, uint32_t n, const VkWriteDescriptorSet* w),
          (cb, p, l, s, n, w))
UNTRACKED(vkCmdPushDescriptorSetWithTemplate, (VkCommandBuffer cb, VkDescriptorUpdateTemplate t, VkPipelineLayout l, uint32_t s, const void* x),
          (cb, t, l, s, x))
UNTRACKED(vkCmdPushDescriptorSetWithTemplateKHR, (VkCommandBuffer cb, VkDescriptorUpdateTemplate t, VkPipelineLayout l, uint32_t s, const void* x),
          (cb, t, l, s, x))
UNTRACKED(vkCmdBindDescriptorBuffersEXT, (VkCommandBuffer cb, uint32_t n, const VkDescriptorBufferBindingInfoEXT* i), (cb, n, i))
UNTRACKED(vkCmdSetDescriptorBufferOffsetsEXT,
          (VkCommandBuffer cb, VkPipelineBindPoint p, VkPipelineLayout l, uint32_t f, uint32_t n, const uint32_t* i, const VkDeviceSize* o),
          (cb, p, l, f, n, i, o))
UNTRACKED(vkCmdBindShadersEXT, (VkCommandBuffer cb, uint32_t n, const VkShaderStageFlagBits* s, const VkShaderEXT* x), (cb, n, s, x))
#undef UNTRACKED

namespace {
PFN_vkVoidFunction local(const char* name)
{
#define MATCH(f) \
    if (!std::strcmp(name, #f)) return reinterpret_cast<PFN_vkVoidFunction>(f)
    MATCH(vkGetInstanceProcAddr); MATCH(vkGetDeviceProcAddr); MATCH(vkCreateInstance); MATCH(vkDestroyInstance);
    MATCH(vkCreateDevice); MATCH(vkDestroyDevice);
    if (mode() == xpfsr::Mode::Passthrough)
    {
        if (timingOnly()) MATCH(vkQueuePresentKHR);  // frame timing only
        return nullptr;                              // otherwise fully transparent
    }
    MATCH(vkCreateImage); MATCH(vkDestroyImage); MATCH(vkCreateImageView); MATCH(vkDestroyImageView);
    MATCH(vkCreateBuffer); MATCH(vkDestroyBuffer); MATCH(vkCreateShaderModule); MATCH(vkDestroyShaderModule);
    MATCH(vkCreateComputePipelines); MATCH(vkDestroyPipeline); MATCH(vkCreatePipelineLayout); MATCH(vkDestroyPipelineLayout);
    MATCH(vkCreateDescriptorSetLayout); MATCH(vkDestroyDescriptorSetLayout); MATCH(vkAllocateDescriptorSets);
    MATCH(vkFreeDescriptorSets); MATCH(vkResetDescriptorPool); MATCH(vkDestroyDescriptorPool); MATCH(vkUpdateDescriptorSets);
    MATCH(vkCreateDescriptorUpdateTemplate); MATCH(vkDestroyDescriptorUpdateTemplate); MATCH(vkUpdateDescriptorSetWithTemplate);
    MATCH(vkCreateDescriptorUpdateTemplateKHR); MATCH(vkDestroyDescriptorUpdateTemplateKHR); MATCH(vkUpdateDescriptorSetWithTemplateKHR);
    MATCH(vkAllocateCommandBuffers); MATCH(vkFreeCommandBuffers); MATCH(vkDestroyCommandPool); MATCH(vkResetCommandPool);
    MATCH(vkBeginCommandBuffer); MATCH(vkResetCommandBuffer); MATCH(vkCmdBindPipeline); MATCH(vkCmdBindDescriptorSets);
    MATCH(vkCmdDispatch); MATCH(vkQueuePresentKHR); MATCH(vkQueueSubmit); MATCH(vkQueueSubmit2); MATCH(vkQueueSubmit2KHR);
    MATCH(vkCmdBindDescriptorSets2); MATCH(vkCmdBindDescriptorSets2KHR); MATCH(vkCmdPushDescriptorSet); MATCH(vkCmdPushDescriptorSetKHR);
    MATCH(vkCmdPushDescriptorSetWithTemplate); MATCH(vkCmdPushDescriptorSetWithTemplateKHR); MATCH(vkCmdBindDescriptorBuffersEXT);
    MATCH(vkCmdSetDescriptorBufferOffsetsEXT); MATCH(vkCmdBindShadersEXT);
    MATCH(vkCreateRenderPass); MATCH(vkCreateRenderPass2); MATCH(vkCreateRenderPass2KHR); MATCH(vkDestroyRenderPass);
    MATCH(vkCreateFramebuffer); MATCH(vkDestroyFramebuffer); MATCH(vkCreateGraphicsPipelines);
    MATCH(vkCmdBeginRenderPass); MATCH(vkCmdBeginRenderPass2); MATCH(vkCmdBeginRenderPass2KHR);
    MATCH(vkCmdEndRenderPass); MATCH(vkCmdEndRenderPass2); MATCH(vkCmdEndRenderPass2KHR);
    MATCH(vkCmdBeginRendering); MATCH(vkCmdBeginRenderingKHR); MATCH(vkCmdEndRendering); MATCH(vkCmdEndRenderingKHR);
    MATCH(vkCmdSetViewport); MATCH(vkCmdSetViewportWithCount); MATCH(vkCmdSetViewportWithCountEXT);
#undef MATCH
    return nullptr;
}
PFN_vkVoidFunction VKAPI_CALL physicalProc(VkInstance h, const char* name)
{
    if (!h) return nullptr;
    auto i = instance(h);
    return i->gpdpa ? i->gpdpa(h, name) : i->gipa(h, name);
}
}  // namespace

EXPORT PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance h, const char* name)
{
    if (!name) return nullptr;
    if (!std::strcmp(name, "vkCreateInstance") || !std::strcmp(name, "vkGetInstanceProcAddr") || !std::strcmp(name, "vkCreateDevice") ||
        !std::strcmp(name, "vkGetDeviceProcAddr"))
        return local(name);
    if (!h) return nullptr;
    auto i    = instance(h);
    auto next = i->gipa(h, name);
    if (!next) return nullptr;
    auto p = local(name);
    return p ? p : next;
}

EXPORT PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice h, const char* name)
{
    if (!h || !name) return nullptr;
    if (!std::strcmp(name, "vkGetDeviceProcAddr")) return local(name);
    auto d    = device(h);
    auto next = d->gdpa(h, name);
    if (!next) return nullptr;
    auto p = local(name);
    return p ? p : next;
}

EXPORT VkResult VKAPI_CALL vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface* v)
{
    if (!v || v->loaderLayerInterfaceVersion < 2) return VK_ERROR_INITIALIZATION_FAILED;
    v->loaderLayerInterfaceVersion = 2;
    v->pfnGetInstanceProcAddr      = vkGetInstanceProcAddr;
    v->pfnGetDeviceProcAddr        = vkGetDeviceProcAddr;
    v->pfnGetPhysicalDeviceProcAddr = physicalProc;
    return VK_SUCCESS;
}
