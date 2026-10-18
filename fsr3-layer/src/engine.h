// FSR 3.1 engine for the X-Plane layer: owns the upscaler, the layer's images and its compute
// passes, and records them into X-Plane's command buffers at two points:
//   onSsr  - after X-Plane's scene SSR dispatch: derive depth and motion vectors (inputs.comp).
//   onEasu - in place of X-Plane's FSR 1 EASU dispatch: copy the input in, run FSR 3.1, write the
//            result to EASU's output. Returns false (the caller then records EASU unchanged)
//            whenever a precondition fails; the reason is logged.
// The inputs onSsr writes are numbered (a sequence). X-Plane records SSR and EASU in different
// command buffers, so FSR reads the newest inputs no earlier EASU consumed, and onSubmitted checks,
// in submission order, that the buffer that wrote them reached the queue first and nothing newer
// overwrote them before the read.
// Every Vulkan call goes to the next layer's functions, never back through this layer.
#pragma once

#include <vulkan/vulkan.h>

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace xpfsr {

enum class Mode { Passthrough, Observe, Replace };

using Log = std::function<void(const std::string& event, const std::string& jsonFields)>;

struct DeviceSetup {
    VkInstance               instance = VK_NULL_HANDLE;
    VkPhysicalDevice         physical = VK_NULL_HANDLE;
    VkDevice                 device   = VK_NULL_HANDLE;
    PFN_vkGetInstanceProcAddr gipa    = nullptr;  // next layer
    PFN_vkGetDeviceProcAddr  gdpa     = nullptr;  // next layer
    std::vector<std::string> enabledExtensions;
    bool                     shaderFloat16 = false;
    bool                     shaderInt16   = false;
    Mode                     mode = Mode::Passthrough;
    int                      abSeconds = 0;  // XP_FSR3_AB: alternate FSR 3.1 / X-Plane's FSR 1 every N s (replace mode)
    Log                      log;
    // Plain-English state for users (status.txt), every 10 s while X-Plane renders.
    std::function<void(const std::string& text)> status;
};

// X-Plane's compute bindings at the intercepted dispatch, restored after the layer's own passes.
struct ComputeState {
    VkPipeline       pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout   = VK_NULL_HANDLE;
    struct Set {
        uint32_t              index;
        VkDescriptorSet       set;
        std::vector<uint32_t> dynamicOffsets;
    };
    std::vector<Set> sets;
};

struct SsrBinding {
    VkImageView            pos = VK_NULL_HANDLE, vel = VK_NULL_HANDLE;
    VkImageLayout          posLayout{}, velLayout{};
    VkDescriptorBufferInfo gbufferData{}, ssrData{};  // effective offsets (dynamic offset applied)
    VkExtent2D             extent{};                  // gbuffer_pos image extent
    bool                   array  = false;
    uint32_t               layers = 1;                // dispatch z: one per eye
};

struct EasuBinding {
    VkImageView            input = VK_NULL_HANDLE, output = VK_NULL_HANDLE;
    VkImageLayout          inputLayout{}, outputLayout{};
    VkExtent2D             inputExtent{}, outputExtent{};
    VkDescriptorBufferInfo constants{};  // u_fsr_data, effective offset
    uint32_t               layers = 1;   // dispatch z
};

// What one command buffer did to the layer's input images, in recording order.
struct InputsEvent {
    bool     write;  // true: onSsr wrote inputs `seq`; false: FSR read inputs `seq`
    uint64_t seq;
};

class Engine {
public:
    explicit Engine(const DeviceSetup& setup);
    ~Engine();
    Engine(const Engine&)            = delete;
    Engine& operator=(const Engine&) = delete;

    Mode mode() const { return setup_.mode; }
    // Returns the inputs sequence it recorded, or 0.
    uint64_t onSsr(VkCommandBuffer cmd, uint64_t recording, const SsrBinding& b, const ComputeState& restore);
    // readSeq: the inputs sequence FSR read (0 if FSR did not run). Returns true if EASU was replaced.
    bool onEasu(VkCommandBuffer cmd, uint64_t recording, const EasuBinding& b, const ComputeState& restore, uint64_t& readSeq);
    // Per submitted command buffer, in submission order (vkQueueSubmit/2), with its InputsEvents.
    void onSubmitted(VkQueue queue, uint64_t submitCall, uint64_t recording, const std::vector<InputsEvent>& events);
    void onPresent();
    // X-Plane recorded its MSAA scene SSR: MSAA is on, so there is no depth FSR 3.1 can use. The
    // EASU refusals that follow say "msaa", and status.txt tells the user to turn MSAA off.
    void onMsaaScene();

    // Sub-pixel camera jitter, in render pixels, for main-scene passes recorded now. True only while
    // FSR 3.1 is replacing EASU (it did within the last 2 frames), so nothing jitters with X-Plane's
    // FSR switched off, in its FSR 1 phase of an A/B, or while every EASU is refused (MSAA on); always
    // false outside Replace.
    bool jitter(float& x, float& y);
    // A main-scene pass of the current frame was recorded with jitter(); FSR then gets the same offset.
    void noteJitterApplied();
    // Around X-Plane's own EASU dispatch when it runs unchanged: GPU timestamps (gpu_easu_ms).
    void markEasu(VkCommandBuffer cmd, bool begin);

private:
    struct Impl;
    DeviceSetup           setup_;
    std::unique_ptr<Impl> impl_;
    std::mutex            lock_;
};

}  // namespace xpfsr
