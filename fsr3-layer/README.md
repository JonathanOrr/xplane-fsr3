# VK_LAYER_XPLANE_fsr3

An explicit Vulkan layer that replaces X-Plane 12's FSR 1 EASU pass with AMD FSR 3.1 temporal
upscaling. X-Plane's files are never modified. User documentation: `package/README.md`.

## How it works

| Step | Where | What |
|---|---|---|
| Recognise passes | `src/shader_classify.cpp`, at `vkCreateShaderModule` | Classifies each SPIR-V module by its interface, so any X-Plane build with the same interface works.<br>• EASU: X-Plane's `fsr_1/4`.<br>• Scene SSR: `ssr_deferred_0,6,12,18` (2D) and `1,7,13,19` (array). Water and MSAA SSR are excluded.<br>A module is rejected, with a reason, if it has the names but not exactly what the layer's shaders read: bindings, set 0, image types, uniform member offsets, column-major matrices. `status.txt` then reports an unsupported X-Plane. `tools/classify_shaders.py` is the reference implementation. |
| Inputs | `shaders/inputs.comp`, after X-Plane's SSR dispatch | Reads, at the same stage and in the same layouts, X-Plane's own:<br>• `gbuffer_pos`: depth, 0 = sky;<br>• `gbuffer_vel`: bit 4 = own aircraft;<br>• `u_gbuffer_data` and `u_ssr_data`.<br>Writes layer-owned depth (near/distance) and motion vectors, `u_proj·(M·P)` with M = `u_local_reproj` or `u_reproj`, exactly as SSR reprojects. The previous frame's `u_proj` is used, so zooming (an FOV change) gets correct motion. |
| Jitter | `layer.cpp` viewport hooks | Adds the frame's Halton offset to `VkViewport.x/y` inside render passes on the main scene depth. That is the depth/stencil image SSR samples as `gbuffer_pos`/`gbuffer_vel`, or failing that, the depth of a pass with a G-buffer colour attachment. Not in observe mode. |
| Upscale | `engine.cpp`, at X-Plane's EASU dispatch | Copies EASU's input in, runs FSR 3.1 (`../fsr3-linux`), writes EASU's output as EASU would (alpha 1), skips EASU, and restores X-Plane's compute bindings. Any unmet precondition leaves EASU unchanged and logs why. X-Plane's own RCAS sharpening still follows. |
| Order check | `layer.cpp` `vkQueueSubmit`/`2` hooks | Inputs are numbered. Each SSR's inputs feed one FSR dispatch, which may sit in a later command buffer. At submission the layer checks that the writing buffer reached the queue first and that nothing newer overwrote the inputs. Results: `inputs_order` (`in_order`, `missing`, `overwritten`, `cross_queue`), with totals in `fsr_summary`. |
| Status | `layer.cpp` `writeStatus`, engine summary | `status.txt` in the log folder, in plain English: LOADED, WAITING, ACTIVE, PARTLY ACTIVE, NOT USED, UNSUPPORTED, STOPPED or OFF. Each entry covers at least 5 s and names the version and GPU. |

Environment variables (the launcher sets the first two):

| Variable | Effect |
|---|---|
| `XP_FSR3` | `replace`; `observe` (the whole pipeline into the layer's own images, X-Plane's picture unchanged, no jitter); `timing` (inert except for frame timing: a baseline). Anything else passes through. |
| `XP_FSR3_OUTPUT` | Log folder: `fsr3-<pid>.jsonl` and `status.txt`. A `DISABLED` file there makes the layer inert. |
| `XP_FSR3_AB=N` | Alternates replace with X-Plane's FSR 1 every N s. `ab_phase` logs fps and GPU ms per phase. |
| `XP_FSR3_PROFILE=1` | Logs the layer's CPU cost per hook, and lock waits (`cpu_profile`). |
| `XP_FSR3_FP16=1` | Enables shaderFloat16/Int16 so FSR 3.1 runs its 16-bit shaders. Off by default: on an RX 9070 XT it saved no GPU time (2.23 vs 2.24 ms at 4233x1666 to 5504x2166). |

GPU time of the inputs pass, FSR 3.1, and EASU when it runs unchanged is in `fsr_summary`.

## Build and test

```bash
../scripts/fetch-sdk.sh   # once
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DXPLANE_ROOT="/path/to/X-Plane 12" -DFSR3_TEST_GPU=9070
cmake --build build -j
ctest --test-dir build --output-on-failure
```

| Test | Checks |
|---|---|
| `shader_classify` | Compares the runtime classifier with `tools/classify_shaders.py` over every SPIR-V module in the installed X-Plane: 6,775 modules in 98 archives for 12.4.4-b4, classified in 18.8 ms in total. Five tampered copies (a member offset or binding changed) must be rejected with the right reason. |
| `portable` | `tests/check_portable.py`:<br>• glibc ≤ 2.34, X-Plane 12's own requirement;<br>• GLIBCXX ≤ 3.4.29 (GCC 11);<br>• no libraries beyond libc/libm/libstdc++/libgcc;<br>• only `vk*` exports. |
| `fsr3_selftest_*` | `../fsr3-linux`: FSR 3.1 against a supersampled ground truth. |
| `fake_xplane_*` | `tests/run_fake_xplane.py`: the layer driven by `tests/fake_xplane.cpp`, a small Vulkan program that records X-Plane's real SSR and EASU shaders (read from `XPLANE_ROOT`) the way X-Plane does. |

`fake_xplane` cases, 48 frames, 640x360 to 1280x720, Khronos validation on, 0 errors in every
case. Lavapipe shown; an RX 9070 XT matches except for a jittered panel score of 37.15 dB:

| case | moving wall | static cockpit panel |
|---|---|---|
| X-Plane's EASU | 13.62 dB | 32.62 dB |
| FSR 3.1, G-buffer uploaded (no jitter possible) | 15.71 | 28.20 (softer than EASU) |
| FSR 3.1, G-buffer rasterised, viewport jitter | **16.55** | **37.61** |
| the same two with SSR and EASU in separate command buffers | identical | identical |
| flipped reprojection (control) | 13.68 | unaffected |
| buffers submitted in the wrong order (control, Lavapipe only) | layer reports `missing` | |
| observe mode | output byte-identical to EASU | |
| `XP_FSR3_FP16=1` | within 0.5 dB of fp32 | |

Without jitter FSR 3.1 loses to EASU on static content; `../fsr3-linux` with `FSR_NO_JITTER=1`
shows the same on its own. That is why the viewport jitter matters.

`fake_xplane` options for measurement:
- `FAKE_SIZE=<rw>x<rh>:<dw>x<dh>`: render and display sizes.
- `--load N --load-threads T`: X-Plane-like descriptor churn on T worker threads.
- `FAKE_PANEL_ROW`: how much of the screen the cockpit covers.
- `FAKE_START_DELAY`: idle seconds before the first frame, like a loading screen.

## Threads and locking

X-Plane records on many worker threads (24 on a 12-core CPU). The hot paths take only sharded leaf
locks: 64 independently locked shards per table. A command buffer's state is used without a lock,
since Vulkan's external synchronisation already rules out concurrent use. Compute bindings are
decoded only at SSR and EASU dispatches. Per-thread caches serve the few templates and set layouts
every thread reads. `guard` covers only the engine, main-scene passes and present (see the comment
above `struct Device`).

Measured with `fake_xplane --raster ... --load 1230 --load-threads 24`, worker time per frame
(medians of 5):

| | worker time per frame |
|---|---|
| Without the layer | 72 µs |
| With the layer | 134 µs |
| An earlier single-lock design | 1704 µs |

A ThreadSanitizer build (`-fsanitize=thread`) of that run reports no races.

## User package

`./package.sh` builds `dist/xplane-fsr3-<version>/` and its `.tar.gz`. The version is
`project(... VERSION)` in CMakeLists.txt. Contents:
- the launcher (`package/xplane-fsr3`, POSIX sh);
- the user README;
- the stripped layer and its manifest;
- `LICENSE` and `THIRD_PARTY_NOTICES.md`.

The layer is built in `build-release/` without an X-Plane install, then checked by
`tests/check_portable.py`. If `dist/xplane-fsr3-<version>/` is in use as an install, its `logs/` is
kept.

What the launcher does:
- Sets `VK_ADD_LAYER_PATH` and `VK_INSTANCE_LAYERS`, adding to any existing values.
- Outside Steam's runtime, with a system Vulkan loader older than 1.3.211, also sets `VK_LAYER_PATH`
  to the package folder plus the standard layer folders. Older loaders ignore `VK_ADD_LAYER_PATH`;
  Steam's runtime ships 1.3.239.
- Keeps the 10 newest logs, in `logs/` or, if the folder is read-only, `$XDG_STATE_HOME/xplane-fsr3`.
- Never changes GPU selection or X-Plane's settings.

`python3 tests/run_package.py build dist/xplane-fsr3-<version>.tar.gz <device>` extracts the tarball
into a folder with a space in its name and runs `fake_xplane` through the launcher. It checks:
- the layer is active and writes `status.txt`;
- log pruning;
- the launcher environment, both for Steam's runtime and for an old system loader;
- a tampered EASU shader leaves X-Plane's output unchanged and reports `UNSUPPORTED`;
- `XP_FSR3=off`;
- a read-only install.

Why the C++ runtime is not linked statically: a current toolchain's `libstdc++.a` and `libgcc_eh.a`
need glibc 2.35-2.38, above X-Plane's 2.34. `std::atoi` is avoided for the same reason, because
glibc 2.38 headers turn it into `__isoc23_strtol`.

## Known limits

- **Trails:** there is no reactive or transparency mask. Clouds, particles, rain, glass and animated
  parts (needles, props, control surfaces) get only the camera's motion.
- **Moving objects:** AI traffic and moving scenery get world motion vectors.
- **Not supported:** VR (more than one view layer), MSAA SSR variants, and the wave64 FSR path (see
  `../fsr3-linux/PORTING.md`).
- **Frame generation:** not attempted.
