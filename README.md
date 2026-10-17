# FSR 3.1 for X-Plane 12 (Linux)

X-Plane 12 can render at a lower resolution and upscale with AMD FSR 1. This mod replaces that
upscaler with **AMD FSR 3.1**, which builds each frame from several previous ones. Gauges, text and
fine detail come out much sharper than with FSR 1, sometimes sharper than native resolution.

It is a Vulkan layer, started with X-Plane by a small launcher. X-Plane's files are not modified,
and X-Plane's own "Rendering Resolution (FSR Supersampling)" setting still chooses the render
resolution. If anything the mod depends on is missing or different, it steps aside and X-Plane's
FSR 1 runs as usual.

## Using it

Download the `xplane-fsr3-<version>.tar.gz` release and follow the README inside
([`fsr3-layer/package/README.md`](fsr3-layer/package/README.md)). In short:
1. Extract it into your home folder or the X-Plane 12 folder.
2. Set X-Plane 12's Steam launch options to `"/path/to/xplane-fsr3/xplane-fsr3" %command%`.
   Without Steam, start X-Plane through the launcher instead.
3. In X-Plane, set *Rendering Resolution (FSR Supersampling)* to anything but Off.

`logs/status.txt` in the mod's folder tells you whether FSR 3.1 is active.

**Requirements:** Linux and X-Plane 12 (Vulkan). Not for VR: in VR, X-Plane's FSR 1 is used.

**Tested on:** X-Plane 12.4.4-b4 with an AMD Radeon RX 9070 XT (Mesa RADV), output 5504×2166.
Every frame was upscaled by FSR 3.1, including while the quality setting was changed in flight.

| X-Plane setting | Render resolution | fps | FSR 3.1 GPU time |
|---|---|---|---|
| 1.5× | 3669×1444 | 80–82 | 1.8 ms |
| 1.3× | 4233×1666 | 69–70 | 2.1 ms |
| 2× | 2752×1083 | 94–98 | 1.4 ms |

At the 1.3× setting, X-Plane's own FSR 1 ran at 78–79 fps.

Other GPUs, distributions and X-Plane versions have not been tested yet. Reports are welcome:
include `logs/status.txt` and the newest `logs/fsr3-*.jsonl`.

## How it works

The layer watches X-Plane's Vulkan calls and recognises two of its compute passes by what they
bind, not by an exact copy of X-Plane's shaders:
- **Scene screen-space reflections:** the layer reads the depth buffer and camera matrices this pass
  already uses, and turns them into the depth and motion vectors FSR 3.1 needs.
- **FSR 1 upscale (EASU):** the layer runs FSR 3.1 in its place and writes its output where EASU
  would have.

FSR 3.1 also needs the camera to shift by a fraction of a pixel each frame. The layer adds that
jitter to the viewport of X-Plane's main scene passes. X-Plane's own sharpening pass still runs
afterwards.

Details are in [`fsr3-layer/README.md`](fsr3-layer/README.md). AMD's SDK ships for Windows only, so
[`fsr3-linux/`](fsr3-linux/PORTING.md) builds its FSR 3.1 upscaler and Vulkan backend for Linux.

## Building from source

You need Linux, git, CMake 3.20+, GCC with C++17, the Vulkan headers and loader, glslangValidator
and Python 3. The tests also need Mesa's Lavapipe (`mesa-vulkan-drivers`).

```bash
scripts/fetch-sdk.sh            # AMD FidelityFX SDK v1.1.4 + this project's patches, into third_party/
cd fsr3-layer
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DXPLANE_ROOT="/path/to/X-Plane 12" \
      -DFSR3_TEST_GPU=9070
cmake --build build -j
ctest --test-dir build --output-on-failure
./package.sh                    # dist/xplane-fsr3-<version>.tar.gz
python3 tests/run_package.py build dist/xplane-fsr3-<version>.tar.gz llvmpipe
```

`XPLANE_ROOT` is needed only for the tests, which read X-Plane's installed shaders. The layer itself
builds without X-Plane. `FSR3_TEST_GPU` (optional) is part of a Vulkan device name, used to run the
tests on that GPU as well as on Lavapipe.

| Folder | Contents |
|---|---|
| `fsr3-layer/` | The Vulkan layer, its shaders, the launcher and user README (`package/`), tests and tools. |
| `fsr3-linux/` | Linux build of AMD's FSR 3.1 upscaler, its patches and a self-test against ground truth. |
| `scripts/` | `fetch-sdk.sh`. |

## Limitations

- **Trails:** fast gauge needles and spinning propellers can leave faint trails. The layer has no
  per-object motion vectors or reactive mask, only the camera's motion.
- **VR:** not supported; X-Plane's FSR 1 is used.
- **Frame generation:** not attempted.

## Licence

MIT, see [`LICENSE`](LICENSE). The FSR 3.1 code is AMD's FidelityFX SDK, also MIT; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). X-Plane is a trademark of Laminar Research.
This project is not affiliated with or endorsed by Laminar Research or AMD.
