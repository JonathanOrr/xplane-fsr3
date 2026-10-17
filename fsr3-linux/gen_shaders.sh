#!/usr/bin/env bash
# Generates the FSR 3.1 upscaler's Vulkan shader permutation headers on Linux.
# Mirrors sdk/include/FidelityFX/gpu/fsr3upscaler/CMakeCompileFSR3UpscalerShaders.txt
# and sdk/include/FidelityFX/gpu/CMakeCompileShaders.txt (FidelityFX SDK v1.1.4)
# for the GLSL path: four header sets per pass (wave32, wave64, 16-bit, wave64 16-bit).
# Usage: gen_shaders.sh <FidelityFX_SC> <sdk dir> <output dir> [pass name filter]
set -euo pipefail
SC=$1; SDK=$2; OUT=$3; FILTER=${4:-}
GPU=$SDK/include/FidelityFX/gpu
mkdir -p "$OUT"

BASE=(-reflection -deps=gcc -DFFX_GPU=1
      -DFFX_FSR3UPSCALER_OPTION_UPSAMPLE_SAMPLERS_USE_DATA_HALF=0
      -DFFX_FSR3UPSCALER_OPTION_ACCUMULATE_SAMPLERS_USE_DATA_HALF=0
      -DFFX_FSR3UPSCALER_OPTION_REPROJECT_SAMPLERS_USE_DATA_HALF=1
      -DFFX_FSR3UPSCALER_OPTION_POSTPROCESSLOCKSTATUS_SAMPLERS_USE_DATA_HALF=0
      -DFFX_FSR3UPSCALER_OPTION_UPSAMPLE_USE_LANCZOS_TYPE=2)
API=(-compiler=glslang -e CS --target-env vulkan1.2 -S comp -Os -DFFX_GLSL=1)
PERM=('-DFFX_FSR3UPSCALER_OPTION_REPROJECT_USE_LANCZOS_TYPE={0,1}'
      '-DFFX_FSR3UPSCALER_OPTION_HDR_COLOR_INPUT={0,1}'
      '-DFFX_FSR3UPSCALER_OPTION_LOW_RESOLUTION_MOTION_VECTORS={0,1}'
      '-DFFX_FSR3UPSCALER_OPTION_JITTERED_MOTION_VECTORS={0,1}'
      '-DFFX_FSR3UPSCALER_OPTION_INVERTED_DEPTH={0,1}'
      '-DFFX_FSR3UPSCALER_OPTION_APPLY_SHARPENING={0,1}')
INC=(-I"$GPU" -I"$GPU/fsr3upscaler")

for shader in "$SDK"/src/backends/vk/shaders/fsr3upscaler/*.glsl; do
    name=$(basename "$shader" .glsl)
    [[ -n "$FILTER" && "$name" != *"$FILTER"* ]] && continue
    for variant in ":0" "_wave64:0" "_16bit:1" "_wave64_16bit:1"; do
        suffix=${variant%%:*}; half=${variant##*:}
        "$SC" "${BASE[@]}" "${API[@]}" "${PERM[@]}" -name="${name}${suffix}" \
            -DFFX_HALF="$half" "${INC[@]}" -output="$OUT" "$shader"
    done
done
