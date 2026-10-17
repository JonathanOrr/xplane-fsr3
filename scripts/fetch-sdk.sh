#!/usr/bin/env bash
# Fetches AMD FidelityFX SDK v1.1.4, the last release with a Vulkan backend (SDK 2.x is DirectX 12
# only), into third_party/FidelityFX-SDK (its sdk/ folder only), then applies this project's
# patches: the Linux build fixes and the FSR 3.1.5 backport (see fsr3-linux/PORTING.md).
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
dest=$root/third_party/FidelityFX-SDK
commit=c6efa6bf7f2027b3ec94f28578bb5965eabb9e55
if [ -e "$dest" ]; then
    echo "$dest already exists; delete it to fetch again" >&2
    exit 1
fi
mkdir -p "$root/third_party"
git -c advice.detachedHead=false clone --quiet --depth 1 --branch v1.1.4 --filter=blob:none --sparse \
    https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK.git "$dest"
git -C "$dest" sparse-checkout set sdk
if [ "$(git -C "$dest" rev-parse HEAD)" != "$commit" ]; then
    echo "tag v1.1.4 is not commit $commit" >&2
    exit 1
fi
for patch in fidelityfx-sdk-v1.1.4-linux.patch fsr-3.1.5-backport.patch; do
    git -C "$dest" apply "$root/fsr3-linux/patches/$patch"
done
echo "FidelityFX SDK v1.1.4 fetched and patched in $dest"
