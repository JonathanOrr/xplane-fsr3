#!/usr/bin/env bash
# Builds the user package: dist/xplane-fsr3-<version>/ and dist/xplane-fsr3-<version>.tar.gz.
# The layer is built in build-release/ without an X-Plane install (it needs none), with the same
# flags as the tested build (RelWithDebInfo), then stripped. tests/run_package.py tests the result.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
build=$here/build-release
cmake -S "$here" -B "$build" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DXPLANE_ROOT= >/dev/null
cmake --build "$build" -j"$(nproc)" --target VkLayer_xplane_fsr3 >/dev/null
version=$(sed -n 's/.*"implementation_version": "\(.*\)",/\1/p' "$build/VkLayer_xplane_fsr3.json")
name=xplane-fsr3-$version
dist=$here/dist/$name
# The folder may be in use as an install (a launch option pointing here): keep its logs/.
kept=""
if [ -d "$dist/logs" ]; then kept=$(mktemp -d); mv "$dist/logs" "$kept/"; fi
rm -rf "$dist" "$dist.tar.gz"
mkdir -p "$dist"
install -m 755 "$here/package/xplane-fsr3" "$dist/xplane-fsr3"
install -m 644 "$here/package/README.md" "$dist/README.md"
strip --strip-unneeded -o "$dist/libVkLayer_xplane_fsr3.so" "$build/libVkLayer_xplane_fsr3.so"
chmod 755 "$dist/libVkLayer_xplane_fsr3.so"
sed "s|\"library_path\": \".*\"|\"library_path\": \"./libVkLayer_xplane_fsr3.so\"|" "$build/VkLayer_xplane_fsr3.json" > "$dist/VkLayer_xplane_fsr3.json"
install -m 644 "$here/../LICENSE" "$dist/LICENSE"
install -m 644 "$here/../THIRD_PARTY_NOTICES.md" "$dist/THIRD_PARTY_NOTICES.md"
python3 "$here/tests/check_portable.py" "$dist/libVkLayer_xplane_fsr3.so"
tar -C "$here/dist" --owner=0 --group=0 -czf "$dist.tar.gz" "$name"
if [ -n "$kept" ]; then mv "$kept/logs" "$dist/"; rmdir "$kept"; fi
ls -l "$dist" "$dist.tar.gz"
sha256sum "$dist.tar.gz"
