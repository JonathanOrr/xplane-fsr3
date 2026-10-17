#!/usr/bin/env python3
"""Fails if the layer needs more from a user's system than X-Plane 12 itself does, or exports more
than the Vulkan entry points. Usage: check_portable.py <libVkLayer_xplane_fsr3.so>"""
import re
import subprocess
import sys

lib = sys.argv[1]
dyn = subprocess.run(['objdump', '-T', lib], capture_output=True, text=True, check=True).stdout
needed = re.findall(r'NEEDED\s+(\S+)', subprocess.run(['objdump', '-p', lib], capture_output=True, text=True, check=True).stdout)
limits = {'GLIBC': (2, 34), 'GLIBCXX': (3, 4, 29), 'CXXABI': (1, 3, 13)}
failures = []
for name, version in sorted(set(re.findall(r'\b(GLIBC|GLIBCXX|CXXABI)_([0-9.]+)', dyn))):
    if tuple(map(int, version.split('.'))) > limits[name]:
        users = sorted({line.split()[-1] for line in dyn.splitlines() if f'{name}_{version})' in line or f'{name}_{version} ' in line})
        failures.append(f'needs {name}_{version} (limit {".".join(map(str, limits[name]))}): {", ".join(users)}')
allowed = {'libstdc++.so.6', 'libm.so.6', 'libgcc_s.so.1', 'libc.so.6', 'ld-linux-x86-64.so.2'}
failures += [f'links {n}' for n in needed if n not in allowed]
exports = [l.split()[-1] for l in dyn.splitlines() if re.search(r'\sg\s+DF\s+\.text', l)]
failures += [f'exports {e}' for e in exports if not e.startswith('vk')]
print(f'{lib}: {len(exports)} exports, needs {", ".join(needed)}')
for f in failures:
    print('FAIL', f)
print('PASS' if not failures else f'{len(failures)} failure(s)')
sys.exit(1 if failures else 0)
