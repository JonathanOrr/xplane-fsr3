#!/usr/bin/env python3
"""Runtime shader recognition (src/shader_classify.cpp) against the reference classifier
(tools/classify_shaders.py) over every SPIR-V module in an installed X-Plane, plus tampered copies of
the recognised shaders that must be rejected.

Usage: run_classify.py <build dir> <X-Plane 12 dir>. Exit 77 (skipped) without an X-Plane install.
"""
import struct
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

build, xplane = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import classify_shaders as reference  # noqa: E402

spv = xplane / 'Resources' / 'shaders' / 'bin' / 'spv'
archives = sorted(spv.glob('*.xsa'))
if not archives:
    print('no X-Plane shader archives: skipped')
    sys.exit(77)

from spirv_tamper import tamper_binding, tamper_offset  # noqa: E402


failures = []
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    files, expected = [], {}
    for archive in archives:
        with zipfile.ZipFile(archive) as package:
            for name in package.namelist():
                if not name.endswith('.spv'):
                    continue
                data = package.read(name)
                path = tmp / f'{archive.stem}__{name.replace("/", "_")}'
                path.write_bytes(data)
                resources, gathers = reference.parse(data)
                expected[str(path)] = reference.role(resources, gathers) or 'None'
                files.append(path)
    # Tampered copies: each changes one thing the layer's shaders depend on.
    easu = next(p for p in files if expected[str(p)] == 'Easu')
    ssr = next(p for p in files if expected[str(p)] == 'SsrSceneArray')
    tampered = {
        'easu_offset': (tamper_offset(easu.read_bytes(), 'u_fsr_data', 2, 16), 'easu_u_fsr_data_layout'),
        'easu_binding': (tamper_binding(easu.read_bytes(), 3, 5), 'easu_bindings'),
        'ssr_reproj_offset': (tamper_offset(ssr.read_bytes(), 'u_ssr_data', 1, 64), 'ssr_u_ssr_data_layout'),
        'ssr_clip_offset': (tamper_offset(ssr.read_bytes(), 'u_gbuffer_data', 3, 16), 'ssr_u_gbuffer_data_layout'),
        'ssr_binding': (tamper_binding(ssr.read_bytes(), 12, 13), 'ssr_bindings'),
    }
    for name, (data, _) in tampered.items():
        (tmp / f'tampered__{name}.spv').write_bytes(data)
    paths = [str(p) for p in files] + [str(tmp / f'tampered__{n}.spv') for n in tampered]
    out = subprocess.run([str(build / 'classify_check')] + paths, capture_output=True, text=True, check=True).stdout.splitlines()
    got = {}
    for line in out[:-1]:
        path, role, reason = line.rsplit(' ', 2)
        got[path] = (role, reason)
    roles = {}
    for path in map(str, files):
        role, reason = got[path]
        roles[role] = roles.get(role, 0) + 1
        if role != expected[path]:
            failures.append(f'{Path(path).name}: runtime {role} ({reason}), reference {expected[path]}')
        elif role == 'None' and reason != '-':
            failures.append(f'{Path(path).name}: rejected ({reason}) though the reference finds no role either')
    for name, (_, why) in tampered.items():
        role, reason = got[str(tmp / f'tampered__{name}.spv')]
        if role != 'None' or reason != why:
            failures.append(f'tampered {name}: {role} ({reason}), expected None ({why})')
    print(f'{len(files)} modules from {len(archives)} archives: {roles}')
    print(out[-1])
    if roles.get('Easu', 0) == 0 or roles.get('SsrScene2D', 0) + roles.get('SsrSceneArray', 0) == 0:
        failures.append('no EASU or no scene SSR recognised')

for f in failures:
    print('FAIL', f)
print('PASS' if not failures else f'{len(failures)} failure(s)')
sys.exit(1 if failures else 0)
