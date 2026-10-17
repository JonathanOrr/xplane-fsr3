#!/usr/bin/env python3
"""Tests the user package as a user would run it: the tarball extracted into a folder with a space in
its name, X-Plane (here fake_xplane) started through the launcher with no Vulkan variables set.

Usage: run_package.py <build dir> <package .tar.gz> [device]   (device as FAKE_DEVICE, default 9070)
Exit 77 (skipped) if the device is absent.
"""
import json
import os
import stat
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

build, tarball = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
device = sys.argv[3] if len(sys.argv) > 3 else '9070'
sys.path.insert(0, str(Path(__file__).resolve().parent))
from spirv_tamper import tamper_offset  # noqa: E402

failures = []


def clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith(('VK_', 'XP_FSR3'))}
    env['FAKE_DEVICE'] = device
    env.update(extra)
    return env


def fake(out, launcher=None, easu=None, env=None):
    """fake_xplane, optionally through the launcher; returns (stdout result, output image bytes)."""
    out.mkdir(parents=True, exist_ok=True)
    cmd = [str(build / 'fake_xplane'), str(build / 'xp/ssr_deferred_1.spv'), str(easu or build / 'xp/fsr_4.spv'),
           '--frames', '30', '--out', str(out), '--raster', str(build / 'xp/gbuffer.vert.spv'), str(build / 'xp/gbuffer.frag.spv')]
    if launcher:
        cmd = [str(launcher)] + cmd
    p = subprocess.run(cmd, env=env or clean_env(), capture_output=True, text=True, timeout=600)
    if p.returncode == 2 and 'no device matching' in p.stderr:
        print(f'device {device} absent: skipped')
        sys.exit(77)
    if p.returncode != 0:
        failures.append(f'{out.name}: exit {p.returncode}: {p.stderr[-500:]}')
        return None, b''
    return json.loads([l for l in p.stdout.splitlines() if l.startswith('{')][-1]), (out / 'output.ppm').read_bytes()


def log_events(logs):
    return [json.loads(l)['event'] for f in logs.glob('fsr3-*.jsonl') for l in f.read_text().splitlines() if l.strip()]


def status(logs):
    s = logs / 'status.txt'
    return s.read_text() if s.exists() else ''


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    home = tmp / 'My Mods'
    with tarfile.open(tarball) as t:
        t.extractall(home, filter='data')
    pkg = next(home.iterdir())
    launcher, logs = pkg / 'xplane-fsr3', pkg / 'logs'
    print(f'package: {pkg.name}: {sorted(p.name for p in pkg.iterdir())}')

    # Reference image: X-Plane's own EASU, no layer.
    _, baseline = fake(tmp / 'baseline')

    # 1. Normal use: the layer loads from the package and replaces EASU.
    for i in range(12):  # stale logs from earlier launches: the launcher keeps the 10 newest
        (logs / f'fsr3-{i}.jsonl').parent.mkdir(exist_ok=True)
        (logs / f'fsr3-{i}.jsonl').write_text('')
    _, image = fake(tmp / 'normal', launcher)
    events = log_events(logs)
    if 'easu_replaced' not in events:
        failures.append(f'normal: no easu_replaced in the log ({sorted(set(events))})')
    if image == baseline:
        failures.append('normal: output identical to X-Plane\'s EASU (layer not active)')
    if not status(logs).splitlines()[1:2] or not status(logs).splitlines()[1].startswith(('LOADED on', 'ACTIVE')):
        failures.append(f'normal: status.txt {status(logs)!r}')
    if len(list(logs.glob('fsr3-*.jsonl'))) > 11:
        failures.append(f'normal: {len(list(logs.glob("fsr3-*.jsonl")))} logs kept (10 old + 1 new expected)')

    # 2. Launcher environment: Steam's runtime vs a system loader older than 1.3.211.
    def launcher_env(args, path_prefix=None):
        env = clean_env()
        if path_prefix:
            env['PATH'] = f'{path_prefix}:{env["PATH"]}'
        out = subprocess.run([str(launcher), 'sh', '-c', 'env', 'sh'] + args, env=env, capture_output=True, text=True).stdout
        return dict(l.split('=', 1) for l in out.splitlines() if '=' in l)
    steam = launcher_env(['/x/SteamLinuxRuntime_soldier/_v2-entry-point'])
    if steam.get('VK_ADD_LAYER_PATH', '').split(':')[0] != str(pkg) or 'VK_LAYER_PATH' in steam:
        failures.append(f"steam env: VK_ADD_LAYER_PATH={steam.get('VK_ADD_LAYER_PATH')} VK_LAYER_PATH={steam.get('VK_LAYER_PATH')}")
    if steam.get('VK_INSTANCE_LAYERS') != 'VK_LAYER_XPLANE_fsr3' or steam.get('XP_FSR3') != 'replace':
        failures.append(f"steam env: VK_INSTANCE_LAYERS={steam.get('VK_INSTANCE_LAYERS')} XP_FSR3={steam.get('XP_FSR3')}")
    old = tmp / 'old-loader'
    old.mkdir()
    (old / 'libvulkan.so.1.3.204').write_bytes(b'')
    (old / 'libvulkan.so.1').symlink_to(old / 'libvulkan.so.1.3.204')
    (old / 'ldconfig').write_text(f'#!/bin/sh\nprintf "\\tlibvulkan.so.1 (libc6,x86-64) => {old}/libvulkan.so.1\\n"\n')
    (old / 'ldconfig').chmod(0o755)
    legacy = launcher_env([], path_prefix=str(old))
    if legacy.get('VK_LAYER_PATH', '').split(':')[:2] != [str(pkg), '/usr/local/share/vulkan/explicit_layer.d']:
        failures.append(f"old loader: VK_LAYER_PATH={legacy.get('VK_LAYER_PATH')}")
    if 'VK_LAYER_PATH' in launcher_env([]):
        failures.append('current loader: VK_LAYER_PATH set although this loader reads VK_ADD_LAYER_PATH')
    # The fallback variable alone must find the layer.
    _, image = fake(tmp / 'layer_path', env=clean_env(VK_LAYER_PATH=str(pkg), VK_INSTANCE_LAYERS='VK_LAYER_XPLANE_fsr3',
                                                        XP_FSR3='replace', XP_FSR3_OUTPUT=str(tmp / 'layer_path_logs')))
    if image == baseline:
        failures.append('VK_LAYER_PATH only: layer not active')

    # 3. An X-Plane whose EASU interface differs: FSR 1 unchanged, and status.txt says why.
    tampered = tmp / 'fsr_4_tampered.spv'
    tampered.write_bytes(tamper_offset((build / 'xp/fsr_4.spv').read_bytes(), 'u_fsr_data', 2, 16))
    _, tampered_baseline = fake(tmp / 'tampered_baseline', easu=tampered)
    _, image = fake(tmp / 'tampered', launcher, easu=tampered)
    if image != tampered_baseline:
        failures.append('unsupported X-Plane: output differs from X-Plane\'s own EASU')
    if not status(logs).splitlines()[1:2] or not status(logs).splitlines()[1].startswith('UNSUPPORTED X-Plane version'):
        failures.append(f'unsupported X-Plane: status.txt {status(logs)!r}')

    # 4. XP_FSR3=off: X-Plane untouched.
    _, image = fake(tmp / 'off', launcher, env=clean_env(XP_FSR3='off'))
    if image != baseline:
        failures.append('XP_FSR3=off: output differs from X-Plane\'s own EASU')

    # 5. A read-only package folder: logs go to $XDG_STATE_HOME/xplane-fsr3.
    state = tmp / 'state'
    for p in [pkg] + list(pkg.rglob('*')):
        p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    try:
        _, image = fake(tmp / 'readonly', launcher, env=clean_env(XDG_STATE_HOME=str(state)))
        if image == baseline or not status(state / 'xplane-fsr3'):
            failures.append(f'read-only folder: active={image != baseline}, status={status(state / "xplane-fsr3")!r}')
    finally:
        for p in [pkg] + list(pkg.rglob('*')):
            p.chmod(p.stat().st_mode | stat.S_IWUSR)

for f in failures:
    print('FAIL', f)
print('PASS' if not failures else f'{len(failures)} failure(s)')
sys.exit(1 if failures else 0)
