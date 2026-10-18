#!/usr/bin/env python3
"""Runs fake_xplane with and without VK_LAYER_XPLANE_fsr3 and checks what the layer did.

Cases (each with the Khronos validation layer when it is unpacked in third_party):
  baseline  no FSR layer: X-Plane's own EASU shader produces the output.
  replace   XP_FSR3=replace: FSR 3.1 from the layer's SSR-derived inputs.
  flipped   replace, with the scene's reprojection sign flipped (wrong motion vectors).
  observe   XP_FSR3=observe: the layer runs everything into its own images only.
  r_*       the same with --raster: the G-buffer is drawn as geometry (pos/vel = the depth and stencil
            views of the depth buffer, as in X-Plane), so the layer finds the main pass and jitters it.
  split     replace, with SSR and EASU in two command buffers (as X-Plane records them); r_split with --raster.
  r_zoom_*  --raster with --zoom 0.01 (scroll-wheel zoom: the projection changes every frame while
            u_reproj stays rigid). Before the layer used the previous frame's u_proj, FSR scored
            12.80 / 21.17 dB here against EASU's 13.89 / 34.69 (live: heavy ghosting while zooming).
  r_fsr_off --raster, X-Plane's FSR switched off at frame 36 (--fsr-off-after): no EASU from then on and
            the camera still. The scene of the last two frames must be identical: nothing removes
            jitter without FSR 3.1, so jitter left on shakes the picture (seen live in 0.1.x).
  split_rev the two buffers submitted in the wrong order, without validation (Lavapipe only): the
            layer must report the read as "missing" (negative control for its submission-order check).
Checks: no validation errors; the layer engaged every frame after the first; on the moving wall
FSR beats EASU and the flipped control is worse than correct vectors; observe leaves the output
byte-identical to baseline. Uploaded (unjittered) cases: the static panel is reported, not checked,
because without jitter FSR 3.1 is softer than EASU there (fsr3_selftest FSR_NO_JITTER=1 agrees).
Raster cases: with jitter FSR must beat EASU on the panel and the wall.
Usage: run_fake_xplane.py <build dir> <device name substring>   (exit 77 = device absent)
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

build, device = Path(sys.argv[1]), sys.argv[2]
validation = (build / '../../third_party/validation-layers/layer').resolve()
frames = '48'


def run(case, extra_env, args, out, validate=True):
    env = dict(os.environ, FAKE_DEVICE=device, XP_FSR3_OUTPUT=str(out))
    env.pop('XP_FSR3', None)
    env.pop('VK_INSTANCE_LAYERS', None)
    paths = [str(build)]
    if validate and validation.exists():
        env['FAKE_VALIDATION'] = '1'
        paths.append(str(validation))
    env['VK_ADD_LAYER_PATH'] = ':'.join(paths)
    env.update(extra_env)
    p = subprocess.run([str(build / 'fake_xplane'), str(build / 'xp/ssr_deferred_1.spv'), str(build / 'xp/fsr_4.spv'),
                        '--frames', frames, '--out', str(out)] + args, env=env, capture_output=True, text=True, timeout=900)
    if p.returncode == 2 and 'no device matching' in p.stderr:
        print(f'device {device} absent: skipped')
        sys.exit(77)
    lines = [l for l in p.stdout.splitlines() if l.startswith('{')]
    if p.returncode != 0 or not lines:
        print(f'{case}: exit {p.returncode}\n{p.stderr[-3000:]}')
        sys.exit(1)
    result = json.loads(lines[-1])
    log = []
    for f in out.glob('fsr3-*.jsonl'):
        log += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    result['log'] = log
    status = out / 'status.txt'
    result['status'] = status.read_text() if status.exists() else ''
    print(f"{case:9s} wall {result['psnr_wall']:6.2f} dB  panel {result['psnr_panel']:6.2f} dB  "
          f"validation errors {result['validation_errors']}  layer events {len(log)}")
    return result


failures = []
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    cases = {}
    layer = {'VK_INSTANCE_LAYERS': 'VK_LAYER_XPLANE_fsr3'}
    raster = ['--raster', str(build / 'xp/gbuffer.vert.spv'), str(build / 'xp/gbuffer.frag.spv')]
    lavapipe = 'llvmpipe' in device.lower()
    for case, env, args in ([('baseline', {}, []),
                            ('replace', dict(layer, XP_FSR3='replace'), []),
                            ('flipped', dict(layer, XP_FSR3='replace'), ['--flip-reproj']),
                            ('observe', dict(layer, XP_FSR3='observe'), []),
                            # G-buffer drawn as geometry: the layer jitters the viewport.
                            ('r_base', {}, raster),
                            ('r_replace', dict(layer, XP_FSR3='replace'), raster),
                            ('r_observe', dict(layer, XP_FSR3='observe'), raster),
                            # SSR and EASU in separate command buffers, as in the live sim.
                            ('split', dict(layer, XP_FSR3='replace'), ['--split']),
                            ('r_split', dict(layer, XP_FSR3='replace'), raster + ['--split']),
                            ('r_zoom_base', {}, raster + ['--zoom', '0.01']),
                            ('r_zoom', dict(layer, XP_FSR3='replace'), raster + ['--zoom', '0.01']),
                            # Opt-in fp16: the layer enables the features X-Plane leaves off.
                            ('r_fp16', dict(layer, XP_FSR3='replace', XP_FSR3_FP16='1'), raster),
                            # X-Plane's FSR switched off mid-flight: the camera jitter must stop with it.
                            ('r_fsr_off', dict(layer, XP_FSR3='replace'), raster + ['--fsr-off-after', '36']),
                            # MSAA on: X-Plane's multisampled SSR instead of the regular one. FSR 3.1 must
                            # step aside and say why (validation off: the MSAA dispatch is recorded
                            # without bindings into a buffer that is never submitted).
                            ('msaa', dict(layer, XP_FSR3='replace'), ['--msaa-ssr', str(build / 'xp/ssr_deferred_2.spv')])]
                           + ([('split_rev', dict(layer, XP_FSR3='replace'), ['--split-reversed'])] if lavapipe else [])):
        out = tmp / case
        out.mkdir()
        cases[case] = run(case, env, args, out, validate=case not in ('split_rev', 'msaa'))
        cases[case]['image'] = (out / 'output.ppm').read_bytes()

    for case, r in cases.items():
        if r['validation_errors']:
            failures.append(f'{case}: {r["validation_errors"]} validation errors')
        if any(e['event'] == 'engine_disabled' for e in r['log']):
            failures.append(f'{case}: engine disabled: {[e for e in r["log"] if e["event"] == "engine_disabled"]}')
    replaced = [e for e in cases['replace']['log'] if e['event'] == 'easu_replaced']
    unchanged = [e for e in cases['replace']['log'] if e['event'] == 'easu_unchanged']
    if not replaced or unchanged:
        failures.append(f'replace: easu_replaced events {len(replaced)}, easu_unchanged {unchanged[:3]}')
    base, fsr, flip = cases['baseline'], cases['replace'], cases['flipped']
    if not fsr['psnr_wall'] > base['psnr_wall'] + 1.0:
        failures.append(f"wall: FSR {fsr['psnr_wall']:.2f} dB not > EASU {base['psnr_wall']:.2f} + 1 dB")
    if not flip['psnr_wall'] < fsr['psnr_wall'] - 1.0:
        failures.append(f"wall: flipped vectors {flip['psnr_wall']:.2f} dB not < correct {fsr['psnr_wall']:.2f} - 1 dB")
    # The user-facing status file: written at startup by every case that loads the engine.
    for name in ('replace', 'observe', 'r_replace'):
        if 'LOADED on' not in cases[name]['status'] and 'ACTIVE' not in cases[name]['status']:
            failures.append(f"{name}: status.txt missing or unexpected: {cases[name]['status'][:200]!r}")
    if cases['baseline']['status']:
        failures.append('baseline (no layer): status.txt written')
    msaa = cases['msaa']
    if msaa['image'] != base['image']:
        failures.append('msaa: output differs from X-Plane\'s EASU')
    if any(e['event'] == 'easu_replaced' for e in msaa['log']) or 'msaa_scene' not in {e['event'] for e in msaa['log']}:
        failures.append('msaa: EASU replaced, or the MSAA scene SSR not noticed')
    if not any(e['event'] == 'easu_unchanged' and e.get('reason') == 'msaa' for e in msaa['log']):
        failures.append(f"msaa: refusals {sorted({e.get('reason') for e in msaa['log'] if e['event'] == 'easu_unchanged'})}, expected msaa")
    if cases['observe']['image'] != base['image']:
        failures.append('observe: output differs from baseline')
    if not any(e['event'] == 'easu_unchanged' and e.get('reason') == 'observe_mode' for e in cases['observe']['log']):
        failures.append('observe: the layer never ran FSR in observe mode')
    off = cases['r_fsr_off']
    if not any(e['event'] == 'easu_replaced' and e.get('jittered_frames', 0) >= 1 for e in off['log']):
        failures.append('r_fsr_off: never jittered while FSR 3.1 ran (the check below would prove nothing)')
    if off.get('fsr_off_frame_diff') != 0:
        failures.append(f"r_fsr_off: with X-Plane's FSR off the still scene changes between frames by "
                        f"{off.get('fsr_off_frame_diff')} (camera jitter left on)")
    # Jittered (raster) G-buffer: FSR 3.1 must beat EASU on the static panel as well as the wall.
    rb, rr = cases['r_base'], cases['r_replace']
    if not rr['psnr_panel'] > rb['psnr_panel'] + 2.0:
        failures.append(f"raster panel: FSR {rr['psnr_panel']:.2f} dB not > EASU {rb['psnr_panel']:.2f} + 2 dB")
    if not rr['psnr_wall'] > rb['psnr_wall'] + 1.5:
        failures.append(f"raster wall: FSR {rr['psnr_wall']:.2f} dB not > EASU {rb['psnr_wall']:.2f} + 1.5 dB")
    jittered = [e.get('jittered_frames', 0) for e in rr['log'] if e['event'] == 'easu_replaced']
    if not jittered or max(jittered) < 1 or not any(e['event'] == 'main_depth' for e in rr['log']):
        failures.append(f'raster: viewport jitter never applied (jittered_frames {jittered[-1:] if jittered else None})')
    # As in X-Plane, the raster G-buffer's pos/vel are the depth/stencil views of the main depth.
    vias = [e.get('via') for e in rr['log'] if e['event'] == 'main_depth']
    if vias != ['gbuffer_is_depth']:
        failures.append(f'raster: main_depth via {vias}, expected exactly one via gbuffer_is_depth')
    if not any(e['event'] == 'main_pass_census' for e in rr['log']):
        failures.append('raster: no main_pass_census event')
    if cases['r_observe']['image'] != rb['image']:
        failures.append('raster observe: output differs from baseline (observe must not jitter)')
    # Separate command buffers: every EASU replaced, every read submitted after its SSR, same quality.
    for case, panel_floor in (('split', None), ('r_split', rb['psnr_panel'] + 2.0)):
        r = cases[case]
        unchanged = [e for e in r['log'] if e['event'] == 'easu_unchanged']
        orders = [e['order'] for e in r['log'] if e['event'] == 'inputs_order']
        if unchanged or not any(e['event'] == 'easu_replaced' for e in r['log']):
            failures.append(f'{case}: easu_unchanged {unchanged[:3]}')
        if not orders or set(orders) != {'in_order'}:
            failures.append(f'{case}: inputs_order {orders[:6]}, expected only in_order')
        base_wall = (rb if case == 'r_split' else base)['psnr_wall']
        if not r['psnr_wall'] > base_wall + 1.0:
            failures.append(f"{case}: wall FSR {r['psnr_wall']:.2f} dB not > EASU {base_wall:.2f} + 1 dB")
        if panel_floor is not None and not r['psnr_panel'] > panel_floor:
            failures.append(f"{case}: panel FSR {r['psnr_panel']:.2f} dB not > EASU + 2 dB ({panel_floor:.2f})")
    # Zoom: motion vectors must include the change of projection between frames.
    zb, zr = cases['r_zoom_base'], cases['r_zoom']
    if not zr['psnr_wall'] > zb['psnr_wall'] + 2.0:
        failures.append(f"zoom: wall FSR {zr['psnr_wall']:.2f} dB not > EASU {zb['psnr_wall']:.2f} + 2 dB")
    if not zr['psnr_panel'] > zb['psnr_panel'] - 2.0:
        failures.append(f"zoom: panel FSR {zr['psnr_panel']:.2f} dB more than 2 dB below EASU {zb['psnr_panel']:.2f} (zoom ghosting)")
    # fp16: off by default; opted in, the backend must pick its 16-bit shaders (when the GPU has
    # shaderFloat16 and shaderInt16) at the same quality.
    contexts = lambda case: [e.get('fp16_shaders') for e in cases[case]['log'] if e['event'] == 'fsr_context']
    if contexts('r_replace') != [False]:
        failures.append(f"fp16: default fsr_context fp16_shaders {contexts('r_replace')}, expected [False]")
    enabled = [e.get('fp16_enabled_by_layer') for e in cases['r_fp16']['log'] if e['event'] == 'device']
    if enabled == [True] and contexts('r_fp16') != [True]:
        failures.append(f"fp16: features enabled but fsr_context fp16_shaders {contexts('r_fp16')}")
    print(f"fp16: enabled_by_layer {enabled}, fsr_context fp16_shaders {contexts('r_fp16')}")
    for k in ('psnr_wall', 'psnr_panel'):
        if abs(cases['r_fp16'][k] - rr[k]) > 0.5:
            failures.append(f"fp16: {k} {cases['r_fp16'][k]:.2f} dB vs fp32 {rr[k]:.2f} dB")
    if 'split_rev' in cases:
        orders = [e['order'] for e in cases['split_rev']['log'] if e['event'] == 'inputs_order']
        if 'missing' not in orders:
            failures.append(f'split_rev: inputs_order {orders[:6]}, expected "missing" (wrong submission order undetected)')
    telemetry = [e for e in fsr['log'] if e['event'] == 'telemetry']
    if not telemetry or abs(telemetry[-1]['fov_y'] - 1.0471976) > 1e-4:
        failures.append(f'telemetry: fov_y {telemetry[-1]["fov_y"] if telemetry else None}, expected 1.0471976')

print('FAIL\n  ' + '\n  '.join(failures) if failures else 'PASS')
sys.exit(1 if failures else 0)
