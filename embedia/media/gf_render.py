"""Stabilize clips (or trimmed ranges of them) with the Gyroflow CLI.

    python -m embedia.media.gf_render OUT_DIR --list clips.txt --width 640 --height 360 --bitrate 3M
    python -m embedia.media.gf_render OUT_DIR --ranges ranges.json --width 1920 --height 1080 --bitrate 20M

clips.txt: one clip path per line -> <stem>_stabilized.mp4 (whole clip).
ranges.json: {"<clip path>": [[start_ms, end_ms], ...], ...} -> <stem>_stabilized-001.mp4 ... per range.
--calls FILE: a JSON line per render, {clip, output, gyroflow (its version), args (Gyroflow's command line after its
own path)}: how the render was made, for the index's recipe of a proxy.
Gyroflow is taken from --gyroflow, then the GYROFLOW environment variable. It keeps the
stabilization settings saved in its GUI; the bitrate is forced through encoder options because
the CLI never goes below the source bitrate.
"""
import argparse
import json
import os
import subprocess
import time

ap = argparse.ArgumentParser()
ap.add_argument('out_dir')
ap.add_argument('--list', default=None)
ap.add_argument('--ranges', default=None)
ap.add_argument('--width', type=int, default=640)
ap.add_argument('--height', type=int, default=360)
ap.add_argument('--bitrate', default='3M')
ap.add_argument('--gyroflow', default=os.environ.get('GYROFLOW'))
ap.add_argument('--codec', choices=('h264', 'hevc10'), default='h264',
                help='hevc10: 10-bit H.265 (finals from log footage: no banding before the LUT)')
ap.add_argument('--calls', default=None, help='a JSON line per render: how Gyroflow was called')
a = ap.parse_args()
if not a.gyroflow or not os.path.isfile(a.gyroflow):
    raise SystemExit('Gyroflow not found: pass --gyroflow or set GYROFLOW')
version = None
if a.calls:
    out = subprocess.run([a.gyroflow, '--version'], capture_output=True, text=True, errors='replace',
                         cwd=os.path.dirname(a.gyroflow)).stdout
    version = next((x.split()[-1].lstrip('v') for x in out.splitlines() if x.startswith('Gyroflow v')), None)
out_dir = os.path.abspath(a.out_dir)
os.makedirs(out_dir, exist_ok=True)
jobs = []
if a.ranges:
    with open(a.ranges, encoding='utf-8') as fh:
        jobs += [(clip, rr) for clip, rr in json.load(fh).items()]
if a.list:
    with open(a.list, encoding='utf-8') as fh:
        jobs += [(line.strip(), None) for line in fh if line.strip()]
rate = float(a.bitrate.rstrip('Mm'))
# H.264 on NVENC is 8-bit only: a 10-bit source (O4 Pro D-Log M) failed every frame with
# "10 bit encode not supported" unless the encoder is told to take 8-bit frames
opts = f'-b {a.bitrate} -maxrate {rate * 1.4:.0f}M -bufsize {rate * 3:.0f}M -pix_fmt ' + \
    ('p010le' if a.codec == 'hevc10' else 'nv12')
folder = out_dir.replace(os.sep, '/') + '/'
for clip, rr in jobs:
    clip = os.path.abspath(clip)
    stem = os.path.splitext(os.path.basename(clip))[0]
    if rr is None and os.path.exists(os.path.join(out_dir, stem + '_stabilized.mp4')):
        print(f'{stem}: exists, skipped', flush=True)
        continue
    out_params = {'codec': 'H.265/HEVC' if a.codec == 'hevc10' else 'H.264/AVC', 'use_gpu': True, 'audio': True,
                  'output_width': a.width, 'output_height': a.height, 'encoder_options': opts, 'output_folder': folder}
    cmd = [a.gyroflow, clip]
    if rr:
        out_params['export_trims_separately'] = True
        # Smooth over the whole clip, not only inside each range: with Gyroflow's default
        # (trim_range_only on) the camera is taken as frozen past the range edges, and a 6 s piece
        # differed from the whole-clip render by 2-3 (0-255 grey) in its middle and up to 20 in
        # its last 1.5 s; with it off, 0.2-0.4 in the middle and 2-4 at the edges (zoom).
        cmd += ['--preset', json.dumps({'version': 2, 'trim_ranges_ms': rr, 'stabilization': {
            'smoothing_params': [{'name': 'trim_range_only', 'value': 0.0}]}})]
    cmd += ['-p', json.dumps(out_params), '-f']
    t0 = time.time()
    log = os.path.join(out_dir, stem + ('_ranges' if rr else '') + '.gflog')
    with open(log, 'wb') as fh:
        subprocess.run(cmd, stdout=fh, stderr=fh, cwd=os.path.dirname(a.gyroflow))
    stray = os.path.splitext(clip)[0] + '.gyroflow'
    if os.path.isfile(stray) and os.path.getmtime(stray) >= t0 - 1:
        os.remove(stray)                                         # Gyroflow sometimes leaves its project beside the clip
    if a.calls:
        with open(a.calls, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(dict(clip=clip, output=os.path.join(out_dir, stem + '_stabilized.mp4') if not rr else None,
                                     gyroflow=version, args=cmd[1:]), ensure_ascii=False) + '\n')
    print(f'{stem}: {"%d ranges" % len(rr) if rr else "full"} in {time.time() - t0:.0f}s', flush=True)
