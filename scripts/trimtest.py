"""Does a trimmed Gyroflow render match the whole-clip render near its edges? trim_range_only on vs off.

    python trimtest.py CLIP FULL_RENDER START_MS END_MS OUT_DIR
Prints the mean |difference| (0-255 grey, 64x36) against the whole-clip render, per 0.5 s from the start.
"""
import json
import os
import subprocess
import sys

import av
import numpy as np

clip, full, a_ms, b_ms, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), os.path.abspath(sys.argv[5])
exe = os.environ['GYROFLOW']


def grey(path, first, count):
    c = av.open(path)
    st = c.streams.video[0]
    fps = float(st.average_rate)
    c.seek(int(max(first - 2, 0) / fps / st.time_base), stream=st, backward=True)
    res = {}
    for fr in c.decode(st):
        i = int(round(float(fr.pts * st.time_base) * fps))
        if i >= first + count:
            break
        if i >= first:
            res[i - first] = fr.to_ndarray(format='gray', width=64, height=36).astype(np.float32)
    c.close()
    return res, fps


for tag, flag in (('on', None), ('off', 0.0)):
    d = os.path.join(out, tag)
    os.makedirs(d, exist_ok=True)
    preset = {'version': 2, 'trim_ranges_ms': [[a_ms, b_ms]]}
    if flag is not None:
        preset['stabilization'] = {'smoothing_params': [{'name': 'trim_range_only', 'value': flag}]}
    params = {'codec': 'H.264/AVC', 'use_gpu': True, 'audio': False, 'output_width': 640, 'output_height': 360,
              'encoder_options': '-b 3M', 'output_folder': d.replace(os.sep, '/') + '/'}
    subprocess.run([exe, clip, '--preset', json.dumps(preset), '-p', json.dumps(params), '-f'],
                   cwd=os.path.dirname(exe), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    piece = [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.mp4')][0]
    pc, fps = grey(piece, 0, int((b_ms - a_ms) / 1000 * fps if 'fps' in dir() else 300))
    k0 = int(round(a_ms / 1000 * fps))
    fl, _ = grey(full, k0, len(pc))
    diffs = [np.abs(pc[i] - fl[i]).mean() for i in sorted(pc) if i in fl]
    step = int(fps / 2)
    print(f'trim_range_only {tag:3s}: diff vs whole-clip render per 0.5 s: '
          + ' '.join(f'{np.mean(diffs[j:j + step]):4.1f}' for j in range(0, len(diffs), step)))
