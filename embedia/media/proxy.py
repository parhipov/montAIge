"""Small proxies of clips stabilized elsewhere (no motion data left for Gyroflow): frame for frame, scaled.

    python -m embedia.media.proxy OUT_DIR --list clips.txt --width 640 --height 360

clips.txt: one path per line -> OUT_DIR/<stem>_stabilized.mp4, the same name a Gyroflow proxy gets (a clip
named GH010397_stabilized.mp4 gives GH010397_stabilized_stabilized.mp4). Written to .tmp first, so a broken
run leaves nothing that looks done. Every decoded frame is kept, so frame k of the proxy is
frame k of the clip (final.py cuts the clip itself by these numbers); the frames keep the clip's timestamps
(numbering them anew in another time base made the muxer fail with EINVAL).
"""
import argparse
import os
import time

import av

from . import pix

ap = argparse.ArgumentParser()
ap.add_argument('out_dir')
ap.add_argument('--list', required=True)
ap.add_argument('--width', type=int, default=640)
ap.add_argument('--height', type=int, default=360)
ap.add_argument('--crf', type=int, default=20)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
with open(a.list, encoding='utf-8') as fh:
    clips = [x.strip() for x in fh if x.strip()]
for clip in clips:
    stem = os.path.splitext(os.path.basename(clip))[0]
    dst = os.path.join(a.out_dir, stem + '_stabilized.mp4')
    if os.path.exists(dst):
        print(f'{stem}: exists, skipped', flush=True)
        continue
    t0 = time.time()
    ic = av.open(clip)
    ist = ic.streams.video[0]
    ist.thread_type = 'AUTO'
    rate = ist.average_rate
    tmp = dst + '.tmp'
    oc = av.open(tmp, 'w', format='mp4')
    vs = oc.add_stream('libx264', rate=rate)
    vs.width, vs.height, vs.pix_fmt = a.width, a.height, 'yuv420p'
    vs.options = {'crf': str(a.crf), 'preset': 'veryfast'}
    n = 0
    for fr in ic.decode(ist):
        o = pix.reformat(fr, width=a.width, height=a.height, format='yuv420p')   # keeps the clip's own timestamps
        for p in vs.encode(o):
            oc.mux(p)
        n += 1
    for p in vs.encode(None):
        oc.mux(p)
    oc.close()
    ic.close()
    os.replace(tmp, dst)
    print(f'{stem}: {n} frames in {time.time() - t0:.0f}s', flush=True)
