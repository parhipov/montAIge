"""Camera motion of clips our parser cannot read (GoPro), exported by the Gyroflow CLI.

    python -m embedia.media.gf_motion CACHE_DIR CLIP [CLIP ...]   ->  CACHE_DIR/<stem>_gyroflow_camera.json
    python -m embedia.media.gf_motion CACHE_DIR --list clips.txt  (one path per line; spaces are safe here)

Gyroflow cannot write into a path with spaces reliably, so it writes to the temp folder first.
"""
import os
import shutil
import subprocess
import sys
import tempfile

cache = os.path.abspath(sys.argv[1])
clips = sys.argv[2:]
if clips[:1] == ['--list']:
    with open(clips[1], encoding='utf-8') as fh:
        clips = [line.strip() for line in fh if line.strip()]
exe = os.environ.get('GYROFLOW')
if not exe or not os.path.isfile(exe):
    raise SystemExit('set GYROFLOW to Gyroflow.exe')
os.makedirs(cache, exist_ok=True)
for clip in clips:
    if not os.path.isfile(clip):
        print(clip, 'MISSING')
        continue
    clip = os.path.abspath(clip)
    stem = os.path.splitext(os.path.basename(clip))[0]
    dst = os.path.join(cache, stem + '_gyroflow_camera.json')
    if os.path.exists(dst):
        print(stem, 'exists')
        continue
    tmp = os.path.join(tempfile.gettempdir(), stem + '_gyroflow_camera.json')
    subprocess.run([exe, clip, '--export-metadata', '3:' + tmp.replace(os.sep, '/'), '-f'],
                   cwd=os.path.dirname(exe), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists(tmp) and os.path.getsize(tmp) > 10000:
        shutil.move(tmp, dst)
        print(stem, 'ok', os.path.getsize(dst))
    else:
        print(stem, 'FAILED')
    stray = os.path.splitext(clip)[0] + '.gyroflow'
    if os.path.isfile(stray):
        os.remove(stray)
