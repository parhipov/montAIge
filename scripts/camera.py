"""Flown or carried, per clip, from the camera's own accelerometer (a GoPro's GPMF ACCL, ~200 a second).

    python camera.py CLIP [...]                      one line per clip
    python camera.py --list originals.txt --out cameras.json

A drone's thrust shakes the camera: the acceleration under 5 Hz moved by 1.6-5.4 m/s2 RMS over every flight of
over 15 s (63 GoPro clips of six shoots: FPV over towns, forests, water, a canyon), and by 0.2-1.1 on a skier's
helmet or in a hand (16 clips). The camera's rotation does not tell them apart: a calm flight peaked at 93-138
deg/s, a run down a slope at 55-134. A clip without an accelerometer (DJI, a clip stabilized elsewhere) or under
4 s (fiddling before a take-off) gets no verdict: the job's default (montage.py `camera`).
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gps  # noqa: E402

FLOWN_LF = 1.35         # m/s2 RMS of the acceleration magnitude under 5 Hz: over it, flown
MIN_S = 4.0


def kind(path):
    """('drone' | 'carried' | None, the low-frequency acceleration RMS m/s2 or None)."""
    acc, hz = gps.gpmf_stream(path, b'ACCL')
    if acc is None or len(acc) < MIN_S * hz:
        return None, None
    a = np.linalg.norm(acc, axis=1)
    lf = float(np.sqrt(np.mean((sosfiltfilt(butter(2, 5, fs=hz, output='sos'), a) - a.mean()) ** 2)))
    return ('drone' if lf >= FLOWN_LF else 'carried'), lf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('clips', nargs='*')
    ap.add_argument('--list', default=None, help='text file: one clip per line')
    ap.add_argument('--out', default=None, help='JSON {clip name: "drone" | "carried"}, clips with a verdict only')
    a = ap.parse_args()
    clips = list(a.clips)
    if a.list:
        with open(a.list, encoding='utf-8') as fh:
            clips += [x.strip() for x in fh if x.strip()]
    out = {}
    for c in clips:
        try:
            k, lf = kind(c)
        except (OSError, ValueError) as e:
            k, lf = None, None
            print(f'{os.path.basename(c)}: unreadable ({e})')
            continue
        name = os.path.splitext(os.path.basename(c))[0]
        print(f'{name}: ' + (f'{k} (acceleration under 5 Hz {lf:.2f} m/s2)' if k else 'no verdict (no accelerometer, '
                                                                                 'or under 4 s)'))
        if k:
            out[name.lower()] = k
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump(out, fh, indent=1)
    kinds = list(out.values())
    print(f'{kinds.count("drone")} flown, {kinds.count("carried")} carried, {len(clips) - len(kinds)} without a verdict')


if __name__ == '__main__':
    main()
