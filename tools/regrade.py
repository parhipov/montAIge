"""An edit's picture only: every clip's colour matched to one clip's, every shot graded (levels, white balance),
then the snow made white in every clip; the scenes and the music as they were.

    python tools/regrade.py EDIT_edl.json REF_CLIP OUT.mp4 [--keep-wb]

--keep-wb: no white balance at all (the grade's levels only, the same gain on R, G and B): the cameras matched, the
footage's blue kept.

The grade's white balance goes GRADE_WB of the way within GRADE_WB_MAX: on snow under a grey December sky,
the clips matched to a blue one, the snow stayed cyan (Lab a -10, b -9). Here the bright pixels of each clip
(luma over its p85, unclipped, after the match) are taken to SNOW (a slightly cool white), luma kept, per clip
as the match is, before the grade (after it the blue channel of the snow sat at 255 and the correction fell short).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import embedia  # noqa: E402
from montaige import edit, edl_edit  # noqa: E402

SNOW = np.array([0.985, 1.0, 1.04])      # RGB ratio of the snow wanted: Lab b about -3
SNOW_MAX = (0.7, 1.5)
Y = np.array([0.2126, 0.7152, 0.0722])


def clip_white(flights, lut):
    """Each clip's bright pixels (luma over its p85, no channel clipped), after the match, taken to SNOW, luma
    kept: the gains folded into the clip's match (an affine map before the grade), so the grade's levels see
    white snow and nothing clips first."""
    for f in flights:
        px = edit.clip_pixels(f, lut)
        if f.match is not None:
            px = edit.apply_match(px, f.match).astype(np.float32)
        y = px @ Y
        m = px[(y > np.percentile(y, 85)) & (y < np.percentile(y, 99.5)) & (px.max(1) < 250)].mean(0)
        k = SNOW * m.mean() / m
        k = np.clip(k * (m @ Y) / ((m * k) @ Y), *SNOW_MAX)
        if f.match is None:
            f.match = np.c_[np.diag(k), np.zeros(3)].tolist()
        else:
            M = np.asarray(f.match['rgb'], np.float64)
            f.match = dict(f.match, rgb=(M * k[:, None]).tolist())
        print(f'  white {f.name}: rgb {m[0]:.0f} {m[1]:.0f} {m[2]:.0f} -> x{k[0]:.2f} x{k[1]:.2f} x{k[2]:.2f}')


src, ref, out = sys.argv[1:4]
keep_wb = '--keep-wb' in sys.argv          # levels only: the footage's own blue kept (the owner liked it)
if keep_wb:
    def levels(px):
        # the white from the brightest channel, not luma: stretched by luma the snow's blue sat at 255 while green
        # went on rising, and the blue snow turned cyan (Lab a -4 -> -10)
        lo = np.percentile(px @ Y, 0.5)
        hi = np.percentile(px.max(1), 99.5)
        g = float(np.clip((255 * 0.97 - edit.GRADE_BLACK) / max(hi - lo, 1.0), *edit.GRADE_GAIN))
        return [float(lo), g, g, g]
    edit.levels = levels
with open(src, encoding='utf-8') as fh:
    e = json.load(fh)
cache = embedia.cache_of(e['flights'][0]['render'])
flights = edl_edit.load_flights(e, cache)
lut = edl_edit.lut_of(e)
edit.colour_match(flights, ref, lut)
scenes = [dict(s, shots=[dict(x) for x in s['shots']]) for s in e['scenes']]
if not keep_wb:
    clip_white(flights, lut)
edit.grade_all(scenes, flights, lut)
fl = sorted(e['flights'], key=lambda g: g['idx'])
e2 = dict(e, scenes=scenes, grade='auto', match=ref, flights=[dict(g, match=f.match) for g, f in zip(fl, flights)],
          edl_edit=dict(from_edl=src, match=ref, grade='auto', snow=None if keep_wb else SNOW.tolist()))
base = os.path.splitext(os.path.abspath(out))[0]
os.makedirs(os.path.dirname(base), exist_ok=True)
with open(base + '_edl.json', 'w', encoding='utf-8') as fh:
    json.dump(e2, fh, ensure_ascii=False, indent=1)
edl_edit.render_edl(e2, out, 480)
print('preview:', out)
