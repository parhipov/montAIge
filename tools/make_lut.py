"""A creative 3D LUT for graded Rec.709 footage: two colours, blue and red, the rest held back, a film's tone.

    python tools/make_lut.py OUT.cube [--size 33]

In OKLab (perceptual: a hue moved keeps its lightness): hues between cyan and violet drawn toward a deep blue,
magenta to orange toward a dense red; the reds' chroma up, the blues' a little, greens and yellows down (so the
picture holds two colours); shadows a touch blue; the tone a soft film curve: the blacks lifted to a matte, the
highlights rolled off instead of clipped.
"""
import sys

import numpy as np

BLUE_H, BLUE_PULL, BLUE_SPAN = 255.0, 0.4, (150.0, 310.0)      # OKLab hue degrees: cyan ~195, blue ~264
RED_H, RED_PULL, RED_SPAN = 30.0, 0.25, (-50.0, 75.0)          # magenta ~328 (-32), red ~29, orange ~55
CHROMA = dict(red=1.12, blue=1.1, other=0.75)
BLACK, SHOULDER = 0.035, 0.85                                  # output black (matte), where the roll-off starts
SHADOW_B = -0.012                                              # OKLab b added in the shadows (blue)


def to_lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
               [0.0883024619, 0.2817188376, 0.6299787005]])
M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
               [0.0259040371, 0.7827717662, -0.8086757660]])


def oklab(rgb):
    return np.cbrt(to_lin(rgb) @ M1.T) @ M2.T


def from_oklab(lab):
    return to_srgb((lab @ np.linalg.inv(M2).T) ** 3 @ np.linalg.inv(M1).T)


def weight(h, lo, hi):
    """1 inside the hue span, fading to 0 over 25 degrees at its ends (degrees, any wrap)."""
    h = (h - lo) % 360
    span = (hi - lo) % 360
    return np.clip(np.minimum(h, span - h) / 25.0, 0, 1) * (h <= span)


def pull(h, target, w, k):
    d = (target - h + 180) % 360 - 180
    return h + d * w * k


def tone(L):
    """OKLab L: the matte black, a gentle S in the mids, the top rolled off toward 1."""
    x = np.clip(L, 0, 1)
    s = x - 0.12 * np.sin(2 * np.pi * x) / (2 * np.pi)                 # a light S: slope 1.12 in the mids
    top = np.where(s > SHOULDER, SHOULDER + (1 - SHOULDER) * np.tanh((s - SHOULDER) / (1 - SHOULDER) * 1.2) / np.tanh(1.2), s)
    return BLACK + (1 - BLACK) * top


def look(rgb):
    lab = oklab(rgb)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.hypot(a, b)
    h = np.degrees(np.arctan2(b, a)) % 360
    wb, wr = weight(h, *BLUE_SPAN), weight(h, *RED_SPAN)
    sat = np.clip(C / 0.04, 0, 1)                                   # greys stay grey
    h2 = pull(pull(h, BLUE_H, wb * sat, BLUE_PULL), RED_H, wr * sat, RED_PULL)
    k = CHROMA['other'] + (CHROMA['blue'] - CHROMA['other']) * wb + (CHROMA['red'] - CHROMA['other']) * wr
    k = 1 + (k - 1) * sat
    C2 = C * k
    L2 = tone(L)
    lo, hi = np.zeros_like(C2), np.ones_like(C2)                    # chroma kept inside RGB (bisection)
    for _ in range(14):
        mid = (lo + hi) / 2
        rgb2 = lab_rgb(L2, C2 * mid, h2)
        ok = ((rgb2 >= -1e-4) & (rgb2 <= 1 + 1e-4)).all(-1)
        lo, hi = np.where(ok, mid, lo), np.where(ok, hi, mid)
    full = lab_rgb(L2, C2, h2)
    inside = ((full >= -1e-4) & (full <= 1 + 1e-4)).all(-1)
    return to_srgb(np.where(inside[..., None], full, lab_rgb(L2, C2 * lo, h2)))   # linear -> sRGB


def lab_rgb(L, C, h):
    """OKLab L, C, h (the shadows' blue added) -> linear RGB."""
    a, b = C * np.cos(np.radians(h)), C * np.sin(np.radians(h)) + SHADOW_B * (1 - np.clip(L, 0, 1)) ** 2
    return (np.stack([L, a, b], -1) @ np.linalg.inv(M2).T) ** 3 @ np.linalg.inv(M1).T


def main():
    out = sys.argv[1]
    n = int(sys.argv[sys.argv.index('--size') + 1]) if '--size' in sys.argv else 33
    g = np.linspace(0, 1, n)
    b, gg, r = np.meshgrid(g, g, g, indexing='ij')                 # red runs fastest in a .cube
    rgb = np.stack([r, gg, b], -1).reshape(-1, 3)
    o = look(rgb)
    with open(out, 'w', encoding='utf-8') as fh:
        fh.write('TITLE "blue-red film"\nLUT_3D_SIZE %d\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n' % n)
        fh.writelines('%.6f %.6f %.6f\n' % tuple(v) for v in o)
    # what it does to a few colours
    for name, c in (('snow blue', (0.80, 0.86, 0.98)), ('coat red', (0.75, 0.12, 0.10)), ('board orange', (0.95, 0.35, 0.15)),
                    ('forest dark', (0.12, 0.14, 0.18)), ('green', (0.3, 0.5, 0.3)), ('grey', (0.5, 0.5, 0.5)),
                    ('black', (0, 0, 0)), ('white', (1, 1, 1))):
        print(f'{name:13s} {np.round(np.array(c) * 255).astype(int)} -> {np.round(look(np.array([c]))[0] * 255).astype(int)}')


if __name__ == '__main__':
    main()
