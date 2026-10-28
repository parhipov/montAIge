"""Contact sheets of an EDL from the small renders, DJI crop shown.

    python edl_sheet.py EDIT_edl.json OUT.jpg [--crop-bottom 0.16 --crop-top 0.01]
    python edl_sheet.py EDIT_edl.json OUT.jpg --cuts [--cache DIR]

Shots (default): a row per shot, 3 frames (start, middle, end).

--cuts: a row per cut, to judge the rhythm where it is decided rather than from stills. Left, the outgoing
shot 0.5 s before the cut and its last frame; right, the incoming shot's first frame and 0.5 s after; between
them 4 beats of music each side of the cut (white line): beats (downbeats brighter), musical hits (red, on
top), the track's onset envelope (grey, at the bottom), the visible rotation of both shots at their playback
speed (orange out, blue in; faint where it is not shown: what the cut left out of the outgoing clip and what
came before the incoming one), motion onsets (ticks; the long ones are strong enough to meet a hit) and
manoeuvres (red, with their kind). A line per cut on stdout, the same as its row's caption, then a summary:
the beat of the bar the cut is on and how far off, a musical hit on it, whether the camera keeps turning the
same way across it (edit.continuity, 1 = the same way) and the jump in brightness (0-255).
"""
import argparse
import contextlib
import io
import json
import os
import sys

import av
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # the repository: embedia, montaige (run as a file)
from embedia import paths  # noqa: E402
import embedia  # noqa: E402
from embedia import pix  # noqa: E402

FW = 240                # frame width on the cut sheet
PW = 480                # width of the music panel between the frames
BEATS_AROUND = 4        # beats of music each side of a cut
ROT_MAX = 250.0         # deg/s at the top of the panel
YELLOW, ORANGE, BLUE, RED, WHITE = (0, 255, 255), (0, 150, 255), (255, 170, 40), (60, 60, 255), (255, 255, 255)


def framing(im, original, a):
    """The picture as the edit shows it: DJI's crop."""
    if os.path.basename(original).upper().startswith('DJI_'):
        h, w = im.shape[:2]
        y0, y1 = int(a.crop_top * h), int((1 - a.crop_bottom) * h)
        cw = int(round((y1 - y0) * w / h))
        x0 = (w - cw) // 2
        im = im[y0:y1, x0:x0 + cw]
    return im


def frames(path, fps, idx):
    """Frames idx of a render, BGR (the last one if the render ends before)."""
    out = []
    with av.open(path) as c:
        st = c.streams.video[0]
        for fr_i in idx:
            c.seek(int(max(fr_i - 2, 0) / fps / st.time_base), stream=st, backward=True)
            im = None
            for fr in c.decode(st):
                im = pix.ndarray(fr, format='bgr24')
                if int(round(float(fr.pts * st.time_base) * fps)) >= fr_i:
                    break
            out.append(im)
    return out


def caption(row, txt):
    cv2.rectangle(row, (0, 0), (row.shape[1], 16), (0, 0, 0), -1)
    cv2.putText(row, txt, (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, YELLOW, 1, cv2.LINE_AA)


def shot_sheet(e, a):
    fl = {f['idx']: f for f in e['flights']}
    rows = []
    for s in e['scenes']:
        for x in s['shots']:
            f = fl[x['flight']]
            idx = (int(x['anchors'][0]) + 3, int((x['anchors'][0] + x['anchors'][-1]) / 2), int(x['anchors'][-1]) - 3)
            row = np.hstack([cv2.resize(framing(im, f['original'], a), (320, 180), interpolation=cv2.INTER_AREA)
                             for im in frames(f['render'], f['fps'], idx)])
            caption(row, f"{x['t0']:5.1f}-{x['t1']:5.1f}s  {s.get('day')}  {os.path.basename(f['original'])} "
                         f"@{x['src_t0']:.0f}s  {x['move']}  sk {x['skill']:.2f}")
            rows.append(row)
    half = (len(rows) + 1) // 2
    left, right = rows[:half], rows[half:]
    while len(right) < len(left):
        right.append(np.zeros_like(left[0]))
    return np.vstack([np.hstack([l, r]) for l, r in zip(left, right)]), f'ok {len(rows)} shots'


def cut_sheet(e, a):
    from montaige import edit
    fl = sorted(e['flights'], key=lambda f: f['idx'])
    with contextlib.redirect_stdout(io.StringIO()):
        flights = edit.find_flights_list([f['original'] for f in fl], a.cache or embedia.cache_of(fl[0]['render']),
                                         os.path.dirname(fl[0]['render']))
    if [os.path.basename(f.src) for f in flights] != [os.path.basename(f['original']) for f in fl]:
        raise SystemExit('the footage is not the one the EDL was made from')
    strong = 0.5 * float(np.median(np.concatenate([f.peak_strength for f in flights])))     # as edit.hit_stats
    m = edit.analyse_music(e['music'], e['bpm'], booms=e.get('booms', False))
    beats, P, hit = m['beats'], m['period'], m['hit']
    env = m['env'] / max(float(np.percentile(m['env'], 99)), 1e-9)

    def where(x, t):
        """Source frame and playback speed at music time t; outside the shot, the clip runs on at the speed of
        the shot's nearest end."""
        tm, d = edit.time_map(x), x['t1'] - x['t0']
        u = np.asarray(t, float) - x['t0']
        dm = tm.derivative()
        v0, v1 = float(dm(0.0)), float(dm(d))
        fr = np.where(u < 0, x['anchors'][0] + u * v0,
                      np.where(u > d, x['anchors'][-1] + (u - d) * v1, tm(np.clip(u, 0, d))))
        sp = np.where(u < 0, v0, np.where(u > d, v1, dm(np.clip(u, 0, d))))
        return fr, sp

    shots = [(s, x) for s in e['scenes'] for x in s['shots']]
    rows, lines, fh = [], [], None
    n_down = n_hit = n_off = n_turn = n_bright = 0
    for k in range(1, len(shots)):
        (_, xa), (sb, xb) = shots[k - 1], shots[k]
        fa, fb = flights[xa['flight']], flights[xb['flight']]
        tc = xb['t0']
        lo, hi = tc - BEATS_AROUND * P, tc + BEATS_AROUND * P
        last = int(np.clip(np.round(where(xa, tc - 0.04)[0]), 0, fa.n - 1))
        first = int(np.clip(xb['anchors'][0], 0, fb.n - 1))

        # the frames either side
        ia = [int(np.clip(np.round(where(xa, tc - 0.5)[0]), 0, fa.n - 1)), last]
        ib = [first, int(np.clip(np.round(where(xb, tc + 0.5)[0]), 0, fb.n - 1))]
        ims = ([framing(im, fa.src, a) for im in frames(fa.stab, fa.fps, ia)]
               + [framing(im, fb.src, a) for im in frames(fb.stab, fb.fps, ib)])
        if fh is None:
            fh = int(round(FW * ims[0].shape[0] / ims[0].shape[1] / 2)) * 2
        tiles = [cv2.resize(im, (FW, fh), interpolation=cv2.INTER_AREA) for im in ims]

        # the music and the motion around the cut
        pan = np.full((fh, PW, 3), 28, np.uint8)
        env_h = fh // 4
        top, bot = 16, fh - env_h - 4
        X = lambda t: int(round((t - lo) / (hi - lo) * (PW - 1)))
        Y = lambda v: int(round(bot - np.clip(v / ROT_MAX, 0, 1) * (bot - top)))
        for t, v in zip(m['env_t'], env):
            if lo <= t <= hi:
                cv2.line(pan, (X(t), fh - 1), (X(t), fh - 1 - int(min(v, 1.0) * env_h)), (85, 85, 85), 1)
        for j in np.flatnonzero((beats >= lo) & (beats <= hi)):
            down = (j - m['downbeat']) % 4 == 0
            cv2.line(pan, (X(beats[j]), top), (X(beats[j]), fh - 1), (120, 120, 120) if down else (60, 60, 60), 1)
            if hit[j] > 0:
                r = int(3 + 2 * min(hit[j], 2.0))
                cv2.fillPoly(pan, [np.array([(X(beats[j]) - r, 2), (X(beats[j]) + r, 2), (X(beats[j]), 2 + 2 * r)])], RED)
        tt = np.linspace(lo, hi, PW)
        for x, f, col, shown in ((xa, fa, ORANGE, lambda t: t < tc), (xb, fb, BLUE, lambda t: t >= tc)):
            fr, sp = where(x, tt)
            ok = (fr >= 0) & (fr <= f.n - 1) & ((tt >= x['t0']) if x is xa else (tt <= x['t1']))
            rot = sp / f.fps * f.vis[np.clip(np.round(fr).astype(int), 0, f.n - 1)]
            for j in range(PW - 1):
                if ok[j] and ok[j + 1]:
                    s = shown(tt[j])
                    cv2.line(pan, (j, Y(rot[j])), (j + 1, Y(rot[j + 1])), col if s else tuple(c // 2 for c in col),
                             2 if s else 1, cv2.LINE_AA)
            if not ok.any():
                continue
            a0, a1 = fr[ok].min(), fr[ok].max()
            for p, st in zip(f.peak_frames, f.peak_strength):
                if a0 <= p <= a1:
                    t = float(np.interp(p, fr[ok], tt[ok]))
                    c = col if shown(t) else tuple(v // 2 for v in col)
                    cv2.line(pan, (X(t), bot + 1), (X(t), bot + (9 if st >= strong else 4)), c, 1)
            for p, kind in zip(f.tricks, f.trick_kind):
                if a0 <= p <= a1:
                    t = float(np.interp(p, fr[ok], tt[ok]))
                    c = RED if shown(t) else (40, 40, 110)
                    cv2.circle(pan, (X(t), top + 4), 3, c, -1)
                    cv2.putText(pan, kind, (X(t) + 5, top + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1, cv2.LINE_AA)
        cv2.line(pan, (X(tc), 0), (X(tc), fh - 1), WHITE, 1)

        # the caption
        bi = int(np.argmin(np.abs(beats - tc)))
        off = (tc - beats[bi]) * 1000
        bar = (bi - m['downbeat']) % 4 + 1
        h = hit[bi] if abs(off) < 100 else 0.0                   # edit.hit_stats' reach
        cont = float(edit.continuity(fa.w_back[last], fb.w_fwd[first]))
        dbr = (fb.flow['bright'][first] - fa.flow['bright'][last]) if fa.flow is not None and fb.flow is not None \
            else np.nan
        txt = (f"cut {k}  {tc:5.1f}s  {'scene' if xb is sb['shots'][0] else 'jump '}  beat {bar}/4 {off:+4.0f}ms"
               + (f"  HIT {h:.1f}" if h else '')
               + f"  |  {fa.name} @{last / fa.fps:.1f}s {xa['move']} {where(xa, tc - 0.04)[1] / fa.fps:.2f}x"
               f"  ->  {fb.name} @{first / fb.fps:.1f}s {xb['move']} {where(xb, tc)[1] / fb.fps:.2f}x"
               f"  |  turn {cont:.2f}  bright {dbr:+.0f}")
        lines.append(txt)
        n_down += bar == 1
        n_hit += h > 0
        n_off += abs(off) > 40
        n_turn += cont > 0.6
        n_bright += abs(dbr) > 40
        row = np.vstack([np.zeros((16, 4 * FW + PW, 3), np.uint8), np.hstack(tiles[:2] + [pan] + tiles[2:])])
        caption(row, txt)
        rows.append(row)
    if not rows:
        raise SystemExit('one shot, no cuts')
    n = len(rows)
    summary = (f'{n} cuts: on a downbeat {n_down}, on a musical hit {n_hit} (of {int((hit > 0).sum())} hits), '
               f'off the beat by over 40 ms {n_off}, turning on the same way (over 0.6) {n_turn}, '
               f'brightness jumps over 40: {n_bright}')
    return np.vstack(rows), '\n'.join(lines + [summary])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('edl')
    ap.add_argument('out')
    ap.add_argument('--cuts', action='store_true', help='a row per cut with the music around it (see above)')
    ap.add_argument('--cache', default=None, help="camera motion cache (--cuts; default: the index's)")
    ap.add_argument('--crop-bottom', type=float, default=0.16)
    ap.add_argument('--crop-top', type=float, default=0.01)
    a = ap.parse_args()
    e = json.load(open(a.edl, encoding='utf-8'))
    img, report = (cut_sheet if a.cuts else shot_sheet)(e, a)
    cv2.imwrite(a.out, img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    print(report)


if __name__ == '__main__':
    main()
