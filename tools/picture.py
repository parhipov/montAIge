"""A finished edit's picture changed frame by frame (a preview, or a final from final.py), the cuts and the sound as
they are: a look, a light on people, motion blur that varies with the music and the motion.

    python tools/picture.py IN.mp4 EDIT_edl.json OUT.mp4 [--look festive|frost|kino] [--blur 0.5]
        [--blur-hits 2.5] [--blur-speed 2] [--blur-dive 4.5] [--people BOXES.npz] [--lut FILE.cube]
        [--ffmpeg FFMPEG --range FROM-TO]

The motion blur needs SharpFrame (https://github.com/parhipov/SharpFrame): cloned beside this repository,
or its folder in SHARPFRAME.
A 4K final: --ffmpeg (hevc_nvenc), --people-width of the preview the boxes were found in; try --range first.

--look festive: a New Year look over the graded picture: an S-curve on luma (hue kept), vibrance (the duller
colours more, the reds a little more: the coat, the board), shadows a touch blue and highlights a touch warm,
a soft glow off the bright snow, a gentle vignette.

--blur S: motion blur of a shutter open S of the frame interval (0.5 = 180 degrees, as a film camera), drawn
from the picture's own motion: dense flow (DIS) to the frames before and after within the same shot (never
across a cut, taken from the EDL), each pixel smeared along its path over the exposure with SharpFrame's
streak() (a sample counts only where its source moves as the pixel does, so a person against the ground does
not drag the ground's colour into it). A repeated frame (a slowed piece) takes the motion of the other side.

--look frost: the footage's own blue kept (the owner: "the bluish was cool, the contrast of the blue and the red"):
the mids lifted a little, an S-curve, vibrance, the reds much more, shadows a touch blue, no warm highlights, the
glow and the vignette.

--look kino: the shadows up on people only (--people NPZ: tools/people_boxes.py's boxes of this video; a soft ellipse
round each person, the boxes linked into tracks with the gaps filled and the ends faded (people_tracks: no blinking); inside it a dark pixel gains up to
x2, by its own luma, the snow round them untouched: a fill light on the people, the forest and the water stay
deep), then --lut (tools/make_lut.py's blue-red film LUT), a little glow, the vignette.

--blur-hits MAX: the shutter --blur most of the time, opening up to MAX frame intervals on each musical hit (the
EDL's music, edit.py's hits: the strongest to MAX), in 0.08 s and closing over ~0.25 s: the picture smears on the beat.
--blur-speed MAX: the shutter by how fast the picture moves (p90 of the motion): --blur at 5 px a frame (640 px
wide) or less, MAX at 18 or more: the passes and the fast runs streak, the calm stays sharp.
--blur-dive MAX: the shots the EDL calls a dive blurred by speed up to this (alone, with --blur-speed or with
--blur-hits: the larger shutter wins; the owner: "the drone falling down: bang the blur").
"""
import argparse
import json
import os
import subprocess
import sys

import av
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the repository
sys.path.insert(0, ROOT)
SHARPFRAME_URL = 'https://github.com/parhipov/SharpFrame'
sys.path.insert(0, os.environ.get('SHARPFRAME', os.path.join(os.path.dirname(ROOT), 'SharpFrame')))
try:
    import sharpframe as sf  # noqa: E402
except ImportError:
    raise SystemExit(f'SharpFrame not found: git clone {SHARPFRAME_URL} beside this repository, or set SHARPFRAME '
                     'to its folder')

Y = np.array([0.2126, 0.7152, 0.0722], np.float32)


def festive(img):
    x = img.astype(np.float32) / 255
    y = x @ Y
    # S-curve on luma, the colour scaled with it: slope 1.25 in the mids, 0.75 at the ends
    y2 = y - 0.25 * np.sin(2 * np.pi * y) / (2 * np.pi)
    x = x * (y2 / np.maximum(y, 1e-4))[..., None]
    y = y2
    # vibrance: chroma up by 35 % where dull, less where already strong; reds +15 % more
    mx, mn = x.max(-1), x.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1e-4)
    red = np.clip((x[..., 0] - np.maximum(x[..., 1], x[..., 2])) * 4, 0, 1)
    x = y[..., None] + (x - y[..., None]) * (1 + 0.35 * (1 - sat) + 0.15 * red)[..., None]
    # split tone
    ws, wh = ((1 - y) ** 2)[..., None], (y ** 2)[..., None]
    x = x + ws * np.float32([-0.012, 0.0, 0.03]) + wh * np.float32([0.012, 0.004, -0.012])
    # glow off the brights, screened
    h, w = y.shape
    hi = np.clip((x - 0.72) / 0.28, 0, 1)
    glow = cv2.GaussianBlur(hi, (0, 0), w * 0.015)
    x = 1 - (1 - np.clip(x, 0, 1)) * (1 - 0.22 * glow)
    # vignette
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx - w / 2) / (w / 2)) ** 2 * 0.5 + ((yy - h / 2) / (h / 2)) ** 2 * 0.5
    x = x * (1 - 0.18 * r2)[..., None]
    return np.clip(x * 255 + 0.5, 0, 255).astype(np.uint8)


def frost(img, lift=0.8):
    x = img.astype(np.float32) / 255
    y = x @ Y
    h, w = y.shape
    # the shadows up where the place around is dark (the owner: "the shadows need pulling up"): a gain by the
    # surrounding luma (blurred ~2 % of the width), up to 1 + lift in dark places, none from luma 0.5 up; local,
    # so a dark coat on bright snow comes up as a dark forest does and the snow stays as it is
    sur = cv2.GaussianBlur(y, (0, 0), w * 0.02)
    t = np.clip((sur - 0.08) / 0.42, 0, 1)
    y2 = np.clip(y * (1 + lift * (1 - t * t * (3 - 2 * t))), 0, 1)
    y2 = np.power(y2, 0.92)                                                     # the mids up a little
    y2 = np.where(y2 > 0.5, y2 - 0.2 * np.sin(2 * np.pi * y2) / (2 * np.pi), y2)   # contrast in the brights only
    x = x * (y2 / np.maximum(y, 1e-4))[..., None]
    y = y2
    mx, mn = x.max(-1), x.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1e-4)
    red = np.clip((x[..., 0] - np.maximum(x[..., 1], x[..., 2])) * 4, 0, 1)
    x = y[..., None] + (x - y[..., None]) * (1 + 0.2 * (1 - sat) + 0.35 * red)[..., None]
    x = x + ((1 - y) ** 2)[..., None] * np.float32([-0.012, 0.0, 0.02])
    hi = np.clip((x - 0.72) / 0.28, 0, 1)
    glow = cv2.GaussianBlur(hi, (0, 0), w * 0.015)
    x = 1 - (1 - np.clip(x, 0, 1)) * (1 - 0.2 * glow)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx - w / 2) / (w / 2)) ** 2 * 0.5 + ((yy - h / 2) / (h / 2)) ** 2 * 0.5
    x = x * (1 - 0.18 * r2)[..., None]
    return np.clip(x * 255 + 0.5, 0, 255).astype(np.uint8)


TRACK_GAP, TRACK_FADE, TRACK_MIN = 15, 6, 3     # frames: a gap filled, the fade in and out, detections a track needs


def people_tracks(rows, shot_of_frame):
    """{frame: [(x0, y0, x1, y1, strength)]}: tools/people_boxes.py's boxes linked into tracks within a shot (the nearest
    centre within 0.8 of the box's height, across gaps up to TRACK_GAP frames), the gaps filled by the box moved
    linearly, the strength (confidence over 0.45, capped at 1) smoothed over +-3 frames and faded in and out over
    TRACK_FADE frames at the track's ends (the box held there); tracks of fewer than TRACK_MIN detections dropped.
    A person the detector misses for a few frames keeps the light; it never blinks."""
    by = {}
    for r in rows:
        by.setdefault(int(r[0]), []).append(np.asarray(r[1:], np.float32))
    tracks, live = [], []
    for i in sorted(by):
        live = [t for t in live if i - t[-1][0] <= TRACK_GAP and shot_of_frame(t[-1][0]) == shot_of_frame(i)]
        free = list(live)
        for bx in sorted(by[i], key=lambda q: -q[4]):
            c, hgt = np.array([(bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2]), bx[3] - bx[1]
            best, bd = None, 0.8
            for t in free:
                p = t[-1][1]
                d = np.hypot(*(c - [(p[0] + p[2]) / 2, (p[1] + p[3]) / 2])) / max(hgt, p[3] - p[1], 1)
                if d < bd:
                    best, bd = t, d
            if best is None:
                best = []
                tracks.append(best)
                live.append(best)
            else:
                free = [t for t in free if t is not best]
            best.append((i, bx))
    out = {}
    for t in tracks:
        if len(t) < TRACK_MIN:
            continue
        f = np.array([q[0] for q in t])
        B = np.array([q[1] for q in t])
        sh = shot_of_frame(f[0])
        frames = np.arange(f[0] - TRACK_FADE, f[-1] + TRACK_FADE + 1)
        frames = frames[[shot_of_frame(j) == sh and j >= 0 for j in frames]]
        box = np.stack([np.interp(frames, f, B[:, k]) for k in range(4)], 1)
        st = np.interp(frames, f, np.clip(B[:, 4] / 0.45, 0, 1))
        st = np.convolve(np.pad(st, 3, mode='edge'), np.ones(7) / 7, mode='valid')
        ramp = np.clip(np.minimum(frames - (f[0] - TRACK_FADE), (f[-1] + TRACK_FADE) - frames) / TRACK_FADE, 0, 1)
        for j, bx, s_ in zip(frames, box, st * ramp):
            if s_ > 0.01:
                out.setdefault(int(j), []).append((*bx, float(s_)))
    return out


def people_mask(tracks, i, shape, wide=1.0):
    """0..1: a soft ellipse round each person of frame i (people_tracks). wide > 1: the ellipse that many times the
    person (at least a quarter of the frame's height across) and feathered over most of it: the light spreads
    round them and moves slowly with them, unseen as kids slide down a hill (the owner)."""
    h, w = shape
    m = np.zeros((h // 2, w // 2), np.float32)
    for x0, y0, x1, y1, st in tracks.get(i, ()):
        cx, cy, ax, ay = (x0 + x1) / 4, (y0 + y1) / 4, (x1 - x0) / 4 * 1.3, (y1 - y0) / 4 * 1.2
        feather = 0.25
        if wide > 1:
            ax, ay = max(ax * wide, h / 16), max(ay * wide, h / 16)
            feather = 0.6
        one = np.zeros_like(m)
        cv2.ellipse(one, (int(cx), int(cy)), (max(int(ax), 1), max(int(ay), 1)), 0, 0, 360, 1.0, -1)
        one = cv2.GaussianBlur(one, (0, 0), max(1.0, feather * ay))
        m = np.maximum(m, one * st)
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)


R2 = {}                                     # the vignette's r^2 per frame size


def kino(img, mask, lut, lift=1.0, mix=1.0, glow_k=0.15, vig=0.15):
    x = img.astype(np.float32) / 255
    y = x @ Y
    h, w = y.shape
    if mask is not None:
        t = np.clip((y - 0.05) / 0.5, 0, 1)
        y2 = np.clip(y * (1 + lift * mask * (1 - t * t * (3 - 2 * t))), 0, 1)
        x = np.clip(x * (y2 / np.maximum(y, 1e-4))[..., None], 0, 1)
    if lut is not None:
        q = np.clip(x * 255 + 0.5, 0, 255).astype(np.uint8)
        x = x + mix * (lut.table[q[..., 2], q[..., 1], q[..., 0]].astype(np.float32) / 255 - x)
    small = cv2.resize(x, (w // 4, h // 4), interpolation=cv2.INTER_AREA)   # the glow on a quarter-size frame
    glow = cv2.GaussianBlur(np.clip((small - 0.75) / 0.25, 0, 1), (0, 0), w / 4 * 0.015)
    glow = cv2.resize(glow, (w, h), interpolation=cv2.INTER_LINEAR)
    x = 1 - (1 - np.clip(x, 0, 1)) * (1 - glow_k * glow)
    if (h, w) not in R2:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        R2[h, w] = ((xx - w / 2) / (w / 2)) ** 2 * 0.5 + ((yy - h / 2) / (h / 2)) ** 2 * 0.5
    x = x * (1 - vig * R2[h, w])[..., None]
    return np.clip(x * 255 + 0.5, 0, 255).astype(np.uint8)


def hit_shutter(e, base, top):
    """t -> the shutter: base, up to top on each musical hit of the EDL's music."""
    from montaige import edit
    m = edit.analyse_music(e['music'], e['bpm'], booms=e.get('booms'))
    hits = [(t, min(h, 2.0) / 2) for t, h in zip(m['beats'], m['hit']) if h > 0]
    print(f'  {len(hits)} hits: ' + ' '.join(f'{t:.1f}' for t, _ in hits))

    def at(t):
        s = base
        for th, h in hits:
            d = t - th
            env = (1 + d / 0.08) if -0.08 <= d < 0 else np.exp(-d / 0.25) if d >= 0 else 0.0
            s = max(s, base + (top - base) * h * env)
        return s
    return at


def flow(a, b):
    """F with a(x) ~ b(x + F(x)), px of this frame."""
    return sf.flow_to(a, b)


SHUT = []                                   # the shutter of every frame blurred


FLOW_W, STREAK_W = 960, 1920    # px: the flow and the streaks are drawn at most this wide (a 4K final)


def blur(prev, cur, nxt, shutter):
    """cur (rgb uint8) with the motion blur of the exposure around it; prev/nxt None outside the shot; shutter a
    number or (V, speed in px a frame of a 640-wide picture) -> number. On a frame wider than STREAK_W the streaks
    are drawn on a STREAK_W grid and blended into the full frame where they are over ~1 px long (what stands still
    keeps its full detail); the flow is found on a FLOW_W grid."""
    H, W = cur.shape[:2]
    fw = min(W, FLOW_W)
    fh = int(round(H * fw / W))
    g = lambda im: cv2.cvtColor(im if fw == W else cv2.resize(im, (fw, fh), interpolation=cv2.INTER_AREA),
                                cv2.COLOR_RGB2GRAY)
    c = g(cur)
    sides = []
    if nxt is not None:
        sides.append(flow(c, g(nxt)))
    if prev is not None:
        sides.append(-flow(c, g(prev)))
    if not sides:
        return cur
    mags = [np.median(np.hypot(F[..., 0], F[..., 1])) for F in sides]
    if len(sides) == 2 and min(mags) > 0.25 * max(mags):
        V = (sides[0] + sides[1]) / 2
    else:                                   # a repeated frame on one side: the other side's motion
        V = sides[int(np.argmax(mags))]
    speed = np.hypot(V[..., 0], V[..., 1])
    if callable(shutter):
        shutter = shutter(V, speed * (640 / fw))
    SHUT.append(shutter)
    bw = min(W, STREAK_W)
    bh = int(round(H * bw / W))
    if bw != fw:
        V = cv2.resize(V, (bw, bh), interpolation=cv2.INTER_LINEAR) * (bw / fw)
        speed = np.hypot(V[..., 0], V[..., 1])
    if np.percentile(speed, 99) * shutter < 0.7:
        return cur
    steps = int(np.clip(np.ceil(np.percentile(speed, 99) * shutter / 1.0), 1, 128))
    dts = [shutter * ((i + 0.5) / steps - 0.5) for i in range(steps)]
    A = cur if bw == W else cv2.resize(cur, (bw, bh), interpolation=cv2.INTER_AREA)
    out = sf.streak(A.astype(np.float32), V.astype(np.float32), dts)
    if bw == W:
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)
    m = np.clip((speed * shutter * (W / bw) - 1) / 4, 0, 1).astype(np.float32)
    m = cv2.resize(cv2.GaussianBlur(m, (0, 0), 2), (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
    up = cv2.resize(out, (W, H), interpolation=cv2.INTER_CUBIC)
    return np.clip(cur * (1 - m) + up * m + 0.5, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('edl')
    ap.add_argument('out')
    ap.add_argument('--look', choices=('none', 'festive', 'frost', 'kino'), default='none')
    ap.add_argument('--blur', type=float, default=0.0)
    ap.add_argument('--people', default=None, help="--look kino: tools/people_boxes.py's NPZ of this video")
    ap.add_argument('--lut', default=None, help='--look kino: a .cube applied after the lift')
    ap.add_argument('--gentle', action='store_true', help='--look kino at half: the lift x0.5, the LUT mixed 50 %%, '
                    'less glow and vignette (the owner: "you are visibly pulling the curves")')
    ap.add_argument('--lut-mix', type=float, default=None, help="--look kino: the LUT's share (1 = in full; --gentle 0.5)")
    ap.add_argument('--wide', type=float, default=1.0, help="--look kino: the light round people this many times "
                    "their size, feathered wide")
    ap.add_argument('--blur-hits', type=float, default=0.0)
    ap.add_argument('--blur-speed', type=float, default=0.0)
    ap.add_argument('--blur-dive', type=float, default=0.0)
    ap.add_argument('--people-width', type=int, default=640, help='the width of the video the --people boxes were '
                    'found in (the preview; the final shares its timeline)')
    ap.add_argument('--ffmpeg', default=None, help='encode through this ffmpeg with hevc_nvenc (a 4K final)')
    ap.add_argument('--cq', type=int, default=18, help='--ffmpeg: hevc_nvenc constant quality')
    ap.add_argument('--range', default=None, help='FROM-TO seconds only (a test)')
    a = ap.parse_args()
    with open(a.edl, encoding='utf-8') as fh:
        e = json.load(fh)
    on_hit = hit_shutter(e, a.blur, a.blur_hits) if a.blur_hits else None
    boxes, lut = {}, None
    if a.lut:
        from montaige import edit
        lut = edit.CubeLUT(a.lut)
    cuts = sorted({x['t0'] for s in e['scenes'] for x in s['shots']} - {0.0})
    moves = sorted((x['t0'], x.get('move')) for s in e['scenes'] for x in s['shots'])

    ic = av.open(a.src)
    vs = ic.streams.video[0]
    fps = vs.average_rate
    W, H = vs.codec_context.width, vs.codec_context.height
    t_from, t_to = (float(v) for v in a.range.split('-')) if a.range else (0.0, 1e5)
    oa = pipe = None
    if a.ffmpeg:
        cmd = [a.ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
               '-r', str(fps), '-i', '-', '-ss', f'{t_from:.3f}', '-t', f'{t_to - t_from:.3f}', '-i', a.src,
               '-map', '0:v', '-map', '1:a?', '-c:a', 'copy', '-c:v', 'hevc_nvenc', '-preset', 'p5', '-rc', 'vbr',
               '-cq', str(a.cq), '-b:v', '0', '-pix_fmt', 'yuv420p', '-tag:v', 'hvc1', '-shortest', a.out]
        pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    else:
        oc = av.open(a.out, 'w')
        ov = oc.add_stream('libx264', rate=fps)
        ov.width, ov.height, ov.pix_fmt = W, H, 'yuv420p'
        ov.options = {'crf': '17', 'preset': 'medium'}
        if ic.streams.audio:
            oa = oc.add_stream_from_template(ic.streams.audio[0])

    def shot_of(t):
        return int(np.searchsorted(cuts, t + 1e-3, side='right'))

    if a.people:
        rows = np.load(a.people)['boxes'].copy()
        rows[:, 1:5] *= W / a.people_width
        boxes = people_tracks(rows, lambda j: shot_of(j / float(fps)))
        print(f'  people: {sum(len(v) for v in boxes.values())} boxes in {len(boxes)} frames after tracking')

    def emit(item, prev, nxt):
        t, im = item
        if not t_from - 1e-3 <= t < t_to:
            return
        if a.blur > 0:
            same = lambda o: o is not None and shot_of(o[0]) == shot_of(t)
            fixed = on_hit(t) if on_hit else a.blur
            top = a.blur_dive if a.blur_dive and moves[shot_of(t)][1] == 'dive' else a.blur_speed
            if top:                         # by speed, the hits' shutter as the floor: the modes add up
                sh = lambda V, sp: max(fixed, a.blur + (top - a.blur) * float(np.clip((np.percentile(sp, 90) - 5) / 13, 0, 1)))
            else:
                sh = fixed
            im = blur(prev[1] if same(prev) else None, im, nxt[1] if same(nxt) else None, sh)
        if a.look == 'festive':
            im = festive(im)
        elif a.look == 'frost':
            im = frost(im)
        elif a.look == 'kino':
            i = int(round(t * float(fps)))
            mask = people_mask(boxes, i, im.shape[:2], a.wide) if boxes else None
            kw = dict(lift=0.5, mix=0.5, glow_k=0.08, vig=0.08) if a.gentle else {}
            if a.lut_mix is not None:
                kw['mix'] = a.lut_mix
            im = kino(im, mask, lut, **kw)
        if pipe:
            pipe.stdin.write(np.ascontiguousarray(im).tobytes())
            return
        fr = av.VideoFrame.from_ndarray(im, format='rgb24')
        for p in ov.encode(fr):
            oc.mux(p)

    win = []                                # the last three frames: (time, rgb)
    if t_from > 1:
        ic.seek(int((t_from - 1) / vs.time_base), stream=vs, backward=True)
    for fr in ic.decode(vs):
        if fr.time < t_from - 0.2:
            continue
        if fr.time > t_to + 0.2:
            break
        win.append((float(fr.time), fr.to_ndarray(format='rgb24')))
        if len(win) == 2:
            emit(win[0], None, win[1])
        elif len(win) == 3:
            emit(win[1], win[0], win[2])
            win.pop(0)
    if len(win) == 2:
        emit(win[1], win[0], None)
    ic.close()
    if pipe:
        pipe.stdin.close()
        pipe.wait()
    else:
        for p in ov.encode():
            oc.mux(p)
    if oa is not None and not a.range:                      # the sound packets as they were
        ic = av.open(a.src)
        for p in ic.demux(ic.streams.audio[0]):
            if p.dts is None:
                continue
            p.stream = oa
            oc.mux(p)
        ic.close()
    if not pipe:
        oc.close()
    if SHUT:
        sh = np.array(SHUT)
        print(f'  shutter: median {np.median(sh):.2f}, over 1 frame in {np.mean(sh > 1) * 100:.0f} % of the frames, '
              f'max {sh.max():.2f}')
    print('written', a.out)


if __name__ == '__main__':
    main()
