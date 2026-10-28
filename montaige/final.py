"""Final render of an edit.py EDL from fresh Gyroflow renders of just the shots used.

    python final.py EDIT_edl.json OUT.mp4 [--width 1920 --height 1080 --fps 50] [--captions]

1. every shot (plus 0.6 s each side) becomes a Gyroflow trim range of its original clip,
   overlapping ranges are merged, and gf_render.py renders them at the output size; a clip stabilized
   before (a `_stabilized` export without its original) is cut from its own file, scaled;
2. each piece is aligned to the full-clip small render the EDL frames refer to, by
   comparing frames (Gyroflow's trim start is a few frames off the millisecond value);
3. the pieces are cut to the music, with a short dissolve where the recording day changes
   and an optional date caption at the start of each day; the EDL's LUT, colour match and grade (edit.py
   --lut, --match, --grade auto, keys through each shot) are applied on the GPU to 16-bit frames: log footage gets
   10-bit pieces (--pieces hevc10, the default with a LUT), so nothing bands before the LUT.
   Encoded with NVENC (hevc_nvenc by default). GPU only, run it with GPU_PY.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # the repository: embedia, montaige (run as a file)
from montaige import edit as bf  # noqa: E402
from embedia.read import stabilized_before  # noqa: E402
from embedia import pix  # noqa: E402
MARGIN_S, DISSOLVE_S = 1.0, 0.5
FADE_IN_S, FADE_OUT_S = 0.6, 2.0     # as in edit.py's preview, so an --end moment sits right before the fade
MONTHS = ['января', 'февраля', 'марта', 'апреля', 'мая', 'июня', 'июля', 'августа', 'сентября',
          'октября', 'ноября', 'декабря']


def thumbs(path, first, count, size=(64, 36)):
    """Grey thumbnails of frames first .. first+count-1 of a video."""
    c = av.open(path)
    st = c.streams.video[0]
    fps = float(st.average_rate)
    c.seek(int(max(first - 2, 0) / fps / st.time_base), stream=st, backward=True)
    out = {}
    for fr in c.decode(st):
        i = int(round(float(fr.pts * st.time_base) * fps))
        if i >= first + count:
            break
        if i >= first:
            out[i] = pix.ndarray(fr, format='gray', width=size[0], height=size[1]).astype(np.float32)
    c.close()
    return out


def motion_energy(th, first, count):
    """Mean |difference| between consecutive thumbnails, centre half only (zoom-tolerant)."""
    out = []
    for i in range(first, first + count):
        if i in th and i + 1 in th:
            a, b = th[i], th[i + 1]
            h, w = a.shape
            out.append(np.abs(a[h // 4:-h // 4, w // 4:-w // 4] - b[h // 4:-h // 4, w // 4:-w // 4]).mean())
        else:
            out.append(np.nan)
    return np.array(out)


def align(piece, proxy, expect, search=4, probe=60):
    """Proxy frame shown as frame 0 of the piece.

    Gyroflow starts a trim within a frame or two of the asked millisecond, but frames the shorter
    range with its own dynamic zoom, so pixels do not match; the rhythm of motion does. Returns
    (frame, correlation); falls back to `expect` when the motion is too even to tell."""
    a = motion_energy(thumbs(piece, 0, probe + 1), 0, probe)
    lo = max(expect - search, 0)
    th = thumbs(proxy, lo, probe + 2 * search + 2)
    best = (-2.0, expect)
    for k in range(lo, expect + search + 1):
        b = motion_energy(th, k, probe)
        ok = ~(np.isnan(a) | np.isnan(b))
        if ok.sum() < probe // 2 or a[ok].std() < 1e-6 or b[ok].std() < 1e-6:
            continue
        c = float(np.corrcoef(a[ok], b[ok])[0, 1])
        if c > best[0]:
            best = (c, k)
    return (best[1], best[0]) if best[0] > 0.6 else (expect, best[0])


class GpuGrade:
    """The EDL's LUT (trilinear in the 3D table), a clip's colour match (edit.py --match) and a shot's grade, on
    the GPU, for 8- or 16-bit RGB."""
    def __init__(self, lut_path):
        import torch
        from embedia import gpu
        self.torch, self.dev = torch, gpu.device()
        self.vol = None
        if lut_path:
            lut = bf.CubeLUT(lut_path)
            self.vol = torch.tensor(lut.raw).permute(3, 0, 1, 2)[None].contiguous().to(self.dev)   # (1, 3, b, g, r)
            self.dmin = torch.tensor(lut.dmin, dtype=torch.float32, device=self.dev)
            self.dmax = torch.tensor(lut.dmax, dtype=torch.float32, device=self.dev)

    def __call__(self, img, g, match=None, log=True):
        torch = self.torch
        x = torch.from_numpy(img.astype(np.float32) if img.dtype == np.uint16 else img).to(self.dev).float()
        x = x / (65535.0 if img.dtype == np.uint16 else 255.0)
        if self.vol is not None and log:                         # the EDL's flight not log footage (--lut-clips)
            grid = ((x - self.dmin) / (self.dmax - self.dmin) * 2 - 1)[None, None]    # (1, 1, H, W, rgb) -> (W, H, D)
            x = torch.nn.functional.grid_sample(self.vol, grid, mode='bilinear', padding_mode='border',
                                                align_corners=True)[0, :, 0].permute(1, 2, 0)
        v = x * 255.0
        if match:
            if isinstance(match, dict):                          # its luma curve (bf.apply_tone), then its colours
                cv = torch.tensor(match['curve'], dtype=torch.float32, device=self.dev)
                y = (v @ torch.tensor([0.2126, 0.7152, 0.0722], device=self.dev)).clamp(0, 255)
                lo = y.floor().long().clamp(max=254)
                k = torch.lerp(cv[lo], cv[lo + 1], y - lo) / y.clamp(min=1.0)
                v = (v * k[..., None]).clamp(0, 255)
                match = match['rgb']
            mt = torch.tensor(match, dtype=torch.float32, device=self.dev)
            v = v @ mt[:, :3].T + mt[:, 3]
        if g:
            v = (v - g[0]) * torch.tensor(g[1:], dtype=torch.float32, device=self.dev) + bf.GRADE_BLACK
        return v.clamp(0, 255).round().to(torch.uint8).cpu().numpy()


class Reader:
    """Sequential frame access to one piece; frames are asked for in increasing order.

    `crop` = (x0, y0) of the W x H window cut out of the (larger) piece; `scale` = (Wp, Hp) to scale every
    frame to first and `first` = the first frame wanted, for a clip stabilized before (its own file is the piece)."""
    def __init__(self, path, W, H, crop=(0, 0), scale=None, first=0):
        self.c = av.open(path)
        self.st = self.c.streams.video[0]
        self.st.thread_type = 'AUTO'
        self.fps = float(self.st.average_rate)
        if first > 2:
            self.c.seek(int((first - 2) / self.fps / self.st.time_base), stream=self.st, backward=True)
        self.it = self.c.decode(self.st)
        self.cur, self.img, self.W, self.H, self.crop, self.scale = -1, None, W, H, crop, scale
        self.prev = (-1, None)                                   # the frame before self.img, for at()

    def get(self, i):
        while self.cur < i:
            try:
                fr = next(self.it)
            except StopIteration:
                break
            n = int(round(float(fr.pts * self.st.time_base) * self.fps))
            if n >= i - 1 or self.img is None:
                x0, y0 = self.crop
                deep = '10' in fr.format.name or fr.format.name.startswith('p01')
                sw, sh = self.scale or (None, None)
                img = pix.ndarray(fr, width=sw, height=sh, format='rgb48le' if deep else 'rgb24', src_colorspace='itu709',
                                  dst_colorspace='itu709', src_color_range='MPEG', dst_color_range='JPEG')
                self.prev = (self.cur, self.img)
                self.img = np.ascontiguousarray(img[y0:y0 + self.H, x0:x0 + self.W])
            self.cur = n
        if self.prev[1] is not None and self.prev[0] >= i and self.cur > i:
            return self.prev[1]
        return self.img

    def at(self, pos, slow):
        """The picture at a fractional frame number: slower than real time, the two frames around it mixed
        (edit.py SLOW_BLEND, as in the preview); else the nearest frame."""
        if not slow:
            return self.get(int(round(pos)))
        i = int(np.floor(pos))
        a = self.get(i)
        b, w = self.get(i + 1), pos - i
        if b is a or w < 0.02:
            return a
        return (a.astype(np.float32) * (1.0 - w) + b.astype(np.float32) * w + 0.5).astype(a.dtype)


def cover_size(W, H, aspect):
    """The size of a picture of this aspect that covers W x H (the preview's crop, edit.py render): a 16:9 hand
    camera in a 4:3 final is cut from a picture H high, never stretched to W x H."""
    if abs(aspect - W / H) < 0.01:
        return W, H
    if aspect > W / H:
        return int(round(H * aspect / 2)) * 2, H
    return W, int(round(W / aspect / 2)) * 2


def piece_size(W, H, crop_bottom, crop_top, zoom=1.0):
    """Render size of a piece and the window to cut, so the cut window is W x H without scaling: the band
    between the bottom and top crops, then its centre 1/zoom (the EDL's zoom, as in the preview)."""
    z = zoom / (1.0 - crop_bottom - crop_top)
    Wp, Hp = int(round(W * z / 2)) * 2, int(round(H * z / 2)) * 2
    return Wp, Hp, ((Wp - W) // 2, int(round(crop_top * Hp + (H * zoom - H) / 2)))


def caption_layer(text, W, H):
    """RGBA overlay with the text bottom-left, soft shadow."""
    size = int(H * 0.065)
    font = None
    for name in ('segoeui.ttf', 'calibri.ttf', 'arial.ttf'):
        try:
            font = ImageFont.truetype(os.path.join(os.environ.get('WINDIR', 'C:/Windows'), 'Fonts', name), size)
            break
        except OSError:
            pass
    im = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    x, y = int(W * 0.05), int(H * 0.86) - size
    sh = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(sh).text((x + 2, y + 2), text, font=font, fill=(0, 0, 0, 170))
    im = Image.alpha_composite(im, sh.filter(ImageFilter.GaussianBlur(size / 10)))
    ImageDraw.Draw(im).text((x, y), text, font=font, fill=(255, 255, 255, 235))
    a = np.asarray(im).astype(np.float32)
    return a[..., :3], a[..., 3:] / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('edl')
    ap.add_argument('out')
    ap.add_argument('--width', type=int, default=1920)
    ap.add_argument('--height', type=int, default=1080)
    ap.add_argument('--fps', type=int, default=50)
    ap.add_argument('--bitrate', default='25M', help='of the Gyroflow pieces')
    ap.add_argument('--crf', type=int, default=19)
    ap.add_argument('--captions', action='store_true')
    ap.add_argument('--crop-bottom', type=float, default=0.0,
                    help='DJI clips: cut this share off the bottom (off by default: framing is Gyroflow'"'"'s job)')
    ap.add_argument('--crop-top', type=float, default=0.0)
    ap.add_argument('--zoom', type=float, default=None, help="centre cut 1/ZOOM (default: the EDL's, as in the preview)")
    ap.add_argument('--pieces', default=None, help='folder for the Gyroflow pieces (default: next to OUT)')
    ap.add_argument('--no-grade', action='store_true', help="ignore the EDL's LUT, colour match and per-shot grade")
    ap.add_argument('--pieces-codec', choices=('h264', 'hevc10'), default=None,
                    help='Gyroflow pieces: hevc10 (10-bit, the default with a LUT) or h264')
    ap.add_argument('--encoder', default='hevc_nvenc', help='hevc_nvenc, h264_nvenc or libx264')
    a = ap.parse_args()
    t_all = time.time()
    e = json.load(open(a.edl, encoding='utf-8'))
    flights = {f['idx']: f for f in e['flights']}
    shots = [dict(x, day=s.get('day'), scene=s['idx']) for s in e['scenes'] for x in s['shots']]
    pieces_dir = os.path.abspath(a.pieces or os.path.splitext(a.out)[0] + '_pieces')
    os.makedirs(pieces_dir, exist_ok=True)

    # 1. trim ranges per original, merged, rendered by Gyroflow
    by = {}
    d_s = max(DISSOLVE_S, float(e.get('dissolve') or 0.0))      # the margin a dissolve's pre-roll and tail need
    for k, x in enumerate(shots):
        f = flights[x['flight']]
        lo = x['anchors'][0] - MARGIN_S * f['fps'] - (d_s * f['fps'])
        hi = x['anchors'][-1] + MARGIN_S * f['fps'] + (d_s * f['fps'])
        by.setdefault(x['flight'], []).append([max(lo, 0.0), hi, [k]])
    ranges, owner, owner_lo = {}, {}, {}
    for fi, rr in by.items():
        rr.sort()
        merged = []
        for r in rr:
            if merged and r[0] <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], r[1])
                merged[-1][2] += r[2]
            else:
                merged.append(r)
        f = flights[fi]
        ranges[f['original']] = [[round(lo / f['fps'] * 1000.0), round(hi / f['fps'] * 1000.0)] for lo, hi, _ in merged]
        for j, (lo, hi, ks) in enumerate(merged):
            owner_lo[(fi, j)] = lo
            for k in ks:
                owner[k] = (fi, j, lo)
    rj = os.path.join(pieces_dir, 'ranges.json')
    with open(rj, 'w', encoding='utf-8') as fh:
        json.dump(ranges, fh, ensure_ascii=False, indent=1)
    stem = lambda fi: os.path.splitext(os.path.basename(flights[fi]['original']))[0]
    def piece(fi, j):
        if stabilized_before(flights[fi]['original']):       # stabilized before: cut from the clip itself
            return flights[fi]['original']
        # Gyroflow numbers the trims (-001, -002 ...) only when a clip has more than one
        if len(ranges[flights[fi]['original']]) == 1:
            return os.path.join(pieces_dir, f'{stem(fi)}_stabilized.mp4')
        return os.path.join(pieces_dir, f'{stem(fi)}_stabilized-{j + 1:03d}.mp4')
    is_dji = lambda c: os.path.basename(c).upper().startswith('DJI_')
    zoom = a.zoom if a.zoom is not None else e.get('zoom') or 1.0
    def size_of(c):
        """(Wp, Hp, (x0, y0)): the piece's render size and where the final's W x H window sits in it."""
        with av.open(c) as cc:
            ctx = cc.streams.video[0].codec_context
            Wc, Hc = cover_size(a.width, a.height, ctx.width / ctx.height)
        cb, ct = (a.crop_bottom, a.crop_top) if is_dji(c) else (0.0, 0.0)
        Wp, Hp, (x0, y0) = piece_size(Wc, Hc, cb, ct, zoom)
        return Wp, Hp, (x0 + (Wc - a.width) // 2, y0 + (Hc - a.height) // 2)
    sizes = {c: size_of(c) for c in ranges}
    for Wp, Hp in sorted({s[:2] for c, s in sizes.items() if not stabilized_before(c)}):
        todo = {c: r for c, r in ranges.items() if sizes[c][:2] == (Wp, Hp) and not stabilized_before(c) and
                not all(os.path.exists(piece(fi, j)) for fi in flights if flights[fi]['original'] == c for j in range(len(r)))}
        if not todo:
            continue
        with open(rj + '.todo', 'w', encoding='utf-8') as fh:
            json.dump(todo, fh, ensure_ascii=False)
        print(f'gyroflow: {sum(len(r) for r in todo.values())} pieces from {len(todo)} clips at {Wp}x{Hp}', flush=True)
        t0 = time.time()
        # H.264 on NVENC stops at 4096 px: a 4K 4:3 final with --zoom 1.16 needs 4454 x 3340 pieces
        codec = a.pieces_codec or ('hevc10' if (e.get('lut') and not a.no_grade) or max(Wp, Hp) > 4096 else 'h264')
        from embedia import run as er
        subprocess.run(er.command('gf_render', [pieces_dir, '--ranges', rj + '.todo', '--width', Wp, '--height', Hp,
                                                '--bitrate', a.bitrate, '--codec', codec]),
                       check=True, cwd=er.ROOT, env=er.env())
        print(f'gyroflow done in {time.time() - t0:.0f}s', flush=True)

    # 2. where each piece starts on the small render's frame grid
    starts = {}
    for k, (fi, j, lo) in sorted(owner.items()):
        if (fi, j) in starts:
            continue
        if stabilized_before(flights[fi]['original']):       # the proxy is this file, frame for frame
            starts[(fi, j)] = 0
            continue
        asked = int(round(round(lo / flights[fi]['fps'] * 1000.0) / 1000.0 * flights[fi]['fps']))
        k0, cor = align(piece(fi, j), flights[fi]['render'], asked)
        starts[(fi, j)] = k0
        print(f'  {os.path.basename(piece(fi, j))}: starts at frame {k0} (asked {asked}, motion correlation {cor:.2f})', flush=True)

    # 3. cut to the music
    W, H, fps = a.width, a.height, a.fps
    dur = max(x['t1'] for x in shots)
    oc = av.open(a.out, 'w')
    vs = oc.add_stream(a.encoder, rate=fps)
    vs.width, vs.height, vs.pix_fmt = W, H, 'yuv420p'
    vs.options = ({'crf': str(a.crf), 'preset': 'faster'} if a.encoder.startswith('lib') else
                  {'rc': 'vbr', 'cq': str(a.crf), 'preset': 'p5', 'b': '0'})
    for key in ('colorspace', 'color_primaries', 'color_trc'):
        try:
            setattr(vs.codec_context, key, 1)
        except (AttributeError, TypeError, ValueError):
            pass
    au = oc.add_stream('aac', rate=48000)
    au.layout, au.bit_rate = 'stereo', 192000
    bf.music_audio(oc, au, e['music'], e.get('bites') or ())  # the music, with the clips' own sound at the bites

    readers = {}
    grade = GpuGrade(None if a.no_grade else e.get('lut'))

    def frame_of(k, t):
        """Picture of shot k at music time t (t may run past the shot's ends by the margins)."""
        x = shots[k]
        fi, j, _ = owner[k]
        if (fi, j) not in readers:
            Wp, Hp, crop = sizes[flights[fi]['original']]
            if stabilized_before(flights[fi]['original']):
                readers[(fi, j)] = Reader(piece(fi, j), W, H, crop, scale=(Wp, Hp), first=int(owner_lo[(fi, j)]))
            else:
                readers[(fi, j)] = Reader(piece(fi, j), W, H, crop)
        u = np.array(x['anchor_t'])
        v = np.array(x['anchors'])
        sp = (v[-1] - v[0]) / max(u[-1] - u[0], 1e-9)

        def src_at(tt):
            if tt < u[0]:
                return v[0] + (tt - u[0]) * sp
            if tt > u[-1]:
                return v[-1] + (tt - u[-1]) * sp
            return float(np.interp(tt, u, v))
        src = src_at(t - x['t0'])
        rd = readers[(fi, j)]                                    # x real time here, for mixing frames
        slow = (src_at(t - x['t0'] + 0.02) - src_at(t - x['t0'] - 0.02)) / 0.04 < bf.SLOW_BLEND * rd.fps
        img = rd.at(max(src - starts[(fi, j)], 0.0), slow)
        g = None if a.no_grade else bf.grade_at(x.get('grade'), (src - x['src_frame']) / max(x['src_frames'], 1))
        return grade(img, g, None if a.no_grade else flights[fi].get('match'), flights[fi].get('log', True))

    # a chapter is a day, or a session of one ('2022-08-06 19:23', edit.py --sessions): a caption per day
    dates = [(x['day'] or '')[:10] for x in shots]
    day_starts = [k for k in range(len(shots)) if k == 0 or dates[k] != dates[k - 1]]
    # a dissolve across a change of day (DISSOLVE_S), and into the scenes the EDL marks (edit.py --dissolve S)
    dlen = {k: DISSOLVE_S for k in day_starts if k > 0}
    k_shot = 0
    for s in e['scenes']:
        if s.get('dissolve_in') and k_shot > 0:
            dlen[k_shot] = max(dlen.get(k_shot, 0.0), float(e.get('dissolve') or DISSOLVE_S))
        k_shot += len(s['shots'])
    caps = {}
    if a.captions:
        for k in day_starts:
            d = datetime.date.fromisoformat(shots[k]['day'][:10])
            caps[k] = caption_layer(f'{d.day} {MONTHS[d.month - 1]}', W, H)
    nout = int(np.floor(dur * fps))
    k = 0
    t_start = time.time()
    for jj in range(nout):
        t = jj / fps
        while k + 1 < len(shots) and t >= shots[k + 1]['t0']:
            k += 1
        img = frame_of(k, t)
        # a dissolve centred on the cut: across a change of day, or into a scene the plan marks
        for kk in (k, k + 1):
            if 0 < kk < len(shots) and kk in dlen:
                tc, D = shots[kk]['t0'], dlen[kk]
                if abs(t - tc) < D / 2:
                    w_in = (t - tc) / D + 0.5
                    other = frame_of(kk - 1 if kk == k else kk, t).astype(np.float32)
                    img = img.astype(np.float32) * (w_in if kk == k else 1 - w_in) + other * (1 - w_in if kk == k else w_in)
        for kc, (rgb, al) in caps.items():
            tc = shots[kc]['t0'] + 0.6
            g = np.clip(min((t - tc) / 0.6, (tc + 3.4 - t) / 0.6), 0, 1)
            if g > 0:
                img = img.astype(np.float32) * (1 - al * g) + rgb * (al * g)
        g = min(1.0, t / FADE_IN_S, (dur - t) / FADE_OUT_S)
        if g < 1:
            img = img.astype(np.float32) * max(g, 0.0)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        vf = pix.reformat(av.VideoFrame.from_ndarray(img, format='rgb24'), format='yuv420p', src_colorspace='itu709',
                          dst_colorspace='itu709', src_color_range='JPEG', dst_color_range='MPEG')
        vf.pts = jj
        for p in vs.encode(vf):
            oc.mux(p)
        if jj % 250 == 0:
            print(f'\r  encode {t:6.1f}/{dur:.1f}s ({time.time() - t_start:.0f}s)', end='', flush=True)
    for p in vs.encode(None):
        oc.mux(p)
    oc.close()
    print(f'\ndone in {time.time() - t_all:.0f}s -> {a.out}')


if __name__ == '__main__':
    main()
