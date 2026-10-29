"""Where and how fast a GoPro clip was shot: the GPS in its GPMF track (GPS5, or GPS9 from HERO11 on).

    python gps.py CLIP.MP4 [...]      one line per clip: fixes, the first and the middle point, altitude
    python gps.py track --list originals.txt --proxy-dir DIR    -> DIR/<clip>_stabilized_gps.npz per clip

A place named from the picture is a guess (a turquoise river in a canyon read as the Sulak canyon; the GPS put
it 30 km south, by the Irganay reservoir): check names against this before they go into captions or titles.
Clips stabilized elsewhere have lost the track; a clip of the same session tells where it was.

track   every GPS point (about 18 a second) with its time in the clip: lat, lon, alt (m), speed (2D, m/s),
        climb (m/s: altitude smoothed over CLIMB_S, + up), fix and DOP; points without a fix are kept with
        fix 0. Speed is what the action of a camera carried by a person is (edit.py --speed: skiing down a
        slope, a bike, a car), climb tells a lift going up from a run going down. One line per clip: the
        share of time with a fix, speed p50 / p90 / max, the altitude range and the metres up and down.
"""
import argparse
import os
import struct
import sys

import numpy as np

if not __package__:                               # run as a file: the package of its folder (PEP 366)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
from .mp4parse import open_mp4  # noqa: E402

# climb: altitude smoothed over this before the derivative (GPS altitude wobbles by metres from fix to fix)
CLIMB_S = 4.0
MAX_DOP = 5.0           # GoPro: under 5 is a good fix
JUMP_M, JUMP_S = 30.0, 3.0


def klv(buf):
    """GPMF entries: (key, type, struct size, repeat, data)."""
    pos = 0
    while pos + 8 <= len(buf):
        key, typ = buf[pos:pos + 4], chr(buf[pos + 4])
        size, rep = buf[pos + 5], struct.unpack('>H', buf[pos + 6:pos + 8])[0]
        n = size * rep
        yield key, typ, size, rep, buf[pos + 8:pos + 8 + n]
        pos += 8 + ((n + 3) & ~3)


def gps_points(buf, every=False):
    """(lat, lon, alt m, speed 2D m/s, fix, DOP) per GPS entry of one GPMF sample: the first of each stream, or
    every entry (about 18 a second)."""
    out = []

    def walk(b):
        scal, fix, dop = None, None, 0.0
        for key, typ, size, rep, data in klv(b):
            if typ == '\x00':                                    # a nested container (DEVC, STRM)
                walk(data)
            elif key == b'SCAL':
                fmt = {'l': 'i', 'L': 'I', 's': 'h', 'S': 'H'}.get(typ)
                if fmt:
                    scal = struct.unpack('>%d%s' % (len(data) // struct.calcsize(fmt), fmt), data)
            elif key == b'GPSF':
                fix = struct.unpack('>I', data[:4])[0]
            elif key == b'GPSP':
                dop = struct.unpack('>H', data[:2])[0] / 100.0
            elif key == b'GPS5' and scal and size == 20:
                sc = scal if len(scal) >= 5 else scal * 5
                for r in range(rep if every else min(rep, 1)):
                    v = struct.unpack('>5i', data[20 * r:20 * r + 20])
                    out.append((v[0] / sc[0], v[1] / sc[1], v[2] / sc[2], v[3] / sc[3], fix, dop))
            elif key == b'GPS9' and scal and size == 32 and len(scal) >= 9:
                # lat, lon, alt, speed 2D, speed 3D, days since 2000, seconds of the day, DOP, fix: per entry
                for r in range(rep if every else min(rep, 1)):
                    v = struct.unpack('>7i2H', data[32 * r:32 * r + 32])
                    out.append((v[0] / scal[0], v[1] / scal[1], v[2] / scal[2], v[3] / scal[3], v[8], v[7] / scal[7]))
    walk(buf)
    return out


def gpmf_stream(path, want):
    """Every sample of one 3-axis int16 GPMF stream of a clip (b'ACCL' m/s2, b'GYRO' rad/s), scaled, and its rate
    (samples a second); (None, 0) without it."""
    f, _, _, tracks = open_mp4(path)
    rows = []

    def walk(b):
        scal = 1.0
        for key, typ, size, rep, data in klv(b):
            if typ == '\x00':
                walk(data)
            elif key == b'SCAL' and typ in 'lLsS':
                fmt = {'l': 'i', 'L': 'I', 's': 'h', 'S': 'H'}[typ]
                scal = np.array(struct.unpack('>%d%s' % (len(data) // struct.calcsize(fmt), fmt), data), float)
            elif key == want and typ == 's' and size == 6:
                rows.append(np.frombuffer(data[:6 * rep], '>i2').reshape(rep, 3) / scal)
    try:
        tr = next((t for t in tracks if t.formats and t.formats[0] == b'gpmd'), None)
        if tr is None:
            return None, 0.0
        for _, _, data in tr.samples(f):
            walk(data)
        dur = tr.duration / tr.timescale
    finally:
        f.close()
    if not rows:
        return None, 0.0
    a = np.vstack(rows)
    return a, len(a) / dur


def clip_gps(path):
    """GPS points of a clip with a 2D or 3D fix, one per GPMF sample (about a second)."""
    f, _, _, tracks = open_mp4(path)
    try:
        tr = next((t for t in tracks if t.formats and t.formats[0] == b'gpmd'), None)
        if tr is None:
            return None
        pts = [p for _, _, data in tr.samples(f) for p in gps_points(data)]
    finally:
        f.close()
    return [(p[0], p[1], p[2], p[4]) for p in pts if p[4] and p[4] >= 2 and abs(p[0]) > 1e-3]


def clip_track(path):
    """Every GPS point of a clip with its time (s from the start of the clip), or None without a GPMF track. The
    entries of one GPMF sample are spread evenly over that sample's duration."""
    f, _, _, tracks = open_mp4(path)
    try:
        tr = next((t for t in tracks if t.formats and t.formats[0] == b'gpmd'), None)
        if tr is None:
            return None
        durs = [d for c, d in tr.stts for _ in range(c)]
        rows = []
        for i, (t0, _, data) in enumerate(tr.samples(f)):
            pts = gps_points(data, every=True)
            dur = durs[i] if i < len(durs) else durs[-1]
            for j, p in enumerate(pts):
                rows.append(((t0 + dur * j / len(pts)) / tr.timescale,) + p)
    finally:
        f.close()
    a = np.array(rows, float).reshape(-1, 7)
    t, lat, lon, alt, speed, fix, dop = a.T
    ok = (fix >= 2) & (np.abs(lat) > 1e-3) & ((dop == 0) | (dop <= MAX_DOP))
    # the first fixes after power-on can be far off (127 m for 1671 m a second later, DOP 4.9 throughout): what
    # comes before a jump of JUMP_M within the first JUMP_S of fixes is no fix; later, a point JUMP_M off the
    # median altitude of its JUMP_S neighbourhood is none either
    if ok.sum() > 2:
        from scipy.ndimage import median_filter
        i = np.flatnonzero(ok)
        j = np.flatnonzero((np.abs(np.diff(alt[i])) > JUMP_M) & (t[i[1:]] - t[i[0]] < JUMP_S))
        if j.size:
            ok[i[:j[-1] + 1]] = False
    if ok.sum() > 2:
        i = np.flatnonzero(ok)
        rate = len(i) / max(t[i[-1]] - t[i[0]], 1e-6)
        med = median_filter(alt[i], size=max(int(JUMP_S * rate) | 1, 3), mode='nearest')
        ok[i[np.abs(alt[i] - med) > JUMP_M]] = False
    fix = np.where(ok, fix, 0)
    climb = np.zeros(len(t))
    if ok.sum() > 2:
        ti, ai = t[ok], alt[ok]
        rate = len(ti) / max(ti[-1] - ti[0], 1e-6)
        k = max(int(CLIMB_S * rate), 1)
        sm = np.convolve(np.pad(ai, k // 2, mode='edge'), np.ones(k) / k, 'valid')[:len(ai)]
        climb[ok] = np.gradient(sm, ti)
    return dict(t=t, lat=lat, lon=lon, alt=alt, speed=speed, climb=climb, fix=fix, dop=dop)


def summary(name, z, duration):
    ok = z['fix'] > 0
    if not ok.any():
        return f'{name}: no GPS fix'
    s, a, t = z['speed'][ok] * 3.6, z['alt'][ok], z['t'][ok]
    da = np.diff(np.convolve(a, np.ones(9) / 9, 'valid'))
    share = (t[-1] - t[0]) / max(duration, 1e-6) if len(t) > 1 else 0.0
    return (f'{name}: fix {share * 100:3.0f} % of {duration:5.0f} s; speed p50 {np.median(s):4.1f}, p90 '
            f'{np.percentile(s, 90):4.1f}, max {s.max():4.1f} km/h; altitude {a.min():.0f}-{a.max():.0f} m, '
            f'up {da[da > 0].sum():.0f} m, down {-da[da < 0].sum():.0f} m')


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'track':
        ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        ap.add_argument('step', choices=('track',))
        ap.add_argument('clips', nargs='*')
        ap.add_argument('--list', default=None, help='a file of clip paths, one a line (non-ASCII paths)')
        ap.add_argument('--proxy-dir', required=True)
        ap.add_argument('--force', action='store_true')
        a = ap.parse_args()
        clips = list(a.clips)
        if a.list:
            with open(a.list, encoding='utf-8') as fh:
                clips += [x.strip() for x in fh if x.strip()]
        import av
        for c in clips:
            stem = os.path.splitext(os.path.basename(c))[0]
            dst = os.path.join(a.proxy_dir, stem + '_stabilized_gps.npz')
            if os.path.exists(dst) and not a.force:
                z = dict(np.load(dst))
            else:
                z = clip_track(c)
                if z is None:
                    print(f'{stem}: no GPMF track')
                    continue
                np.savez(dst, **z)
            with av.open(c) as ct:
                dur = float(ct.duration) / 1e6
            print(summary(stem, z, dur), flush=True)
        return
    for path in sys.argv[1:]:
        pts = clip_gps(path)
        name = os.path.basename(path)
        if pts is None:
            print(f'{name}: no GPMF track')
        elif not pts:
            print(f'{name}: no GPS fix')
        else:
            mid = pts[len(pts) // 2]
            print(f'{name}: {len(pts)} fixes; first {pts[0][0]:.5f}, {pts[0][1]:.5f}; middle {mid[0]:.5f}, {mid[1]:.5f}, '
                  f'{mid[2]:.0f} m')


if __name__ == '__main__':
    main()
