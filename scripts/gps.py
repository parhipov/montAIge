"""Where a GoPro clip was shot: the GPS in its GPMF track (GPS5), when the camera had a fix.

    python gps.py CLIP.MP4 [...]      one line per clip: fixes, the first and the middle point, altitude

A place named from the picture is a guess (a turquoise river in a canyon read as the Sulak canyon; the GPS put
it 30 km south, by the Irganay reservoir): check names against this before they go into captions or titles.
Clips stabilized elsewhere have lost the track; a clip of the same session tells where it was.
"""
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mp4parse import open_mp4  # noqa: E402


def klv(buf):
    """GPMF entries: (key, type, struct size, repeat, data)."""
    pos = 0
    while pos + 8 <= len(buf):
        key, typ = buf[pos:pos + 4], chr(buf[pos + 4])
        size, rep = buf[pos + 5], struct.unpack('>H', buf[pos + 6:pos + 8])[0]
        n = size * rep
        yield key, typ, size, rep, buf[pos + 8:pos + 8 + n]
        pos += 8 + ((n + 3) & ~3)


def gps_points(buf):
    """(lat, lon, alt m, fix) of the first GPS5 entry of each stream in one GPMF sample."""
    out = []

    def walk(b):
        scal, fix = None, None
        for key, typ, size, rep, data in klv(b):
            if typ == '\x00':                                    # a nested container (DEVC, STRM)
                walk(data)
            elif key == b'SCAL':
                fmt = {'l': 'i', 'L': 'I', 's': 'h', 'S': 'H'}.get(typ)
                if fmt:
                    scal = struct.unpack('>%d%s' % (len(data) // struct.calcsize(fmt), fmt), data)
            elif key == b'GPSF':
                fix = struct.unpack('>I', data[:4])[0]
            elif key == b'GPS5' and scal and len(data) >= 20:
                v = struct.unpack('>5i', data[:20])
                out.append((v[0] / scal[0], v[1] / scal[1], v[2] / scal[2], fix))
    walk(buf)
    return out


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
    return [p for p in pts if p[3] and p[3] >= 2 and abs(p[0]) > 1e-3]


def main():
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
