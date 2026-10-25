"""The owner's own exports in a footage folder, found in the originals: often the best is already picked there.

    python exports.py FOLDER [...] --originals out/JOB/originals.txt --proxy-dir out/diagnostics/proxy_JOB

Every video in the folders that is not one of the originals (an old Gyroflow export, a named edit; not
the goggle recordings DJIG*, not the parts _* of a joined recording) is cut into pieces at its cuts, and
each piece of 2 s or more is looked for in the whole-clip renders of the originals: grey thumbnails of
the centre of the frame at 10 Hz, the export's aspect matched (a 16:9 export of a 4:3 clip is its middle
three quarters in height), best mean correlation over every offset. Prints each piece's place with a
ready --must, and the pieces nothing matched. Village: 0037 72.8-107.8 s at 0.85 against 0.13 for the
next best place.
"""
import argparse
import os

import av
import numpy as np

HZ = 10.0
CUT_JUMP = 3.0      # a cut: thumbnail change this many times the export's median change, and over 12 (0-255)
MIN_PIECE_S = 2.0
MATCH = 0.5         # mean correlation for a piece to count as found


def thumbs(path, crop_w=0.5, crop_h=0.5):
    """(n x 192) z-scored grey 16x12 thumbnails of the frame centre at HZ, and the frame aspect."""
    out = []
    with av.open(path) as c:
        st = c.streams.video[0]
        st.thread_type = 'AUTO'
        fps = float(st.average_rate)
        aspect = st.codec_context.width / st.codec_context.height
        step = max(int(round(fps / HZ)), 1)
        for i, f in enumerate(c.decode(st)):
            if i % step:
                continue
            g = f.to_ndarray(format='gray', width=320, height=max(int(round(320 / aspect)), 12)).astype(np.float32)
            h, w = g.shape
            g = g[int(h * (1 - crop_h) / 2):int(h * (1 + crop_h) / 2), int(w * (1 - crop_w) / 2):int(w * (1 + crop_w) / 2)]
            h, w = g.shape
            out.append(g[:h - h % 12, :w - w % 16].reshape(12, h // 12, 16, w // 16).mean(axis=(1, 3)).ravel())
    t = np.array(out)
    return t, aspect, fps / step


def z(x):
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)


def pieces(t):
    """[(first, last+1)] sample ranges between cuts, MIN_PIECE_S or longer."""
    d = np.abs(np.diff(t, axis=0)).mean(1)
    cuts = np.flatnonzero((d > CUT_JUMP * np.median(d)) & (d > 12.0)) + 1
    edges = [0] + list(cuts) + [len(t)]
    return [(a, b) for a, b in zip(edges, edges[1:]) if b - a >= MIN_PIECE_S * HZ]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folders', nargs='+')
    ap.add_argument('--originals', required=True)
    ap.add_argument('--proxy-dir', required=True, help='whole-clip renders <original>_stabilized.mp4')
    a = ap.parse_args()
    originals = [x.strip() for x in open(a.originals, encoding='utf-8') if x.strip()]
    names = {os.path.normcase(os.path.abspath(o)) for o in originals}
    exports = []
    for folder in a.folders:
        for d, _, ns in os.walk(folder):
            for n in sorted(ns):
                p = os.path.join(d, n)
                if (n.lower().endswith(('.mp4', '.mov')) and os.path.normcase(os.path.abspath(p)) not in names
                        and not n.startswith('_') and not n.upper().startswith('DJIG')):
                    exports.append(p)
    if not exports:
        print('no exports beside the originals')
        return
    renders = {}
    for o in originals:
        r = os.path.join(a.proxy_dir, os.path.splitext(os.path.basename(o))[0] + '_stabilized.mp4')
        if os.path.exists(r):
            renders[o] = r
    cache = {}
    for ex in exports:
        te, ae, _ = thumbs(ex)
        found = []
        for p0, p1 in pieces(te):
            q = z(te[p0:p1])
            best = (-1.0, None, 0)
            second = -1.0
            for o, r in renders.items():
                if r not in cache:
                    cache[r] = thumbs(r)
                tr, ar, _ = cache[r]
                if ar != ae and r + str(ae) not in cache:                # the export's share of the render frame
                    cw, ch = (0.5, 0.5 * ar / ae) if ae > ar else (0.5 * ae / ar, 0.5)
                    cache[r + str(ae)] = thumbs(r, cw, ch)
                R = z((cache[r + str(ae)] if ar != ae else cache[r])[0])
                n = len(q)
                if len(R) < n:
                    continue
                C = q @ R.T / q.shape[1]
                score = np.array([C[np.arange(n), k + np.arange(n)].mean() for k in range(len(R) - n + 1)])
                k = int(np.argmax(score))
                far = np.abs(np.arange(score.size) - k) > 2 * HZ
                if score[k] > best[0]:
                    second = max(best[0], score[far].max() if far.any() else -1.0)
                    best = (float(score[k]), o, k)
                else:
                    second = max(second, float(score[k]))
            s, o, k = best
            found.append([p0, p1, o if s >= MATCH else None, k, k + p1 - p0, s, second])
        # pieces that run on in the same original are one piece (a flip or a fast turn looks like a cut)
        merged = []
        for f in found:
            g = merged[-1] if merged else None
            if g and f[2] and g[2] == f[2] and abs(f[3] - g[4]) <= HZ:
                g[1], g[4], g[5], g[6] = f[1], f[4], min(g[5], f[5]), max(g[6], f[6])
            else:
                merged.append(f)
        print(f'{os.path.basename(ex)}: {len(te) / HZ:.1f} s, {len(merged)} pieces')
        for p0, p1, o, k0, k1, s, second in merged:
            where = f'{p0 / HZ:.1f}-{p1 / HZ:.1f}s'
            if o is None:
                print(f'   {where}: not found in the originals (best {s:.2f})')
                continue
            stem = os.path.splitext(os.path.basename(o))[0]
            whole = (k1 - k0) >= 0.9 * len(cache[renders[o]][0])
            print(f'   {where} = {stem} {k0 / HZ:.1f}-{k1 / HZ:.1f}s (corr {s:.2f}, next best {second:.2f})' +
                  ('   the whole clip: a stabilized copy, not a pick' if whole else
                   f'   --must {stem}:{k0 / HZ:.1f}-{k1 / HZ:.1f}'))


if __name__ == '__main__':
    main()
