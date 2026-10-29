"""Image motion features and look thumbnails of small stabilized renders (any aspect).

    python -m embedia.analyzers.analyze RENDER.mp4 [...] [--workers N]   ->  <render>_flow.npz, <render>_look.npz next to each

Flow features at half the frame rate (every 2nd frame), from 160-px-wide grey frames:
  flow      mean |optical flow| px per frame pair
  resid50/90  |flow - best global affine|: what a camera rotation cannot explain -> parallax,
            i.e. something close to the camera moving fast past it
  div       divergence of the affine (forward motion / approach)
  bottom    mean |flow| in the lower quarter (ground rushing by)
  bright, sharp (Laplacian variance)
Look: Lab thumbnail of every 5th frame, 12 cells high, width from the aspect.
Clips are analysed --workers at a time (processes: the flow is single-threaded CPU work; default half the CPUs).
"""
import argparse
import multiprocessing
import os
import time

import av
import cv2
import numpy as np

from ..media import pix

LOOK_STEP = 5


def analyze(src):
    """The flow and look files of one render; returns the line to print."""
    base = os.path.splitext(src)[0]
    if os.path.exists(base + '_flow.npz') and os.path.exists(base + '_look.npz'):
        return f'{os.path.basename(src)} done before'
    t0 = time.time()
    c = av.open(src)
    st = c.streams.video[0]
    st.thread_type = 'AUTO'
    fps = float(st.average_rate)
    W, H = st.codec_context.width, st.codec_context.height
    gw = 160
    gh = int(round(gw * H / W / 2)) * 2
    lw = int(round(9 * W / H))
    ys, xs = np.mgrid[2:gh:4, 2:gw:4]
    X = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)], 1).astype(np.float64)
    bottom = ys.ravel() >= gh * 3 // 4
    rows, look_fr, looks = [], [], []
    prev = None
    n = 0
    for fr in c.decode(st):
        idx = int(round(float(fr.pts * st.time_base) * fps))
        n += 1
        if idx % 2 and idx % LOOK_STEP:
            continue
        bgr = pix.ndarray(fr, format='bgr24', width=gw, height=gh)
        if idx % LOOK_STEP == 0:
            t = cv2.resize(bgr, (lw, 9), interpolation=cv2.INTER_AREA)
            look_fr.append(idx)
            looks.append(cv2.cvtColor(t, cv2.COLOR_BGR2LAB).astype(np.float32).reshape(-1))
        if idx % 2:
            continue
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            fl = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 13, 3, 5, 1.1, 0)
            F = fl[2:gh:4, 2:gw:4].reshape(-1, 2).astype(np.float64)
            A, *_ = np.linalg.lstsq(X, F, rcond=None)
            r = np.linalg.norm(F - X @ A, axis=1)
            mag = np.linalg.norm(F, axis=1)
            rows.append((idx, mag.mean(), np.percentile(r, 50), np.percentile(r, 90), A[0, 0] + A[1, 1],
                         mag[bottom].mean(), g.mean(), cv2.Laplacian(g, cv2.CV_32F).var()))
        prev = g
    c.close()
    a = np.array(rows)
    cols = ('frame', 'flow', 'resid50', 'resid90', 'div', 'bottom', 'bright', 'sharp')
    np.savez(base + '_flow.npz', **{k: a[:, i] for i, k in enumerate(cols)}, fps=fps, n=n)
    np.savez(base + '_look.npz', frame=np.array(look_fr), desc=np.array(looks, np.float32), step=LOOK_STEP)
    return f'{os.path.basename(src)}: {n} frames, {len(rows)} flow rows, {len(looks)} looks in {time.time() - t0:.0f}s'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('renders', nargs='+')
    ap.add_argument('--workers', type=int, default=None, help='clips at once (default: half the CPUs)')
    a = ap.parse_args()
    todo = [r for r in a.renders if not (os.path.exists(os.path.splitext(r)[0] + '_flow.npz') and
                                         os.path.exists(os.path.splitext(r)[0] + '_look.npz'))]
    for r in a.renders:
        if r not in todo:
            print(os.path.basename(r), 'done before', flush=True)
    w = a.workers or max(1, min(len(todo), (os.cpu_count() or 2) // 2))
    if w <= 1:
        for r in todo:
            print(analyze(r), flush=True)
        return
    with multiprocessing.Pool(w) as pool:
        for line in pool.imap_unordered(analyze, todo):
            print(line, flush=True)


if __name__ == '__main__':
    main()
