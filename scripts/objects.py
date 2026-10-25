"""What is in the picture: YOLO (COCO) on the small renders, a few times a second, reduced to numbers.

    python objects.py --weights models/yolo/yolov8s.pt RENDER.mp4 [...]   ->  <render>_objects.npz
    python objects.py --summary RENDER.mp4 [...]                           (no detection, just the report)

Per sampled frame and per group (people, vehicles, animals): how many, and the share of the frame
the biggest one covers. Nothing per box is kept. On top of that, people_motion() turns the size of
the biggest person into events: the quad flies up to someone (the box grows fast), pulls away
(it shrinks fast), or holds a close-up. The report is one line per clip and the best moments;
the raw detections are never printed.
"""
import argparse
import os
import time

import av
import numpy as np

GROUPS = {
    'person': {0},
    'vehicle': {1, 2, 3, 5, 6, 7, 8},          # bicycle car motorcycle bus train truck boat
    'animal': {14, 15, 16, 17, 18, 19},        # bird cat dog horse sheep cow
}
APPROACH_RATE = 0.6      # |d ln(area) / dt| per second that counts as flying up to / away from someone
CLOSE_AREA = 0.04        # a person covering this share of the frame is a close-up
MIN_AREA = 0.002         # smaller people are too far to tell anything


def detect(model, src, every, conf):
    c = av.open(src)
    st = c.streams.video[0]
    st.thread_type = 'AUTO'
    fps = float(st.average_rate)
    frames, imgs = [], []
    for fr in c.decode(st):
        i = int(round(float(fr.pts * st.time_base) * fps))
        if i % every == 0:
            frames.append(i)
            imgs.append(fr.to_ndarray(format='bgr24'))
    c.close()
    H, W = imgs[0].shape[:2]
    rec = {f'{g}_{k}': np.zeros(len(imgs)) for g in GROUPS for k in ('n', 'area')}
    for b0 in range(0, len(imgs), 16):
        for j, r in enumerate(model.predict(imgs[b0:b0 + 16], imgsz=640, conf=conf, verbose=False)):
            cls = r.boxes.cls.cpu().numpy().astype(int)
            wh = r.boxes.xywh.cpu().numpy()[:, 2:4] if len(cls) else np.zeros((0, 2))
            area = wh[:, 0] * wh[:, 1] / (W * H)
            for g, ids in GROUPS.items():
                m = np.isin(cls, list(ids))
                rec[f'{g}_n'][b0 + j] = m.sum()
                rec[f'{g}_area'][b0 + j] = area[m].max() if m.any() else 0.0
    return dict(frame=np.array(frames), fps=fps, **rec)


def people_motion(z):
    """Events from the biggest person's size over time: list of (frame, kind, strength).

    kind: 'approach' (box grows by more than APPROACH_RATE in ln-area per second), 'recede'
    (shrinks as fast), 'close' (covers CLOSE_AREA or more). Samples without a person break runs."""
    fr, fps = z['frame'], float(z['fps'])
    area = np.where(z['person_n'] > 0, z['person_area'], 0.0)
    ok = area >= MIN_AREA
    la = np.log(np.maximum(area, MIN_AREA))
    dt = np.diff(fr) / fps
    rate = np.diff(la) / np.maximum(dt, 1e-6)
    both = ok[1:] & ok[:-1]
    ev = []
    for i in np.flatnonzero(both & (np.abs(rate) > APPROACH_RATE)):
        kind = 'approach' if rate[i] > 0 else 'recede'
        if ev and ev[-1][1] == kind and fr[i] - ev[-1][0] <= 2 * (fr[1] - fr[0]):
            ev[-1] = (ev[-1][0], kind, max(ev[-1][2], abs(rate[i])))   # one event per run
            continue
        ev.append((int(fr[i]), kind, float(abs(rate[i]))))
    for i in np.flatnonzero(area >= CLOSE_AREA):
        if not (ev and ev[-1][1] == 'close' and fr[i] - ev[-1][0] <= 2 * (fr[1] - fr[0])):
            ev.append((int(fr[i]), 'close', float(area[i])))
    return sorted(ev)


def report(srcs, top):
    best = []
    for src in srcs:
        out = os.path.splitext(src)[0] + '_objects.npz'
        if not os.path.exists(out):
            continue
        z = np.load(out)
        name = os.path.basename(src).replace('_stabilized.mp4', '')
        fps = float(z['fps'])
        ev = people_motion(z)
        kinds = {k: sum(1 for e in ev if e[1] == k) for k in ('approach', 'recede', 'close')}
        print(f'{name}: people in {np.mean(z["person_n"] > 0) * 100:3.0f} % of samples (biggest {z["person_area"].max() * 100:4.1f} %), '
              f'events {kinds}, vehicles {np.mean(z["vehicle_n"] > 0) * 100:3.0f} %, animals {np.mean(z["animal_n"] > 0) * 100:3.0f} %')
        best += [(e[2] if e[1] != 'close' else e[2] * 20, name, e[0] / fps, e[1]) for e in ev]
    best.sort(reverse=True)
    print(f'top {top} people moments:')
    shown = []
    for s, name, t, kind in best:
        if any(n == name and abs(t - u) < 4 for n, u in shown):
            continue
        shown.append((name, t))
        print(f'  {name} {t:6.1f}s {kind}')
        if len(shown) >= top:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('renders', nargs='+')
    ap.add_argument('--weights', default=None)
    ap.add_argument('--every', type=int, default=25, help='run on every Nth frame (25 = 2 per second at 50 fps)')
    ap.add_argument('--conf', type=float, default=0.35)
    ap.add_argument('--top', type=int, default=10)
    ap.add_argument('--summary', action='store_true', help='only print the report')
    a = ap.parse_args()
    if not a.summary:
        if not a.weights:
            raise SystemExit('--weights is needed to detect')
        import gpu
        gpu.device()
        from ultralytics import YOLO
        model = YOLO(a.weights)
        for src in a.renders:
            out = os.path.splitext(src)[0] + '_objects.npz'
            if os.path.exists(out):
                continue
            t0 = time.time()
            fps = float(av.open(src).streams.video[0].average_rate)
            np.savez(out, **detect(model, src, max(1, int(round(a.every * fps / 50.0))), a.conf))
            print(f'{os.path.basename(src)}: detected in {time.time() - t0:.0f}s', flush=True)
    report(a.renders, a.top)


if __name__ == '__main__':
    main()
