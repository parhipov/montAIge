"""Contact sheet of an EDL from the small renders: 3 frames per shot (start, middle, end), DJI crop shown.

    python edl_sheet.py EDIT_edl.json OUT.jpg [--crop-bottom 0.16 --crop-top 0.01]
"""
import argparse
import json
import os

import av
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('edl')
ap.add_argument('out')
ap.add_argument('--crop-bottom', type=float, default=0.16)
ap.add_argument('--crop-top', type=float, default=0.01)
a = ap.parse_args()
e = json.load(open(a.edl, encoding='utf-8'))
fl = {f['idx']: f for f in e['flights']}
rows = []
for s in e['scenes']:
    for x in s['shots']:
        f = fl[x['flight']]
        c = av.open(f['render'])
        st = c.streams.video[0]
        tiles = []
        for fr_i in (int(x['anchors'][0]) + 3, int((x['anchors'][0] + x['anchors'][-1]) / 2), int(x['anchors'][-1]) - 3):
            c.seek(int(max(fr_i - 2, 0) / f['fps'] / st.time_base), stream=st, backward=True)
            for fr in c.decode(st):
                if int(round(float(fr.pts * st.time_base) * f['fps'])) >= fr_i:
                    im = fr.to_ndarray(format='bgr24')
                    break
            if os.path.basename(f['original']).upper().startswith('DJI_'):
                h, w = im.shape[:2]
                y0, y1 = int(a.crop_top * h), int((1 - a.crop_bottom) * h)
                cw = int(round((y1 - y0) * w / h))
                x0 = (w - cw) // 2
                im = im[y0:y1, x0:x0 + cw]
            tiles.append(cv2.resize(im, (320, 180), interpolation=cv2.INTER_AREA))
        c.close()
        row = np.hstack(tiles)
        txt = f"{x['t0']:5.1f}-{x['t1']:5.1f}s  {s.get('day')}  {os.path.basename(f['original'])} @{x['src_t0']:.0f}s  {x['move']}  sk {x['skill']:.2f}"
        cv2.rectangle(row, (0, 0), (row.shape[1], 16), (0, 0, 0), -1)
        cv2.putText(row, txt, (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        rows.append(row)
half = (len(rows) + 1) // 2
left, right = rows[:half], rows[half:]
while len(right) < len(left):
    right.append(np.zeros_like(left[0]))
cv2.imwrite(a.out, np.vstack([np.hstack([l, r]) for l, r in zip(left, right)]), [cv2.IMWRITE_JPEG_QUALITY, 82])
print('ok', len(rows), 'shots')
