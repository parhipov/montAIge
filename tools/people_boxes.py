"""People in every frame of a preview (YOLO, persons only) -> NPZ: frame, x0, y0, x1, y1, conf (pixels of the preview).

    python tools/people_boxes.py PREVIEW.mp4 OUT.npz
"""
import os
import sys

import av
import numpy as np
from ultralytics import YOLO

src, out = sys.argv[1:3]
model = YOLO(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'yolo', 'yolov8s.pt'))
rows = []
with av.open(src) as c:
    for i, fr in enumerate(c.decode(video=0)):
        r = model.predict(fr.to_ndarray(format='rgb24')[..., ::-1], classes=[0], conf=0.2, imgsz=640, verbose=False)[0]
        for (x0, y0, x1, y1), p in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            rows.append((i, x0, y0, x1, y1, p))
rows = np.array(rows, np.float32).reshape(-1, 6)
np.savez(out, boxes=rows)
print(f'{len(rows)} boxes, frames with people {len(np.unique(rows[:, 0]))} of {i + 1}')
