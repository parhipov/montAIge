"""DJI clip metadata for the montage: the camera's orientation at every video frame, from the `djmd` track.

    python djimeta.py CLIP.MP4          prints frames, quaternion samples, the first frame's orientation

The `djmd` track holds one protobuf message per video frame; its IMU part carries the fused attitude
(several quaternions per frame interval). They are turned into Gyroflow's camera frame the way
telemetry-parser does (q' = flipY * q * dji_to_cam) and kept continuous across the hemisphere flip, as
Gyroflow does. A frame's orientation is the middle sample of its interval; frames without samples are
interpolated. Self-contained: mp4parse.py and pb.py are generic readers.
"""
import math
import struct
import sys

import numpy as np

import pb
from mp4parse import open_mp4

F_FRAME = 3                                   # ProductMeta.frame_meta
FRAME_IMU = 3                                 # FrameMeta.imu_frame_meta
IMU_ATTITUDE_AFTER_FUSION = 2                 # FrameMetaOfIMU.IMU_attitude_after_fusion
ATT_QUAT = 3                                  # DeviceAttitude.attitude (repeated quaternion)
DJI_TO_CAM = (0.5, -0.5, -0.5, 0.5)           # right-multiplied
CAM_FLIP_Y = (0.0, 0.0, 1.0, 0.0)             # left-multiplied (180 deg about Y)


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)


def sub(data, num):
    """First length-delimited field `num` of a protobuf message, or None."""
    for fn, wt, v, _ in pb.iter_fields(data):
        if fn == num and wt == 2:
            return v
    return None


def frame_quats(path):
    """-> {video frame: [camera quaternions (w, x, y, z) of that frame interval]}."""
    f, _, _, tracks = open_mp4(path)
    try:
        tr = next(t for t in tracks if t.formats and t.formats[0] == b'djmd')
    except StopIteration:
        f.close()
        raise SystemExit(f'{path}: no DJI `djmd` metadata track')
    out, prev, inv = {}, None, False
    for si, (_, _, data) in enumerate(tr.samples(f)):
        fm = sub(data, F_FRAME)
        imu = sub(fm, FRAME_IMU) if fm is not None else None
        att = sub(imu, IMU_ATTITUDE_AFTER_FUSION) if imu is not None else None
        if att is None:
            continue
        for fn, wt, raw, _ in pb.iter_fields(att):
            if fn != ATT_QUAT or wt != 2:
                continue
            vals = {qf: struct.unpack('<f', qv)[0] for qf, qwt, qv, _ in pb.iter_fields(raw) if qwt == 5}
            q = (vals.get(1, 0.0), vals.get(2, 0.0), vals.get(3, 0.0), vals.get(4, 0.0))
            if any(math.isnan(c) for c in q):
                continue
            qc = qmul(CAM_FLIP_Y, qmul(q, DJI_TO_CAM))
            if qc == (0.0, 0.0, 0.0, 0.0):
                continue
            if prev is not None and math.sqrt(sum((a - b) ** 2 for a, b in zip(prev, qc))) > 1.5:
                inv = not inv                                    # Gyroflow keeps the series continuous
            prev = qc
            out.setdefault(si, []).append(tuple(-c for c in qc) if inv else qc)
    f.close()
    if not out:
        raise SystemExit(f'{path}: the DJI metadata has no fused-attitude quaternions')
    return out


def per_frame(path, n_frames):
    """(n_frames, 4) unit quaternions: each frame's middle sample, gaps interpolated."""
    by = frame_quats(path)
    q = np.full((n_frames, 4), np.nan)
    for fr, lst in by.items():
        if fr < n_frames:
            q[fr] = lst[len(lst) // 2]
    ok = ~np.isnan(q[:, 0])
    idx = np.arange(n_frames)
    for c in range(4):
        q[:, c] = np.interp(idx, idx[ok], q[ok, c])
    return q / np.linalg.norm(q, axis=1, keepdims=True)


if __name__ == '__main__':
    for p in sys.argv[1:]:
        by = frame_quats(p)
        print(f'{p}: {len(by)} frames with attitude, {sum(len(v) for v in by.values())} quaternions, '
              f'first {np.round(by[min(by)][0], 4).tolist()}')
