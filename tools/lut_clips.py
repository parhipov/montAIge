"""Apply a .cube LUT to whole clips on the GPU: 10-bit HEVC in, 10-bit HEVC (NVENC) out, the same size and
frame rate. For a delivery of whole clips (stabilize and grade, no edit): first `python -m embedia.gf_render --codec
hevc10 --width W --height H --bitrate 120M` at the clip's own size (given, not left to Gyroflow), then this.

    GPU_PY tools/lut_clips.py PAIRS.json --lut FILE.cube [--bitrate 120M] [--limit N] [--cpu] [--wait]

PAIRS.json: [[src, dst], ...] (the paths come through a file: the GPU venv does not take Cyrillic arguments).
The sound of src (gf_render.py keeps the clip's) is copied as it is.
--wait: sources still missing (Gyroflow at work) are waited for, whichever appears first goes first; a second
instance on the same list takes the clips the first is not writing (a .tmp that grew in the last 2 min).
The decoder is NVDEC (hevc_cuvid) when PyAV has it (--cpu: the CPU one); YUV <-> RGB (BT.709, limited range;
chroma bilinear up, 2x2 mean down) and the LUT (trilinear) run in torch, the LUT exactly as final.py's
GpuGrade does for an edit (0.1 of 255 from it on a frame, before the encoder; compare 10-bit output in 16 bits,
swscale's rgb24 of it is 1.4 darker). Decoding, the GPU work and the encoder run in three threads; frames cross
the bus through pinned buffers. 4K 50 fps 10-bit on an RTX 3050: ~12 fps alone, ~7 beside two Gyroflow renders.
"""
import argparse
import json
import os
import queue
import sys
import threading
import time

import av
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the repository: embedia, montaige (run as a file)
from montaige import edit as bf  # noqa: E402


class Grader:
    def __init__(self, lut_path, dev):
        lut = bf.CubeLUT(lut_path)
        self.vol = torch.tensor(lut.raw).permute(3, 0, 1, 2)[None].contiguous().to(dev)   # (1, 3, b, g, r)
        self.dmin = torch.tensor(lut.dmin, dtype=torch.float32, device=dev)
        self.dmax = torch.tensor(lut.dmax, dtype=torch.float32, device=dev)
        self.dev = dev
        self.pin = {}                                            # pinned host buffers per plane: transfers at
                                                                 # bus speed, not the ~1 GB/s of pageable memory

    def pinned(self, name, shape):
        buf = self.pin.get(name)
        if buf is None or tuple(buf.shape) != tuple(shape):
            buf = self.pin[name] = torch.empty(shape, dtype=torch.int16).pin_memory()
        return buf

    def __call__(self, y16, u16, v16, shift):
        """y16 (H, W), u16 / v16 (H/2, W/2): int16 views of the 10-bit planes (p010le: the value in the top bits,
        shift 6). Returns the graded picture as p010le planes: Y (H, W) and UV (H/2, W/2, 2), int16 (views of
        pinned buffers, valid until the next call)."""
        dev = self.dev

        def up(name, a):
            buf = self.pinned(name, a.shape)
            buf.numpy()[...] = a                                 # every earlier copy from it is done: the last
            return (buf.to(dev, non_blocking=True).int() & 0xFFFF) >> shift   # call ended with a blocking download

        y, u, v = up('y', y16), up('u', u16), up('v', v16)
        H, W = y.shape
        yf = (y.float() - 64.0) / 876.0
        cb = (u.float() - 512.0) / 896.0
        cr = (v.float() - 512.0) / 896.0
        cb = F.interpolate(cb[None, None], size=(H, W), mode='bilinear', align_corners=False)[0, 0]
        cr = F.interpolate(cr[None, None], size=(H, W), mode='bilinear', align_corners=False)[0, 0]
        rgb = torch.stack((yf + 1.5748 * cr, yf - 0.18732 * cb - 0.46812 * cr, yf + 1.8556 * cb), -1).clamp_(0, 1)
        grid = ((rgb - self.dmin) / (self.dmax - self.dmin) * 2 - 1)[None, None]       # (1, 1, H, W, rgb)
        out = F.grid_sample(self.vol, grid, mode='bilinear', padding_mode='border', align_corners=True)[0, :, 0]
        r, g, b = out[0].clamp(0, 1), out[1].clamp(0, 1), out[2].clamp(0, 1)         # (H, W) each
        yy = 0.2126 * r + 0.7152 * g + 0.0722 * b
        cb = F.avg_pool2d(((b - yy) / 1.8556)[None, None], 2)[0, 0]
        cr = F.avg_pool2d(((r - yy) / 1.5748)[None, None], 2)[0, 0]
        y10 = (yy * 876.0 + 64.0).round_().clamp_(64, 940)
        c10 = (torch.stack((cb, cr), -1) * 896.0 + 512.0).round_().clamp_(64, 960)
        y_out = self.pinned('y_out', y10.shape)
        y_out.copy_((y10.int() << 6).to(torch.int16))                                # p010le: 10 bits on top
        uv_out = self.pinned('uv_out', c10.shape)
        uv_out.copy_((c10.int() << 6).to(torch.int16))
        return y_out.numpy(), uv_out.numpy()


def planes(fr):
    """The 10-bit planes of a decoded frame as int16 arrays (y, u, v, shift), without copying."""
    W, H = fr.width, fr.height

    def arr(p, rows, cols):
        stride = p.line_size // 2
        return np.frombuffer(p, dtype=np.int16, count=rows * stride).reshape(rows, stride)[:, :cols]

    y = arr(fr.planes[0], H, W)
    if fr.format.name in ('p010le', 'p010'):
        uv = arr(fr.planes[1], H // 2, W).reshape(H // 2, W // 2, 2)
        return y, uv[..., 0], uv[..., 1], 6
    if fr.format.name == 'yuv420p10le':
        return y, arr(fr.planes[1], H // 2, W // 2), arr(fr.planes[2], H // 2, W // 2), 0
    raise ValueError(f'unexpected pixel format {fr.format.name}')


def put_plane(plane, a):
    """a: int16 (rows, cols) into the plane's buffer (rows may be padded)."""
    rows, cols = a.shape
    stride = plane.line_size // 2
    try:
        view = np.frombuffer(plane, dtype=np.int16, count=rows * stride).reshape(rows, stride)
        view[:, :cols] = a                                       # the plane's buffer is writable
        return
    except (ValueError, TypeError, BufferError):
        pass
    buf = np.zeros(plane.buffer_size, np.uint8)
    buf[:rows * stride * 2].view(np.int16).reshape(rows, stride)[:, :cols] = a
    plane.update(buf)


def frames(c, st, cpu):
    """Decoded frames of the stream: through hevc_cuvid (NVDEC) unless --cpu or it is missing."""
    ctx = None
    if not cpu:
        try:
            ctx = av.codec.CodecContext.create('hevc_cuvid', 'r')
            ctx.extradata = st.codec_context.extradata
        except Exception as error:                              # noqa: BLE001
            print(f'  no hevc_cuvid ({error}): CPU decoder', flush=True)
            ctx = None
    if ctx is None:
        st.thread_type = 'AUTO'
        yield from c.decode(st)
        return
    for pkt in c.demux(st):
        yield from ctx.decode(pkt)


def with_sound(video, src, dst):
    """The graded video and src's sound, both copied -> dst (the video alone when src has no sound)."""
    vi, ai = av.open(video), av.open(src)
    if not ai.streams.audio:
        vi.close()
        ai.close()
        os.replace(video, dst)
        return
    tmp = os.path.splitext(dst)[0] + '.mux.mp4'
    oc = av.open(tmp, 'w')
    out = {id(vi.streams.video[0]): oc.add_stream_from_template(vi.streams.video[0])}
    out.update({id(s): oc.add_stream_from_template(s) for s in ai.streams.audio})
    for pkt in [*vi.demux(vi.streams.video[0]), *ai.demux(*ai.streams.audio)]:
        if pkt.dts is not None:
            st = out[id(pkt.stream)]
            pkt.stream = st
            oc.mux(pkt)
    oc.close()
    vi.close()
    ai.close()
    os.remove(video)
    os.replace(tmp, dst)


def run(src, dst, grader, bitrate, limit, cpu):
    c = av.open(src)
    st = c.streams.video[0]
    W, H, rate = st.width, st.height, st.average_rate
    total = st.frames or int(float(st.duration * st.time_base) * rate)
    mbps = float(bitrate.rstrip('Mm'))
    tmp = os.path.splitext(dst)[0] + '.tmp.mp4'
    oc = av.open(tmp, 'w')
    vs = oc.add_stream('hevc_nvenc', rate=rate)
    vs.width, vs.height, vs.pix_fmt = W, H, 'p010le'
    vs.bit_rate = int(mbps * 1e6)
    vs.options = {'preset': 'p5', 'rc': 'vbr', 'maxrate': f'{mbps * 1.4:.0f}M', 'bufsize': f'{mbps * 3:.0f}M',
                  'profile': 'main10'}
    for key, val in (('colorspace', 1), ('color_primaries', 1), ('color_trc', 1), ('color_range', 1)):
        try:
            setattr(vs.codec_context, key, val)
        except (AttributeError, TypeError, ValueError):
            pass
    q_in, q_out, failed = queue.Queue(maxsize=6), queue.Queue(maxsize=4), []

    def producer():
        try:
            for i, fr in enumerate(frames(c, st, cpu)):
                if limit and i >= limit:
                    break
                q_in.put(fr)
        except Exception as error:                                  # noqa: BLE001
            failed.append(error)
        q_in.put(None)

    def encoder():
        try:
            while True:
                vf = q_out.get()
                for pkt in vs.encode(vf):                           # None flushes
                    oc.mux(pkt)
                if vf is None:
                    return
        except Exception as error:                                  # noqa: BLE001
            failed.append(error)
            while q_out.get() is not None:                          # drain, so the main loop ends
                pass

    threads = [threading.Thread(target=producer, daemon=True), threading.Thread(target=encoder, daemon=True)]
    for t in threads:
        t.start()
    n, t0, fmt = 0, time.time(), None
    while True:
        fr = q_in.get()
        if fr is None or failed:
            break
        fmt = fmt or fr.format.name
        y16, u16, v16, shift = planes(fr)
        y_out, uv_out = grader(y16, u16, v16, shift)
        vf = av.VideoFrame(W, H, 'p010le')
        put_plane(vf.planes[0], y_out)
        put_plane(vf.planes[1], uv_out.reshape(H // 2, W))
        vf.pts = n
        q_out.put(vf)
        n += 1
        if n % 2000 == 0:
            print(f'  {n}/{total} {n / (time.time() - t0):.1f} fps', flush=True)
    q_out.put(None)
    for t in threads:
        t.join()
    oc.close()
    c.close()
    if failed:
        os.remove(tmp)
        raise failed[0]
    with_sound(tmp, src, dst)
    print(f'{os.path.basename(dst)}: {n} frames ({fmt} in) in {time.time() - t0:.0f}s, '
          f'{n / (time.time() - t0):.1f} fps', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('pairs', help='JSON [[src, dst], ...]')
    ap.add_argument('--lut', required=True)
    ap.add_argument('--bitrate', default='120M')
    ap.add_argument('--limit', type=int, default=0, help='frames per clip (a test)')
    ap.add_argument('--cpu', action='store_true', help='decode on the CPU')
    ap.add_argument('--wait', action='store_true', help='wait for missing sources (up to 6 hours)')
    a = ap.parse_args()
    with open(a.pairs, encoding='utf-8') as fh:
        pending = [tuple(p) for p in json.load(fh)]
    grader = Grader(a.lut, torch.device('cuda'))
    deadline = time.time() + 6 * 3600

    def busy(dst):
        """Another instance is writing this one (its .tmp grew in the last 2 min)."""
        tmp = os.path.splitext(dst)[0] + '.tmp.mp4'
        return os.path.exists(tmp) and time.time() - os.path.getmtime(tmp) < 120

    while pending:
        pending = [p for p in pending if not os.path.exists(p[1])]
        ready = [p for p in pending if os.path.exists(p[0]) and not busy(p[1])]
        if not ready:
            if not pending or not a.wait or time.time() > deadline:
                for src, _ in pending:
                    print(f'{os.path.basename(src)}: missing or busy, skipped', flush=True)
                break
            time.sleep(30)
            continue
        src, dst = ready[0]
        pending.remove(ready[0])
        run(src, dst, grader, a.bitrate, a.limit, a.cpu)


if __name__ == '__main__':
    main()
