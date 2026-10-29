"""Frames of a video as arrays, through one conversion context instead of a new one per frame.

PyAV's `frame.to_ndarray(format=..., width=..., height=...)` and `frame.reformat(...)` build a fresh swscale
context for every frame: 3-7 ms each at 640 px, five to twenty times the decoding of the frame itself (a 640 px
proxy decodes at ~2500 fps, converts at ~200). A `VideoReformatter` keeps the context between frames and gives
the same pixels (checked bit for bit). Every script that turns frames into arrays goes through here:

    import pix
    img = pix.ndarray(fr, format='bgr24')                      # like fr.to_ndarray(format='bgr24')
    g = pix.ndarray(fr, format='gray', width=160, height=90)
    vf = pix.reformat(vf, format='yuv420p', ...)               # like vf.reformat(...), for encoding

One context per kind of conversion (source size and format, the arguments) and per thread (a context is not
thread-safe: shots graded in parallel crashed on a shared one), so a loop that decodes and encodes in turn does
not rebuild the context every time.
"""
import threading

from av.video.reformatter import VideoReformatter

_LOCAL = threading.local()


def reformat(frame, **kw):
    """`frame.reformat(**kw)` with a kept context for this kind of conversion."""
    rfs = getattr(_LOCAL, 'rfs', None)
    if rfs is None:
        rfs = _LOCAL.rfs = {}
    key = (frame.format.name, frame.width, frame.height, tuple(sorted(kw.items())))
    rf = rfs.get(key)
    if rf is None:
        rf = rfs[key] = VideoReformatter()
    return rf.reformat(frame, **kw)


def ndarray(frame, **kw):
    """`frame.to_ndarray(**kw)` with a kept context for this kind of conversion."""
    return reformat(frame, **kw).to_ndarray()
