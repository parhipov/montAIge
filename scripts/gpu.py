"""The ML steps run on a CUDA GPU only: without one they stop with a clear message instead of
crawling on the CPU for hours."""
import sys


def device():
    """'cuda', or SystemExit with what to do."""
    try:
        import torch
    except ImportError:
        raise SystemExit(f'{sys.executable}: no torch here; run the GPU steps with a Python that has CUDA torch '
                         '(GPU_PY in README Quick start)')
    if not torch.cuda.is_available():
        raise SystemExit(f'{sys.executable}: torch {torch.__version__} sees no CUDA GPU; the ML steps (YOLO, CLIP) '
                         'need one: run them with GPU_PY')
    return 'cuda'
