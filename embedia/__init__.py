"""embedia: footage in, an index out. Every clip of a shoot measured once (camera motion, people and their poses,
objects, CLIP and CLAP embeddings, captions, speech), kept in one folder with a SQLite database, and asked in
words or by example afterwards (index.py: the layout and the Python interface; run.py: the steps; media/: the
files, their metadata, proxies and Gyroflow; analyzers/: the steps' modules; README.md).

    python -m embedia init out/trip/index --footage /path/to/footage
    python -m embedia run out/trip/index
    python -m embedia search out/trip/index "a jump into the water"
"""
import importlib

from .index import VERSION as __version__  # noqa: F401
from .index import Index, cache_of, create, open  # noqa: F401, A004

MEDIA = ('dji_join', 'djimeta', 'gf_motion', 'gf_render', 'gpu', 'mp4parse', 'pb', 'pix', 'proxy')
ANALYZERS = ('analyze', 'beauty', 'camera', 'captions', 'exports', 'faces', 'gps', 'objects', 'people', 'places',
             'smiles', 'sounds', 'speech', 'vis_motion')


def __getattr__(name):
    """A module of media/ or analyzers/ by its name alone (`from embedia import pix`): readers of the index
    import them so, wherever they live in the package."""
    if name in MEDIA or name in ANALYZERS:
        return importlib.import_module(f'.{"media" if name in MEDIA else "analyzers"}.{name}', __name__)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
