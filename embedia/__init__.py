"""embedia: footage in, an index out. Every clip of a shoot measured once (camera motion, people and their poses,
objects, CLIP and CLAP embeddings, captions, speech), kept in one folder with a SQLite database, and asked in
words or by example afterwards (index.py: the layout and the Python interface; run.py: the steps; README.md).

    python -m embedia init out/trip/index --footage /path/to/footage
    python -m embedia run out/trip/index
    python -m embedia search out/trip/index "a jump into the water"
"""
from embedia.index import VERSION as __version__  # noqa: F401
from embedia.index import Index, cache_of, create, open  # noqa: F401, A004
