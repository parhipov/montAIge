"""Where the toolset's inputs are: the folder EMBEDIA_MODELS names (models shared with another project), else the
first `models/` folder up from the package, so a job's snapshot of the code finds them as well as the repository
does."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def _up(name):
    d = HERE
    while True:
        if os.path.isdir(os.path.join(d, name)):
            return os.path.join(d, name)
        if os.path.dirname(d) == d:
            raise SystemExit(f'no {name}/ folder above {HERE}: set EMBEDIA_MODELS')
        d = os.path.dirname(d)


def expand(args):
    """Paths from a command line: wildcards expanded, a plain path kept even when its file is gone (a proxy that
    is not kept: its features are beside where it was)."""
    import glob
    out = {}
    for a in args:
        for p in (glob.glob(a) if glob.has_magic(a) else [a]):
            out[os.path.normcase(p)] = p
    return sorted(out.values())


MODELS = os.path.abspath(os.environ['EMBEDIA_MODELS']) if os.environ.get('EMBEDIA_MODELS') else _up('models')
ROOT = os.path.dirname(MODELS)          # the folder holding models/: where the steps run
