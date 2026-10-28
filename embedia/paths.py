"""Where the toolset's inputs are: the first `models/` folder up from the package, so a job's snapshot of the code
(out/<job>/embedia, out/<job>/montaige) finds them as well as the repository does."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def _up(name):
    d = HERE
    while True:
        if os.path.isdir(os.path.join(d, name)):
            return os.path.join(d, name)
        if os.path.dirname(d) == d:
            raise SystemExit(f'no {name}/ folder above {HERE}')
        d = os.path.dirname(d)


MODELS = _up('models')
ROOT = os.path.dirname(MODELS)          # ai_montage/
