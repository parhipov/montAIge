"""montaige: edits to music from an embedia index (montage.py runs a job file: the index, the edits, QC, finals;
edit.py plans and renders an edit; README.md of the repository).

    python -m montaige out/<job>/job.toml [--only edits] [--final NAME --size 3840x2880]

The montage reads its index with the embedia package EMBEDIA names (a copy of embedia under another name,
embedia_ics; the job file's `embedia` sets it for the run and every step it starts): `embedia` in its imports is
that package, each module one object under both names.
"""
import importlib
import importlib.abc
import importlib.util
import os
import sys

EMBEDIA = os.environ.get('EMBEDIA') or 'embedia'


class _Embedia(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """`embedia` and `embedia.X` as EMBEDIA's package and its modules."""
    def find_spec(self, name, path=None, target=None):
        if name == 'embedia' or name.startswith('embedia.'):
            return importlib.util.spec_from_loader(name, self)
        return None

    def create_module(self, spec):
        return importlib.import_module(EMBEDIA + spec.name[len('embedia'):])

    def exec_module(self, module):
        pass


if EMBEDIA != 'embedia' and not any(isinstance(f, _Embedia) for f in sys.meta_path):
    sys.meta_path.insert(0, _Embedia())
