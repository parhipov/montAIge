"""python -m montaige out/<job>/job.toml [...]: montage.py's job runner (its --help has every option)."""
import os
import sys

if __package__ in (None, ''):                                    # run as a file
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from montaige import montage  # noqa: E402

if __name__ == '__main__':
    montage.main()
