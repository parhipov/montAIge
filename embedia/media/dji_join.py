"""Join DJI O4 Pro split recordings (the camera cuts a recording into ~4-minute files).

    python -m embedia.media.dji_join FOLDER [--dry-run]      FOLDER and all its subfolders

Parts of one recording have consecutive numbers (..._0020_D, ..._0021_D, ...) and each starts where the
previous one ended (the time in the file name plus the duration). They are joined by mp4_merge from
Gyroflow (github.com/gyroflow/mp4-merge): no re-encoding, every track kept (video, DJI telemetry djmd /
dbgi), the same code Gyroflow uses to join split recordings itself. The result goes next to them,
<first file name>_joined.mp4; after a successful join the parts are renamed to _<name>, so they are easy
to check and delete. Ask the owner of the footage before running it without --dry-run.
"""
MAX_GAP_SECONDS = 3.  # gap allowed between the end of a part and the start of the next
MP4_MERGE_URL = 'https://github.com/gyroflow/mp4-merge/releases/download/v0.1.11/mp4_merge-windows64.exe'

import argparse  # noqa: E402
from datetime import datetime, timedelta  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import urllib.request  # noqa: E402

import av  # noqa: E402

SEGMENT = re.compile(r'^DJI_(\d{14})_(\d{4})_D\.MP4$', re.IGNORECASE)
HERE = Path(__file__).resolve().parent


def mp4_merge_executable():
    executable = HERE / 'mp4_merge.exe'
    if not executable.exists():
        print(f'downloading mp4_merge: {MP4_MERGE_URL}', flush=True)
        partial = executable.with_suffix('.part')
        urllib.request.urlretrieve(MP4_MERGE_URL, partial)
        os.replace(partial, executable)
    return executable


def duration_seconds(path):
    with av.open(str(path)) as container:
        if container.duration is None:
            raise ValueError(f'cannot read the duration: {path}')
        return container.duration / av.time_base


def scan(folder):
    """Parts in the folder (not its subfolders): [(number, start, end, path)] by number."""
    segments = []
    for name in os.listdir(folder):
        match = SEGMENT.match(name)
        if not match:
            continue
        path = Path(folder, name)
        start = datetime.strptime(match.group(1), '%Y%m%d%H%M%S')
        end = start + timedelta(seconds=duration_seconds(path))
        segments.append((int(match.group(2)), start, end, path))
    segments.sort()
    return segments


def groups(segments, max_gap=MAX_GAP_SECONDS):
    """Chains: the next number, starting within max_gap of the end of the previous part."""
    chains = []
    for number, start, end, path in segments:
        if chains:
            last_number, _, last_end, _ = chains[-1][-1]
            gap = (start - last_end).total_seconds()
            if number == last_number + 1 and abs(gap) <= max_gap:
                chains[-1].append((number, start, end, path))
                continue
        chains.append([(number, start, end, path)])
    return [[item[3] for item in chain] for chain in chains if len(chain) > 1]


def chains_under(root, max_gap=MAX_GAP_SECONDS):
    """[(output path, [parts])] for every folder under root; outputs that exist already are left out."""
    out = []
    for folder, _, _ in os.walk(root):
        for chain in groups(scan(folder), max_gap):
            output = chain[0].with_name(chain[0].stem + '_joined.mp4')
            if not output.exists():
                out.append((output, chain))
    return out


def join(paths, output, executable):
    partial = output.with_name(output.name + '.tmp')
    command = [str(executable), *(str(path) for path in paths), '--out', str(partial)]
    try:
        subprocess.run(command, check=True)
        expected = sum(path.stat().st_size for path in paths)
        if not partial.exists() or partial.stat().st_size < expected * 0.999:
            raise ValueError(f'the result is suspiciously small: {partial}')
        os.replace(partial, output)
    finally:
        if partial.exists():
            partial.unlink()


def hide_sources(paths):
    for path in paths:
        hidden = path.with_name('_' + path.name)
        if hidden.exists():
            raise ValueError(f'already exists: {hidden}')
        path.rename(hidden)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('folder')
    parser.add_argument('--dry-run', action='store_true', help='only show what would be joined')
    parser.add_argument('--max-gap', type=float, default=MAX_GAP_SECONDS, help='gap between parts, s')
    parser.add_argument('--keep-names', action='store_true', help='do not rename the parts to _name')
    args = parser.parse_args()
    root = Path(args.folder)
    if not root.is_dir():
        raise ValueError(f'no such folder: {root}')
    todo = chains_under(root, args.max_gap)
    for output, chain in todo:
        print(f'{output}\n    <- ' + ' + '.join(path.name for path in chain), flush=True)
    if args.dry_run:
        print(f'to join: {len(todo)}')
        return
    executable = mp4_merge_executable()
    for output, chain in todo:
        join(chain, output, executable)
        if not args.keep_names:
            hide_sources(chain)
    print(f'joined {len(todo)}')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        sys.exit(f'error: {error}')
