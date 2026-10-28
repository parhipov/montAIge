"""Beats and downbeats as a listener hears them: Beat This! (CPJKU, ISMIR 2024, MIT) on a track, kept beside it as
<track>.beats.npz. edit.py fits its constant-tempo grid to the onsets as before and takes the beat's phase and the
downbeat from here where the two disagree: an oom-pah's strummed chord is the strongest onset, so the grid sat on
it, half a beat off the bass (Brightly Fancy: every tracker beat 0.48 of a beat off the grid, the chords changing
on its "4 and"); where they agree (a jazz track, a fiddle reel) nothing moves.

    SPEECH_PY montaige/beats.py TRACK [...]

Runs in models/venvs/speech (torch, torchaudio; beat-this, rotary-embedding-torch and soxr installed there with
--no-deps); the checkpoint goes to models/beats/final0.ckpt from the authors' public share. montage.py and
music_check.py call it for the job's tracks; a shortened track takes its source's beats through the pieces
(save_cut): on a cut of 6 phrases the tracker heard bars of 2 and 3 after the joins, on the whole track bars of 4.
"""
import base64
import os
import subprocess
import sys
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(os.path.dirname(HERE), 'models')
CKPT = os.path.join(MODELS, 'beats', 'final0.ckpt')
SHARE = ('https://cloud.cp.jku.at/public.php/webdav/final0.ckpt', '7ik4RrBKTS273gp')   # the package's own URL gave 500
SHIFT_MIN = 0.25        # beats: the tracker's beats this far off the grid (circular mean) move it
SURE = 0.6              # the tracker's beats that close together (mean resultant length), its downbeats on one place


def npz_of(path):
    return path + '.beats.npz'


def load(path):
    """(beats, downbeats) in seconds, or None when the track has none."""
    p = npz_of(path)
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return z['beats'], z['downbeats']


def venv_python():
    own = [os.path.join(MODELS, 'venvs', 'speech', *x) for x in (('Scripts', 'python.exe'), ('bin', 'python'))]
    return os.environ.get('SPEECH_PY') or next((x for x in own if os.path.exists(x)), None)


def ensure(paths):
    """The tracker run on the tracks that have no beats file yet (in its venv); False when it cannot run."""
    todo = [p for p in dict.fromkeys(paths) if not os.path.exists(npz_of(p))]
    if not todo:
        return True
    py = venv_python()
    if not py:
        print('  beats: no models/venvs/speech (SPEECH_PY): the grid from the onsets only')
        return False
    return subprocess.run([py, os.path.abspath(__file__)] + todo).returncode == 0


def save_cut(src, pieces, out):
    """A shortened track's beats: its source's, through the (t0, t1) pieces it was spliced from."""
    if os.path.exists(npz_of(out)):
        os.remove(npz_of(out))
    heard = load(src)
    if heard is None:
        return
    cut = []
    for ts in heard:
        c0, xs = 0.0, []
        for a, b in pieces:
            xs += [c0 + t - a for t in ts if a <= t < b]
            c0 += b - a
        cut.append(np.array(xs))
    np.savez(npz_of(out), beats=cut[0], downbeats=cut[1])


def phase_shift(b0, P, tb):
    """Seconds to move a grid (first beat b0, period P) onto the tracker's beats tb, 0 when they agree or the
    tracker is unsure; the circular mean, as half a beat off is +0.5 and -0.5 alike."""
    z = np.exp(2j * np.pi * (np.asarray(tb) - b0) / P).mean()
    ph = np.angle(z) / (2 * np.pi)
    return ph * P if abs(z) >= SURE and abs(ph) >= SHIFT_MIN else 0.0


def downbeat(beats, P, td):
    """The grid's downbeat phase (0-3) where most of the tracker's downbeats fall, None when under SURE of them do."""
    td = np.asarray(td)
    td = td[(td >= beats[0] - P / 4) & (td <= beats[-1] + P / 4)]
    if td.size < 4:
        return None
    k = np.rint((td - beats[0]) / P).astype(int) % 4
    n = np.bincount(k, minlength=4)
    return int(np.argmax(n)) if n.max() >= SURE * td.size else None


def main(paths):
    import torch
    from beat_this.inference import File2Beats
    if not os.path.exists(CKPT):
        os.makedirs(os.path.dirname(CKPT), exist_ok=True)
        req = urllib.request.Request(SHARE[0], headers={
            'Authorization': 'Basic ' + base64.b64encode(f'{SHARE[1]}:'.encode()).decode()})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        if data[:2] != b'PK':
            raise SystemExit(f'beats: {SHARE[0]} gave no checkpoint ({data[:60]!r})')
        with open(CKPT, 'wb') as fh:
            fh.write(data)
    f2b = File2Beats(checkpoint_path=CKPT, device='cuda' if torch.cuda.is_available() else 'cpu', dbn=False)
    for p in paths:
        b, d = f2b(p)
        np.savez(npz_of(p), beats=b, downbeats=d)
        print(f'  beats: {os.path.basename(p)}: {len(b)} beats, {len(d)} downbeats')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
