"""An index: everything known about the footage of one shoot, in one folder. The montage keeps it in
out/<job>/index/; any other project opens the same folder.

    import embedia
    ix = embedia.open('out/zavod/index')          # or the job folder holding index/
    ix.originals()                                 # the clips taken in, paths of the originals
    ix.get(src, 'people')                          # a feature of a clip, as np.load / json.load gives it
    ix.sql('SELECT clip, t FROM t WHERE people > 2')
    ix.search('welding with sparks', by='image')   # image | sound | caption | speech

Layout (every path from here):
  embedia.toml      what to index: footage folders, camera, which analyzers run (run.py)
  originals.txt     the clips the scan took in, one path per line
  embedia.sqlite    `features`: which step made each clip's features, at which version; `questions`: what was asked
                    (who by screenshots, a main character in words, beautiful views, the shoot's activities) with
                    its input, asked again for clips added later; the timeline tables (timeline.py: t, clips, people,
                    labels, passes, captions, phrases), rebuilt from the features after every run; `meta`: the
                    embedia version and code that built it, when
  proxy/            per clip <stem>_stabilized.mp4: 640 px wide, frame for frame with the original (a Gyroflow render
                    of a clip with motion data, the original scaled down otherwise), and its features beside it:
                    <stem>_stabilized_<feature>.npz | .npy | .json
  cache/            camera motion read from the originals (DJI quaternions, Gyroflow's export of GoPro motion)
  checks/           sheets to confirm a detector by eye once: pilot_check.jpg, passes_check.jpg, who_check.jpg...
  refs/             the reference crops `who` cut from the footage
  voices/, voices.json, transcript.txt     speech.py: the voices told apart, each to listen to
  cameras.json      camera.py's verdict per clip (camera = "auto")
  log/              what the steps printed
Times are seconds of the original (the proxy is frame for frame with it); boxes are shares of the proxy's frame.
"""
import builtins
import datetime
import glob
import json
import os
import sqlite3
import tomllib

import numpy as np

VERSION = '0.1.0'       # embedia's own (pyproject.toml's too): in `meta` and on every feature and question
CONFIG = 'embedia.toml'
DB = 'embedia.sqlite'
# a feature's file next to its proxy: .npz unless listed here
SUFFIX = {'pilot': '.npy', 'pilotclip': '.npy', 'notperson': '.npy', 'pclip': '.npy', 'tid': '.npy',
          'speech': '.json'}
# the tables that stay when timeline.py rebuilds the rest
PERSISTENT = ('CREATE TABLE IF NOT EXISTS features (clip TEXT, step TEXT, version INTEGER, made TEXT, embedia TEXT, '
              'PRIMARY KEY (clip, step))',
              'CREATE TABLE IF NOT EXISTS questions (kind TEXT, name TEXT, input TEXT, made TEXT, embedia TEXT, '
              'PRIMARY KEY (kind, name))',
              'CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)')


def code():
    """The embedia code an index is built with: its git commit, +dirty with uncommitted changes in embedia/; None
    outside a git checkout."""
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        head = subprocess.run(['git', '-C', here, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True,
                              timeout=10)
        if head.returncode:
            return None
        dirty = subprocess.run(['git', '-C', here, 'status', '--porcelain', '--', '.'], capture_output=True,
                               text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return head.stdout.strip() + ('+dirty' if dirty else '')


def open(path):  # noqa: A001  (embedia.open, like sqlite3.connect)
    """The index in `path`: an index folder, a job folder with index/ inside, or an embedia.toml."""
    p = os.path.abspath(path)
    if os.path.isfile(p):
        p = os.path.dirname(p)
    if not os.path.exists(os.path.join(p, CONFIG)) and os.path.exists(os.path.join(p, 'index', CONFIG)):
        p = os.path.join(p, 'index')
    if not os.path.exists(os.path.join(p, CONFIG)):
        raise SystemExit(f'no index in {path}: {CONFIG} missing (python -m embedia init)')
    return Index(p)


def cache_of(render):
    """The camera-motion cache of the index a proxy belongs to (<index>/proxy/x.mp4 -> <index>/cache): what an EDL's
    render paths lead back to."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(render))), 'cache')


def create(root, cfg):
    """A new index folder, or the existing one with its embedia.toml rewritten when `cfg` differs (the montage
    writes it from its job file every run)."""
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, CONFIG)
    text = dumps(cfg)
    old = None
    if os.path.exists(path):
        with _open(path) as fh:
            old = fh.read()
    if old != text:
        with _open(path, 'w') as fh:
            fh.write(text)
    return Index(root)


def _open(path, mode='r'):
    """The built-in open (this module's `open` is embedia.open), for text in UTF-8."""
    return builtins.open(path, mode, encoding='utf-8')


def dumps(cfg):
    """TOML of a flat table: strings, numbers, booleans and lists of strings."""
    def val(v):
        if isinstance(v, bool):
            return 'true' if v else 'false'
        if isinstance(v, (int, float)):
            return repr(v)
        if isinstance(v, (list, tuple)):
            return '[' + ', '.join(val(x) for x in v) + ']'
        return json.dumps(str(v), ensure_ascii=False)
    return ''.join(f'{k} = {val(v)}\n' for k, v in cfg.items() if v is not None)


class Index:
    def __init__(self, root):
        self.root = os.path.abspath(root)
        cp = os.path.join(self.root, CONFIG)
        if os.path.exists(cp):
            with builtins.open(cp, 'rb') as fh:
                self.cfg = tomllib.load(fh)
        else:
            self.cfg = {}
        self.proxy_dir = os.path.join(self.root, 'proxy')
        self.cache_dir = os.path.join(self.root, 'cache')
        self.checks_dir = os.path.join(self.root, 'checks')
        self.log_dir = os.path.join(self.root, 'log')
        self.db_path = os.path.join(self.root, DB)
        self.originals_txt = os.path.join(self.root, 'originals.txt')

    def __repr__(self):
        return f'embedia.Index({self.root!r})'

    # ------------------------------------------------------------------- clips and their files
    def originals(self):
        """Paths of the originals the scan took in (empty before the scan)."""
        if not os.path.exists(self.originals_txt):
            return []
        with _open(self.originals_txt) as fh:
            return [x.strip() for x in fh if x.strip()]

    def write_originals(self, srcs):
        with _open(self.originals_txt, 'w') as fh:
            fh.write(''.join(s + '\n' for s in srcs))

    @staticmethod
    def stem(src):
        """A clip's name: its original's file name without the extension (also given as the name itself)."""
        return os.path.splitext(os.path.basename(src))[0]

    def src(self, clip):
        """The original of a clip given by name (case ignored) or path."""
        k = self.stem(clip).lower()
        for s in self.originals():
            if self.stem(s).lower() == k:
                return s
        raise KeyError(f'{clip}: not a clip of {self.root}')

    def render(self, clip):
        """The clip's proxy: <stem>_stabilized.mp4 (the name stays for any proxy, stabilized or scaled)."""
        return os.path.join(self.proxy_dir, self.stem(clip) + '_stabilized.mp4')

    def renders(self):
        return [self.render(s) for s in self.originals()]

    def base(self, clip):
        """The proxy's path without .mp4: every feature file starts with it."""
        return os.path.splitext(self.render(clip))[0]

    def path(self, clip, feature):
        """Where a clip's feature lives: <base>_<feature>.npz (.npy, .json: SUFFIX; subject_NAME is .npy)."""
        ext = SUFFIX.get(feature, '.npy' if feature.startswith('subject') else '.npz')
        return f'{self.base(clip)}_{feature}{ext}'

    def has(self, clip, feature):
        return os.path.exists(self.path(clip, feature))

    def get(self, clip, feature):
        """The feature as np.load or json.load gives it, or None when the clip does not have it."""
        p = self.path(clip, feature)
        if not os.path.exists(p):
            return None
        if p.endswith('.json'):
            with _open(p) as fh:
                return json.load(fh)
        return np.load(p, allow_pickle=False)

    def features(self, clip):
        """The feature names a clip has files for."""
        b = os.path.basename(self.base(clip)) + '_'
        out = []
        for p in glob.glob(glob.escape(self.base(clip)) + '_*'):
            n, ext = os.path.splitext(os.path.basename(p)[len(b):])
            if ext in ('.npz', '.npy', '.json'):
                out.append(n)
        return sorted(out)

    # ------------------------------------------------------------------- the database
    def db(self):
        """The index's SQLite database, with its persistent tables there."""
        os.makedirs(self.root, exist_ok=True)
        con = sqlite3.connect(self.db_path, timeout=60)
        for s in PERSISTENT:
            con.execute(s)
        for t in ('features', 'questions'):                      # an index made before the column
            if 'embedia' not in [r[1] for r in con.execute(f'PRAGMA table_info({t})')]:
                con.execute(f'ALTER TABLE {t} ADD COLUMN embedia TEXT')
        return con

    def stamp(self):
        """`meta`: which embedia built the index (version, code) and when (created, built)."""
        now = datetime.datetime.now().isoformat(timespec='seconds')
        with self.db() as con:
            con.execute("INSERT OR IGNORE INTO meta VALUES ('created', ?)", (now,))
            con.executemany('INSERT OR REPLACE INTO meta VALUES (?, ?)',
                            [('embedia_version', VERSION), ('embedia_code', code()), ('built', now)])

    def meta(self):
        with self.db() as con:
            return dict(con.execute('SELECT key, value FROM meta').fetchall())

    def sql(self, query, params=()):
        """Rows of a query as dicts."""
        with self.db() as con:
            cur = con.execute(query, params)
            head = [d[0] for d in cur.description] if cur.description else []
            return [dict(zip(head, r)) for r in cur.fetchall()]

    def made(self, step):
        """{clip: version} of the step's features as recorded."""
        with self.db() as con:
            return dict(con.execute('SELECT clip, version FROM features WHERE step = ?', (step,)).fetchall())

    def record(self, step, version, clips):
        now = datetime.datetime.now().isoformat(timespec='seconds')
        with self.db() as con:
            con.executemany('INSERT OR REPLACE INTO features (clip, step, version, made, embedia) VALUES (?,?,?,?,?)',
                            [(self.stem(c), step, version, now, VERSION) for c in clips])

    def questions(self):
        """What was asked of the footage: [{kind, name, input, made}], input as given."""
        with self.db() as con:
            rows = con.execute('SELECT kind, name, input, made FROM questions ORDER BY made').fetchall()
        return [dict(kind=k, name=n, input=json.loads(i), made=m) for k, n, i, m in rows]

    def question(self, kind, name=''):
        q = [x for x in self.questions() if x['kind'] == kind and x['name'] == name]
        return q[0] if q else None

    def remember(self, kind, name, inp):
        now = datetime.datetime.now().isoformat(timespec='seconds')
        with self.db() as con:
            con.execute('INSERT OR REPLACE INTO questions (kind, name, input, made, embedia) VALUES (?,?,?,?,?)',
                        (kind, name, json.dumps(inp, ensure_ascii=False, sort_keys=True), now, VERSION))

    # ------------------------------------------------------------------- search in words
    def search(self, text, by='image', top=5, clip_cache=None):
        """Where the footage shows / sounds like / is captioned as / says `text`, best first:
        [{clip, t0, t1, score}] (+ text, voice for speech). image: CLIP on the frames, "A -> B -> C" a route
        (places.find, its best piece only); sound: CLAP (sounds.py); caption: the captions' words (captions.py,
        its best stretch); speech: the phrases' words (speech.py)."""
        renders = [r for r in self.renders() if os.path.exists(r)]
        name = {r: self.stem(s) for s, r in zip(self.originals(), self.renders())}
        if by == 'image':
            from embedia import paths, places
            hit = places.find(renders, text, clip_cache or os.path.join(paths.MODELS, 'clip'))
            return [] if hit is None else [dict(clip=name[hit[0]], t0=hit[1], t1=hit[2], score=float(hit[3]))]
        if by == 'sound':
            from embedia import sounds
            pairs = [(s, r) for s, r in zip(self.originals(), self.renders()) if os.path.exists(r)]
            half = sounds.PIECE_S / 2
            return [dict(clip=name[r], t0=max(t - half, 0.0), t1=t + half, score=float(p))
                    for p, r, t in sounds.moments(pairs, text, top=top)]
        if by == 'caption':
            from embedia import captions
            hit = captions.find(renders, text)
            return [] if hit is None else [dict(clip=name[hit[0]], t0=hit[1], t1=hit[2], score=float(len(hit[3])))]
        if by == 'speech':
            from embedia import speech
            return [dict(clip=name[r], t0=t0, t1=t1, score=float(sh), text=tx, voice=v)
                    for r, t0, t1, tx, v, sh in speech.find(renders, text, top=top)]
        raise ValueError(f'by={by!r}: image, sound, caption or speech')
