"""embedia from the command line, run from the folder above it (`embedia` stands for the name of its folder):

    python -m embedia init INDEX --footage DIR [DIR ...] [--camera drone|carried|auto] [--set KEY=VALUE ...]
    python -m embedia run INDEX [--join]        scan, proxies, features, questions again, timeline
    python -m embedia scan INDEX [--join]       (--join: split DJI recordings are joined; ask the owner first)
    python -m embedia proxies INDEX             every clip's proxy (scaled ones go again after a run without
                                                    stabilize = true)
    python -m embedia analyze INDEX [--only STEP] [--redo STEP]     features, questions again, timeline
    python -m embedia status INDEX              clips, which steps are done, the questions asked
    python -m embedia schema INDEX              the database's tables, columns, rows (README: what they mean)
    python -m embedia sql INDEX "SELECT ..." [--limit N]
    python -m embedia search INDEX "words" [--by image|sound|caption|speech] [--top N]
    python -m embedia who INDEX NAME=SCREENSHOT [NAME=CLIP:SECONDS:X ...]   (_NAME: somebody else)
    python -m embedia subject INDEX NAME --text "a woman pushing a pram" [--not ...]
    python -m embedia beauty INDEX --text "mountains" [--plain "snow up close" ...]
    python -m embedia activities INDEX [--text "welding with sparks" ...] [--only]   (--only: these replace
                                                                                          the general list)
    python -m embedia questions INDEX

INDEX is a folder with embedia.toml (or a job folder holding index/). A question is kept with its input and asked
again for clips added later; asking with another input replaces the answers.
"""
import argparse
import os
import sys

if __package__ in (None, ''):                                    # run as a file: the package of its folder
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

from . import index  # noqa: E402
from . import run as er  # noqa: E402
from . import timeline  # noqa: E402


def value(v):
    """A --set value as TOML would read it: true/false, a number, else the text."""
    if v.lower() in ('true', 'false'):
        return v.lower() == 'true'
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    return v


def status(ix):
    srcs = ix.originals()
    have = sum(os.path.exists(ix.render(s)) for s in srcs)
    kinds = [ix.frame_kind(s) or 'not measured' for s in srcs]
    m = ix.meta()
    print(f'{ix.root}: {len(srcs)} clips (' + ', '.join(f'{kinds.count(k)} {k}' for k in sorted(set(kinds)))
          + f'), {have} proxies kept; built {m.get("built", "never")} by '
          f'{m.get("embedia_package", "embedia")} {m.get("embedia_version", "?")} ({m.get("embedia_code") or "no git"}); '
          f'this is {index.MADE_BY}')
    for st in er.STEPS:
        if not st.on(ix.cfg):
            continue
        try:
            mine = er.step_clips(ix, st, er.camera_kinds(ix) if st.clips == 'flown' else None)
        except (OSError, SystemExit):
            mine = srcs
        made = ix.made(st.name)
        done = sum(made.get(ix.stem(s)) == st.version for s in mine)
        print(f'  {st.name:10s} {done}/{len(mine)}')
    for q in ix.questions():
        print(f'  asked {q["kind"]} {q["name"]}: {q["input"]} ({q["made"]})')


def main():
    ap = argparse.ArgumentParser(prog=f'python -m {__package__}', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('init')
    p.add_argument('index')
    p.add_argument('--footage', nargs='+', required=True)
    p.add_argument('--camera', choices=('drone', 'carried', 'auto'), default='drone')
    p.add_argument('--set', action='append', default=[], help='KEY=VALUE of embedia.toml (captions=true)')
    for c in ('run', 'scan'):
        p = sub.add_parser(c)
        p.add_argument('index')
        p.add_argument('--join', action='store_true')
    for c in ('proxies', 'status', 'questions', 'schema'):
        sub.add_parser(c).add_argument('index')
    p = sub.add_parser('analyze')
    p.add_argument('index')
    p.add_argument('--only', action='append', default=None, choices=[s.name for s in er.STEPS])
    p.add_argument('--redo', action='append', default=[], choices=[s.name for s in er.STEPS])
    p = sub.add_parser('sql')
    p.add_argument('index')
    p.add_argument('query')
    p.add_argument('--limit', type=int, default=40)
    p = sub.add_parser('search')
    p.add_argument('index')
    p.add_argument('text')
    p.add_argument('--by', choices=('image', 'sound', 'caption', 'speech'), default='image')
    p.add_argument('--top', type=int, default=5)
    p = sub.add_parser('who')
    p.add_argument('index')
    p.add_argument('refs', nargs='+', help='NAME=IMAGE or NAME=CLIP:SECONDS:X')
    p = sub.add_parser('subject')
    p.add_argument('index')
    p.add_argument('name')
    p.add_argument('--text', action='append', required=True)
    p.add_argument('--not', dest='nots', action='append', default=[])
    p = sub.add_parser('beauty')
    p.add_argument('index')
    p.add_argument('--text', action='append', required=True)
    p.add_argument('--plain', action='append', default=[])
    p = sub.add_parser('activities')
    p.add_argument('index')
    p.add_argument('--text', action='append', default=[])
    p.add_argument('--only', action='store_true', help='the --text activities replace the general list')
    a = ap.parse_args()

    if a.cmd == 'init':
        cfg = dict(footage=[os.path.abspath(f) for f in a.footage], camera=a.camera)
        for kv in a.set:
            k, _, v = kv.partition('=')
            cfg[k.strip()] = value(v.strip())
        ix = index.create(a.index, cfg)
        print(f'{ix.root}: embedia.toml written')
        return
    ix = index.open(a.index)
    if a.cmd == 'run':
        er.everything(ix, a.join)
    elif a.cmd == 'scan':
        er.scan(ix, a.join)
    elif a.cmd == 'proxies':
        er.proxies(ix)
    elif a.cmd == 'analyze':
        er.analyze(ix, a.only, a.redo)
    elif a.cmd == 'status':
        status(ix)
    elif a.cmd == 'schema':
        with ix.db() as con:
            for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE '%fts%' "
                                    "AND name NOT LIKE 'sqlite%' ORDER BY name"):
                cols = [r[1] for r in con.execute(f'PRAGMA table_info({t})')]
                n = con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
                print(f'{t} ({n}): {", ".join(cols)}')
    elif a.cmd == 'questions':
        for q in ix.questions():
            print(f'{q["kind"]} {q["name"]}: {q["input"]} ({q["made"]})')
    elif a.cmd == 'sql':
        timeline.ask(ix, a.query, a.limit)
    elif a.cmd == 'search':
        for h in ix.search(a.text, a.by, a.top):
            extra = f'  {h["voice"]}: {h["text"]}' if 'text' in h else ''
            print(f'{h["clip"]}:{h["t0"]:.1f}-{h["t1"]:.1f}  {h["score"]:.3f}{extra}')
    else:
        if a.cmd == 'who':
            inp = {}
            for r in a.refs:
                n, _, p = r.partition('=')
                inp.setdefault(n, []).append(os.path.abspath(p) if os.path.exists(p) else p)
            name = ''
        elif a.cmd == 'subject':
            inp, name = dict(texts=a.text, nots=a.nots), a.name
        elif a.cmd == 'beauty':
            inp, name = dict(texts=a.text, plain=a.plain), ''
        else:
            if a.only and not a.text:
                raise SystemExit('activities --only: give the activities with --text')
            inp, name = (dict(texts=a.text, only=True) if a.only else dict(texts=a.text)), ''
        if er.ask(ix, a.cmd, inp, name):
            er.tidy(ix)
            er.timeline(ix)
        else:
            print(f'{a.cmd} {name}: asked before with this input, answered for every clip')


if __name__ == '__main__':
    main()
