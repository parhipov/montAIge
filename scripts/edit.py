"""The edit: cut stabilized footage to a music track; shots chosen by the camera's motion, how it is flown,
who and what is in view.

    python edit.py MUSIC.mp3 --originals originals.txt --proxy-dir PROXY_DIR -o edit.mp4 --target 60
        --people 4 [--fly 1] [--must CLIP:SEC ...] [--end CLIP:START-END] [--seed N]

Music: a constant-tempo beat grid, bars and 16-beat phrases with an energy class each; --target shortens the
track by whole phrases (music_edit.py), keeping the intro and the real ending; --bpm when the detector picks
a wrong multiple. Each phrase is a scene from one clip; its shots jump forward inside that clip.

Footage: every original listed in --originals (DJI `djmd` or GoPro motion from a Gyroflow export) with its
whole-clip Gyroflow render in PROXY_DIR (same frame count). Telemetry from the original, pictures and image
features (analyze.py, people.py, objects.py, places.py) from the render.

Score of a shot window: piloting (closeness from image parallax, lean, rotation; cinematic for quiet music,
action for loud), music energy against motion intensity, motion onsets on beats, people (--people W: by size,
x1.3 facing the camera, x1.6 waving; the pilot is rationed), variety (movement kinds, look-alike thumbnails,
--theme activities), distance in time from other scenes. Same inputs and seed give the same edit; --seed N
adds seeded noise for another variant.

Editor's input: --must / --start / --end CLIP:SEC or CLIP:START-END (or find:TEXT through places.py) put a
moment or a piece in the edit; a piece longer than its phrase plays faster, up to PIECE_MAX_SPEED. --fly N:
flying scenes built around a manoeuvre on the biggest music lifts. --heroes N with --ramps: the classic ramp
(rush in, the manoeuvre slow, back to speed). --want KIND, --arcs N: climbs and up-and-down arcs. A series
from one shoot: --avoid EDL, --subject / --minor / --focus / --peers for edits about one person.

Output: the preview (640 px wide by default, 25 fps), <out>_edl.json (everything final.py needs for the full
quality render), <out>_plan.png.
"""
import argparse
import glob
import json
import os
import sys
import time

import av
import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, find_peaks, sosfiltfilt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import music as mus  # noqa: E402

ROLL_WEIGHT = 0.7       # roll is partly flattened by stabilization
VIS_LP_HZ = 1.5         # what survives Gyroflow smoothing, roughly
ACC_HEIGHT, ACC_PROM = 60.0, 40.0   # motion onset peaks, deg/s^2
ACC_REACH = 6           # frames: sync tolerance while choosing shots (snapping does the rest)
TRICK_DEG = 250.0       # rotation within 1 s that counts as a manoeuvre
DIVE_DEG = 45.0         # camera this far below the horizon counts as a dive
UP = np.array([0.0, 0.0, -1.0])     # world up in DJI's fused frame (image-down agrees 96-99 %)
# on the ground, for motion from the picture: mean flow under this (px per frame pair, analyze.py, 160 px wide).
# GH010401 against its gyro: on the ground median 0.07 (people walking by: p90 0.5), in the air p1 0.11, p5 0.16;
# at 0.1 nothing of the flight is taken for the ground and 41 % of the ground is missed, at 0.2 2 % and 28 %
STILL_FLOW = 0.12
# landings: LANDING_S before a stretch on the ground (Flight._usable). Every Dagestan clip ended in a slow
# approach, a flop at the touchdown (40-190 deg/s, read as a roll) and the ground; the owner called the 8 s before a
# touchdown among people "a dull landing, not the point", and "a landing can be an ending": a window pays W_LANDING x
# its landing share anywhere but in the last scene
LANDING_S, W_LANDING = 10.0, 6.0
SKILL_PARTS = dict(parallax=0.45, lean=0.25, rotation=0.30)       # action: loud music
CINE_PARTS = dict(parallax=0.55, lean=0.30, smooth=0.15)          # cinematic: quiet music
PULLUP_DEG, PASS_RANK = 50.0, 0.985    # camera tilts up this much in 1 s; parallax percentile of a close pass
# descent from height: camera this far below the horizon (2 s mean) for DESCENT_S while everything is far
# (parallax rank under DESCENT_FAR); the manoeuvre is its middle, so a shot shows the way down and the
# next one in the scene the arrival
DESCENT_DEG, DESCENT_S, DESCENT_FAR = 18.0, 1.5, 0.5
# ascent (a climb that opens up the place below): there is no altitude in the telemetry, so it comes from the
# picture. CLIP (places.py embeddings) "a view from high above" against ground-level views rises from under
# ASC_LOW to over ASC_HIGH within ASC_WITHIN_S, and for the next 2.5 s stays over ASC_KEEP with everything far
# (parallax rank under ASC_FAR). Village: all 7 climbs seen on the check sheet and 3 more, nothing else.
# Flying close over the shed roof or a chimney against the sky read "from above" too, until the sky and a
# roof up close were among the alternatives; the camera angle does not help (-8..+6 deg on most climbs)
ASC_HIGH_TEXTS = ['an aerial view of a village from high above', 'a drone view from high in the sky']
ASC_LOW_TEXTS = ['a garden seen from the ground', 'grass and bushes up close', 'a house seen from the ground',
                 'people standing in a garden', 'the sky', 'a roof up close']
ASC_LOW, ASC_HIGH, ASC_KEEP, ASC_WITHIN_S, ASC_FAR = 0.3, 0.6, 0.65, 8.0, 0.5
MOVES = ('dive', 'pull-up', 'pitch-down', 'turn-a', 'turn-b', 'roll', 'pass', 'cruise')
EVENT_KINDS = ('roll', 'spin', 'loop', 'dive', 'pull-up', 'pass', 'descent', 'ascent')
# --want KIND: the edit asks for this manoeuvre kind; a shot with one in the middle half of it gets W_WANT
W_WANT, WANT = 3.0, []
# with climbs asked for: CLIP height (the ascents' probability) in the cinematic piloting score
CINE_HEIGHT = 0.25

# shot scores
W_SKILL, W_INT, W_SYNC, W_OFF, W_NEAR, W_HERO, NOISE = 4.0, 0.5, 1.5, 0.3, 2.0, 6.0, 0.25
# W_NEAR: footage close in time to a shot of another scene, exp(-gap / NEAR_TAU_S); at 8 s and 0.5 two
# passes over the same warm-up 17 s apart (split files of one recording) opened the tennis "game" edit
NEAR_TAU_S = 20.0
# people (objects.py / YOLO): a person covering PEOPLE_FULL of the frame counts fully, smaller ones less
W_PEOPLE, PEOPLE_FULL = 0.0, 0.02
# people-first edits (--people): a shot with nobody in it outside the flying scenes; the pilot alone
# (worth PILOT_WORTH at most) does not count as company
W_EMPTY, EMPTY_BELOW = 4.0, 0.35
# the pilot: worth this much of another person; once PILOT_BUDGET of the edit shows the pilot alone,
# every further pilot-alone shot pays W_PILOT_OVER x its pilot-alone share
PILOT_WORTH, PILOT_BUDGET, W_PILOT_OVER = 0.3, 0.15, 3.0
SUBJECT_BOOST = 2.0     # the film's main character (people.py subject) against anybody else
SUBJECT_TAG = ['']      # --subject NAME: <render>_subject_NAME.npy instead of <render>_subject.npy
MINOR_TAG = ['']        # --minor NAME: that main character (of another edit) counts like the pilot here
# --focus: an edit about the main character; everybody else is worth this much of a person (like the pilot),
# and the main character counts fully only from FOCUS_FULL of the frame (close-ups first; a child far
# away at 2 % of the frame is worth 0.45 of one up close)
FOCUS_OTHERS, FOCUS, FOCUS_FULL = 0.3, [False], 0.10
# ... and a shot with the main character in less than FOCUS_PRESENT of it is empty there (W_EMPTY), however
# many other children it shows: the first per-child films had other kids in 4 of 19 shots, the ending too
FOCUS_PRESENT = 0.5
# ... but people of another kind (adults, the coach) or the place make a fair cutaway: without the main
# character a shot pays only W_AWAY; what it must not do is show a look-alike instead (another child of the
# series, or someone `people.py split` dropped as dressed like them: --peers, <subject>_not by default).
# A peer covering IMPOSTOR_AREA of the frame or more, with the main character not in the picture, is a
# stand-in: W_IMPOSTOR x the share of the shot showing one.
W_AWAY, W_IMPOSTOR, IMPOSTOR_AREA, PEER_TAGS = 2.0, 8.0, 0.01, []
ANIMAL_WORTH = 1.0      # a cat or a dog (objects.py) against a person of the same size
# flying scenes (--fly N): on the biggest music lifts people do not count, piloting counts W_FLY extra
MAX_FLY, W_FLY = 0, 3.0
# editor's moments: a window that shows the moment (from 0.5 s before to 1 s after) gets W_MUST
W_MUST, MUST_OPEN_S, END_OPEN_S, END_SET = 50.0, 3.0, 8.0, [False]
# an editor's piece longer than a whole phrase plays faster, up to ABS_HI (the ramps' ceiling): a 25 s climb
# and descent from height (village) fits a 15 s phrase at 1.7x; PIECE_AIR_S of picture around it
PIECE_AIR_S = 0.5
END_AFTER_S = 2.0       # picture after the --end moment: the fade-out (FADE_OUT_S)
PILOT_USED = [0.0, 1.0]     # seconds of pilot-alone shots placed so far, edit length (set by assign)
# movement variety: same movement signature as a shot placed within ~MOVE_TAU_S of music
W_MOVE, MOVE_TAU_S, MOVE_SIM0 = 2.0, 25.0, 0.6
W_OPEN, MAX_HEROES, HERO_REUSE = 1.5, 0, 0.3
PROP_KEPT = 0.08        # share of red pixels in a bottom corner of the kept band that counts as props in view
# looking like an earlier scene (12x9 Lab thumbnails, RMS distance); the scale follows the footage:
# sigma = this fraction of the median distance between random frames (construction: 39 -> 22)
W_LOOK, LOOK_SIGMA_FRAC = 1.5, 0.56
LOOK_SIGMA = 22.0
# themes in words (--theme "TEXT[:N]"): what the edit shows, each in at most N scenes (default 1). CLIP on the
# frames (places.py embeddings), softmax over the themes and places.GENERIC_NOT. A scene shows a theme when one
# of its shots has a mean probability of THEME_HIT or more; once a theme has its N scenes, a candidate pays
# W_THEME, in full where a frame's probability is THEME_FULL or more, nothing at THEME_LOW. Village: the old man
# on the shed roof was in 5 of 8 shots, from angles the thumbnails (W_LOOK) took for different pictures; the
# themes score 0.5-0.9 on their shots, an open garden 0.36 at most. W_THEME 4 on a linear ramp from 0.3 left
# 3 roof scenes of 1: the old man is a person, and people (W_PEOPLE 4 x 1-1.6) outweighed it
W_THEME, THEME_HIT, THEME_LOW, THEME_FULL = 8.0, 0.4, 0.35, 0.55
THEME_MIN_S = 2.0       # a shot this short does not use up a theme's scenes (a 1.2 s pickup took the lawn mower)
# beautiful views (--beauty W, off by default; beauty.py in the job's own words): per frame the view is the mean of
# the aesthetic rank (LAION, across all footage) and the probability of the job's beautiful views. A window pays
# W_BEAUTY x its view everywhere; in a people edit a beautiful view excuses a shot with nobody in it (W_EMPTY x
# (1 - view)); the cinematic piloting score (quiet music) takes CINE_VIEW of the view
W_BEAUTY, CINE_VIEW = 0.0, 0.3
CHAPTERS = []           # --chapters N,N,...: scenes per recording day in date order, instead of chapters() sharing them
THEMES = []             # [(text, scenes allowed)]
# scene: cuts inside it jump 1..12 s forward; continuity of rotation across them
GAP_MIN_S, GAP_MAX_S, GAP_STEP_S, W_CONT, W_SAME_SCENE = 1.0, 12.0, 0.2, 1.5, 1.0
# speed ramps (relative to the shot's base speed) and the hero ramp
RAMP_LO, RAMP_HI, ABS_LO, ABS_HI, W_SNAP, W_DEV = 0.7, 1.4, 0.5, 1.7, 1.0, 6.0
PIECE_MAX_SPEED = ABS_HI
HERO_RUSH, HERO_SLOW, HERO_AT = 1.4, 0.5, 2.0      # beat 0 fast, beats 1-2 slow, manoeuvre on beat 2
BASE_SPEED = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
BASE_SPEED_PACE = {1: 1.1, 2: 1.15, 3: 1.2}         # --pace: faster flying, hesitation hidden (nov2023)
FADE_IN_S, FADE_OUT_S = 0.6, 2.0
# --grade auto: per shot, after the LUT: luma p0.5 -> GRADE_BLACK, p99.5 -> GRADE_WHITE (gain within
# GRADE_GAIN), and the near-white pixels (top luma decile, low chroma: walls, ceilings) pulled GRADE_WB of
# the way to neutral (each channel within GRADE_WB_MAX)
GRADE_BLACK, GRADE_WHITE, GRADE_GAIN, GRADE_WB, GRADE_WB_MAX = 6.0, 242.0, (0.8, 1.6), 0.8, (0.85, 1.2)
# one grade for the whole shot unless its thirds differ by this much (black in, any gain)
GRADE_STEADY = (8.0, 0.08)


# =================================================================== footage
def frame_count(path):
    with av.open(path) as c:
        s = c.streams.video[0]
        return s.frames, float(s.average_rate)


def rates_from_quats(q, fps):
    """Camera angular velocity (deg/s, x=tilt y=pan z=roll) between consecutive frames."""
    a, b = q[:-1], q[1:]
    b = b * np.where(np.sum(a * b, axis=1) < 0, -1.0, 1.0)[:, None]
    aw, ax, ay, az = a[:, 0], -a[:, 1], -a[:, 2], -a[:, 3]        # conj(a)
    bw, bx, by_, bz = b.T
    w = aw * bw - ax * bx - ay * by_ - az * bz
    v = np.stack([aw * bx + ax * bw + ay * bz - az * by_,
                  aw * by_ - ax * bz + ay * bw + az * bx,
                  aw * bz + ax * by_ - ay * bx + az * bw], 1)
    sn = np.linalg.norm(v, axis=1)
    ang = 2 * np.arctan2(sn, np.clip(w, -1, 1))
    rv = v / np.maximum(sn, 1e-12)[:, None] * ang[:, None]
    rates = np.degrees(rv) * fps
    return np.vstack([rates, rates[-1:]])


def rotate(q, v):
    """Camera-frame vector v in world coordinates for every orientation q (N, 4 wxyz)."""
    w, x, y, z = q.T
    u = np.stack([x, y, z], 1)
    t = 2 * np.cross(u, v)
    return v + w[:, None] * t + np.cross(u, t)


def load_telemetry(src, cache_dir):
    st = os.stat(src)
    key = f'{os.path.splitext(os.path.basename(src))[0]}_{st.st_size}_{int(st.st_mtime)}_q.npz'
    cp = os.path.join(cache_dir, key)
    if os.path.exists(cp):
        z = np.load(cp)
        return z['q'].astype(np.float64), float(z['fps'])
    import djimeta
    n, fps = frame_count(src)
    q = djimeta.per_frame(src, n)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(cp, q=q.astype(np.float32), fps=fps)
    return q, fps


def load_flow(stab, proxy_dir, n):
    """Image motion features of the proxy pass on the full frame grid, or None."""
    if not proxy_dir:
        return None
    p = os.path.join(proxy_dir, os.path.splitext(os.path.basename(stab))[0] + '_flow.npz')
    if not os.path.exists(p):
        return None
    z = np.load(p)
    fr = z['frame'] - 1.0                  # flow between frames f-2 and f is centred on f-1
    i = np.arange(n)
    out = {k: np.interp(i, fr, z[k]) for k in ('flow', 'resid50', 'resid90', 'div', 'bottom', 'bright')}
    pp = os.path.join(proxy_dir, os.path.splitext(os.path.basename(stab))[0] + '_props.npz')
    if os.path.exists(pp):                 # props.py: propellers inside the band the final crop keeps
        zp = np.load(pp)
        if 'kept' in zp.files:
            out['props'] = np.interp(i, zp['frame'], zp['kept'])
    return out


def load_people_pose(base, n):
    """people.py data: people other than the pilot, by size, x1.3 facing the camera, x1.6 waving;
    approach / recede of the biggest of them +0.5; passes between legs / under / over +1.5, past +0.5;
    the pilot alone in the picture -0.5 at most. None without people.py output."""
    if not (os.path.exists(base + '_people.npz') and os.path.exists(base + '_pilot.npy')):
        return None
    import objects
    z = np.load(base + '_people.npz')
    pil = np.load(base + '_pilot.npy')
    notp = np.load(base + '_notperson.npy') if os.path.exists(base + '_notperson.npy') else np.zeros(len(pil))
    sf = subject_file(base)
    subj = np.load(sf) if os.path.exists(sf) else np.zeros(len(pil), bool)
    mf = base + '_subject_' + MINOR_TAG[0] + '.npy' if MINOR_TAG[0] else ''
    minor = np.load(mf) if mf and os.path.exists(mf) else np.zeros(len(pil), bool)
    samples, fps = z['samples'], float(z['fps'])
    k = np.searchsorted(samples, z['frame'])
    area = z['box'][:, 2] * z['box'][:, 3]
    other_a, pilot_a, n_other = np.zeros(len(samples)), np.zeros(len(samples)), np.zeros(len(samples))
    subj_a = np.zeros(len(samples))
    boost = np.ones(len(samples))
    for i, a, p, wv, fc, nq, sb, mn in zip(k, area, pil, z['wave'], z['face'], notp, subj, minor):
        if nq > 0.5:                                             # not a person after all (sand, sky, tree)
            continue
        if p or (mn and not sb):                                 # the pilot, or another edit's main character
            pilot_a[i] = max(pilot_a[i], a)
            continue
        if sb:
            subj_a[i] = max(subj_a[i], a)
            if FOCUS[0]:
                continue
        n_other[i] += 1
        if a > other_a[i]:
            other_a[i] = a
        boost[i] = max(boost[i], SUBJECT_BOOST if sb else 1.6 if wv else 1.3 if fc else 1.0)
    v = np.where(n_other > 0, np.minimum(1.0, np.sqrt(other_a / PEOPLE_FULL)) * boost, 0.0)
    if FOCUS[0]:                                                 # the main character's own size and boost
        v = SUBJECT_BOOST * np.minimum(1.0, np.sqrt(subj_a / FOCUS_FULL)) + FOCUS_OTHERS * v
        n_other = n_other + (subj_a > 0)
    # the pilot is welcome, just worth less than everybody else (and rationed by PILOT_BUDGET in assign)
    v += np.where(n_other == 0, PILOT_WORTH * np.minimum(1.0, np.sqrt(pilot_a / PEOPLE_FULL)), 0.0)
    out = np.interp(np.arange(n), samples, v)
    bump = np.zeros(n)
    half = int(0.5 * fps)
    zz = dict(frame=samples, fps=fps, person_n=n_other, person_area=other_a)
    for f0, kind, _ in objects.people_motion(zz):
        if kind in ('approach', 'recede'):
            bump[max(f0 - half, 0):f0 + half + 1] = np.maximum(bump[max(f0 - half, 0):f0 + half + 1], 0.5)
    if os.path.exists(base + '_passes.npz'):
        pz = np.load(base + '_passes.npz')
        for f0, kind in zip(pz['frame'], pz['kind']):
            w = 1.5 if kind in ('between legs', 'under', 'over') else 0.5 if str(kind).startswith('past') else 0.0
            bump[max(f0 - half, 0):f0 + half + 1] = np.maximum(bump[max(f0 - half, 0):f0 + half + 1], w)
    return np.maximum(out, load_animals(base, n)) + bump


def load_animals(base, n):
    """Per frame: a cat, dog, bird... in view (objects.py), by size like a person, or zeros."""
    if not os.path.exists(base + '_objects.npz'):
        return np.zeros(n)
    z = np.load(base + '_objects.npz')
    v = np.where(z['animal_n'] > 0, ANIMAL_WORTH * np.minimum(1.0, np.sqrt(z['animal_area'] / PEOPLE_FULL)), 0.0)
    return np.interp(np.arange(n), z['frame'], v)


def subject_file(base):
    return base + '_subject' + (f'_{SUBJECT_TAG[0]}' if SUBJECT_TAG[0] else '') + '.npy'


def load_subject(stab, proxy_dir, n):
    """Per frame 0/1: the main character (people.py subject) is in the picture, or zeros."""
    base = os.path.join(proxy_dir, os.path.splitext(os.path.basename(stab))[0]) if proxy_dir else ''
    if not (base and os.path.exists(base + '_people.npz') and os.path.exists(subject_file(base))):
        return np.zeros(n)
    z = np.load(base + '_people.npz')
    sb = np.load(subject_file(base))
    if os.path.exists(base + '_pilot.npy'):
        sb = sb & ~np.load(base + '_pilot.npy')
    has = np.zeros(len(z['samples']))
    has[np.searchsorted(z['samples'], z['frame'])[sb]] = 1
    return np.interp(np.arange(n), z['samples'], has)


def load_stand_ins(stab, proxy_dir, n):
    """Per frame 0/1: a peer of the main character (PEER_TAGS) is in the picture at IMPOSTOR_AREA or more,
    and the main character is not (with them in it, it is a group shot: fine)."""
    base = os.path.join(proxy_dir, os.path.splitext(os.path.basename(stab))[0]) if proxy_dir else ''
    if not (base and PEER_TAGS and os.path.exists(base + '_people.npz')):
        return np.zeros(n)
    z = np.load(base + '_people.npz')
    area = z['box'][:, 2] * z['box'][:, 3]
    k = np.searchsorted(z['samples'], z['frame'])
    hero = np.load(subject_file(base)) if os.path.exists(subject_file(base)) else np.zeros(len(area), bool)
    peer = np.zeros(len(area), bool)
    for t in PEER_TAGS:
        p = base + '_subject_' + t + '.npy'
        if os.path.exists(p):
            peer |= np.load(p)
    peer &= ~hero
    ha, pa = np.zeros(len(z['samples'])), np.zeros(len(z['samples']))
    np.maximum.at(ha, k[hero], area[hero])
    np.maximum.at(pa, k[peer], area[peer])
    return np.interp(np.arange(n), z['samples'], ((pa >= IMPOSTOR_AREA) & (ha == 0)).astype(float))


def load_pilot_alone(stab, proxy_dir, n):
    """Per frame 0/1: the pilot leads the picture, in it and bigger than anybody else (people.py), or zeros.
    Village: the pilot standing up front with granny on the swing far behind read as a shot of the pilot, but
    "nobody else in it" let ~10 s of them through the PILOT_BUDGET in a 48 s edit."""
    base = os.path.join(proxy_dir, os.path.splitext(os.path.basename(stab))[0]) if proxy_dir else ''
    if not (base and os.path.exists(base + '_people.npz') and os.path.exists(base + '_pilot.npy')):
        return np.zeros(n)
    z = np.load(base + '_people.npz')
    pil = np.load(base + '_pilot.npy').astype(bool)
    k = np.searchsorted(z['samples'], z['frame'])
    area = z['box'][:, 2] * z['box'][:, 3]
    pa, oa = np.zeros(len(z['samples'])), np.zeros(len(z['samples']))
    np.maximum.at(pa, k[pil], area[pil])
    np.maximum.at(oa, k[~pil], area[~pil])
    return np.interp(np.arange(n), z['samples'], ((pa > 0) & (pa > oa)).astype(float))


def load_people(stab, proxy_dir, n):
    """Per-frame people value (people.py if present, else objects.py: is there a person, how big), or None."""
    if not proxy_dir:
        return None
    pose = load_people_pose(os.path.join(proxy_dir, os.path.splitext(os.path.basename(stab))[0]), n)
    if pose is not None:
        return pose
    p = os.path.join(proxy_dir, os.path.splitext(os.path.basename(stab))[0] + '_objects.npz')
    if not os.path.exists(p):
        return None
    z = np.load(p)
    v = np.where(z['person_n'] > 0, np.minimum(1.0, np.sqrt(z['person_area'] / PEOPLE_FULL)), 0.0)
    out = np.interp(np.arange(n), z['frame'], v)
    # flying up to someone or pulling away from them is worth more than someone just standing there
    import objects
    bump = np.zeros(n)
    half = int(0.5 * float(z['fps']))
    for f0, kind, _ in objects.people_motion(z):
        if kind in ('approach', 'recede'):
            bump[max(f0 - half, 0):f0 + half + 1] = 0.5
    return out + bump


def load_beauty(stab, n):
    """Aesthetic score and probability of the job's beautiful views (beauty.py) on the full frame grid, or None."""
    p = os.path.splitext(stab)[0] + '_beauty.npz'
    if not os.path.exists(p):
        return None
    z = np.load(p)
    i = np.arange(n)
    return np.interp(i, z['frame'], z['aes']), np.interp(i, z['frame'], z['scenic'])


def set_beauty(flights):
    """f.view per frame (see W_BEAUTY) and its share of the cinematic score; flights without beauty.py get none."""
    seen = [f for f in flights if getattr(f, 'beauty', None) is not None]
    if not seen:
        print('  note: --beauty without beauty.py output (job file: beauty = [...])')
        return
    pool = np.sort(np.concatenate([f.beauty[0][f.usable] for f in seen]))
    words = any(f.beauty[1].max() > 0 for f in seen)
    for f in seen:
        rank = prank(f.beauty[0], pool)
        f.view = 0.5 * rank + 0.5 * f.beauty[1] if words else rank
        f.cvw = csum(f.view)
        f.cine = (1 - CINE_VIEW) * f.cine + CINE_VIEW * f.view
        f.cc = csum(f.cine)


def load_look(stab, proxy_dir):
    """(step, thumbnails every `step` frames) from looks.py, or None."""
    if not proxy_dir:
        return None
    p = os.path.join(proxy_dir, os.path.splitext(os.path.basename(stab))[0] + '_look.npz')
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return int(z['step']), z['desc'].astype(np.float64)


class CubeLUT:
    """3D .cube LUT applied to 8-bit RGB through a full 256^3 table (built once, trilinear)."""
    def __init__(self, path):
        size, dmin, dmax, vals = None, np.zeros(3), np.ones(3), []
        for line in open(path, encoding='utf-8', errors='ignore'):
            t = line.strip()
            if not t or t.startswith('#'):
                continue
            k = t.split()
            if k[0] == 'LUT_3D_SIZE':
                size = int(k[1])
            elif k[0] == 'DOMAIN_MIN':
                dmin = np.array(k[1:4], float)
            elif k[0] == 'DOMAIN_MAX':
                dmax = np.array(k[1:4], float)
            elif k[0] == 'LUT_1D_SIZE':
                raise SystemExit(f'{path}: 1D LUTs are not supported')
            elif k[0][0].isdigit() or k[0][0] in '-.':
                vals.append([float(v) for v in k[:3]])
        a = np.array(vals, np.float32).reshape(size, size, size, 3)   # [b][g][r]: red runs fastest
        self.raw, self.dmin, self.dmax = a.copy(), dmin, dmax                # for GPU sampling (final.py)
        for axis, c in ((2, 0), (1, 1), (0, 2)):                       # r, g, b grid axes
            x = (np.arange(256) / 255.0 - dmin[c]) / (dmax[c] - dmin[c]) * (size - 1)
            x = np.clip(x, 0, size - 1)
            i0 = np.minimum(np.floor(x).astype(int), size - 2)
            w = (x - i0).astype(np.float32)
            shp = [1] * 4
            shp[axis] = 256
            w = w.reshape(shp)
            a = np.take(a, i0, axis=axis) * (1 - w) + np.take(a, i0 + 1, axis=axis) * w
        self.table = np.clip(np.round(a * 255), 0, 255).astype(np.uint8)   # [b, g, r] -> rgb
        self.name = os.path.basename(path)

    def __call__(self, rgb):
        return self.table[rgb[..., 2], rgb[..., 1], rgb[..., 0]]


def render_offset(q, fps, flow_mag):
    """Original frame shown as render frame 0, for a render Gyroflow trimmed.

    Visible rotation from the gyro (1.5 Hz) and mean optical flow of the render both rise
    and fall with every turn; the trim offset is where they correlate best."""
    rates = rates_from_quats(q, fps)
    lp = sosfiltfilt(butter(2, VIS_LP_HZ, fs=fps, output='sos'), rates, axis=0)
    vis = np.log1p(np.linalg.norm(lp, axis=1))
    fm = np.log1p(flow_mag)
    fm = (fm - fm.mean()) / (fm.std() + 1e-9)
    n = fm.size
    cors = []
    for off in range(len(vis) - n + 1):
        v = vis[off:off + n]
        cors.append(float(np.mean((v - v.mean()) / (v.std() + 1e-9) * fm)))
    cors = np.array(cors)
    best = int(np.argmax(cors))
    others = np.delete(cors, np.arange(max(best - 25, 0), min(best + 26, cors.size)))
    return best, cors[best], (others.max() if others.size else -1.0)


def proxy_path(stab, proxy_dir):
    if not proxy_dir:
        return None
    p = os.path.join(proxy_dir, os.path.splitext(os.path.basename(stab))[0] + '_480p.mp4')
    return p if os.path.exists(p) else None


def csum(x):
    return np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)])


def prank(x, ref):
    """Percentile rank of x inside the sorted reference sample, 0..1."""
    return np.searchsorted(ref, x) / max(ref.size, 1)


class Flight:
    def __init__(self, idx, src, stab, q, fps, flow, proxy, look=None, motion='gyro'):
        self.idx, self.src, self.stab, self.fps, self.proxy, self.look = idx, src, stab, fps, proxy, look
        self.motion = motion                                     # 'gyro', or 'picture' (vis_motion.py)
        self.name = os.path.splitext(os.path.basename(src))[0]
        self.n = n = len(q)
        self.rates = rates = rates_from_quats(q, fps)
        lp = sosfiltfilt(butter(2, VIS_LP_HZ, fs=fps, output='sos'), rates, axis=0)
        lp[:, 2] *= ROLL_WEIGHT
        self.lp = lp
        self.vis = np.linalg.norm(lp, axis=1)                    # visible angular speed, deg/s
        # attitude: camera elevation above the horizon, upside down
        self.elev = np.degrees(np.arcsin(np.clip(rotate(q, np.array([0.0, 0.0, 1.0])) @ UP, -1, 1)))
        self.inverted = rotate(q, np.array([0.0, 1.0, 0.0])) @ UP > 0
        self.flow = flow
        self.usable = self._usable()
        d = np.maximum(np.gradient(self.vis) * fps, 0)           # onset of rotation, deg/s^2
        pk, pr = find_peaks(d, height=ACC_HEIGHT, distance=int(0.2 * fps), prominence=ACC_PROM)
        self.peak_frames, self.peak_heights = pk, pr['peak_heights']
        # manoeuvres: a lot of rotation within 1 s (kind from the axis that carried it), a dive
        # (camera well below the horizon), a pull-up (camera tilts up fast); close passes are
        # added in set_scales once the parallax ranks are known
        rot1 = uniform_filter1d(self.vis, int(fps))
        tp, _ = find_peaks(rot1, height=TRICK_DEG, distance=int(1.5 * fps))
        ax1 = uniform_filter1d(np.abs(lp), int(fps), axis=0)
        down = uniform_filter1d(-self.elev, int(0.4 * fps))
        dp, _ = find_peaks(down, height=DIVE_DEG, distance=int(1.5 * fps))
        h = int(0.5 * fps)
        e = np.pad(self.elev, h, mode='edge')
        rise = e[2 * h:] - e[:-2 * h]
        up_, _ = find_peaks(rise, height=PULLUP_DEG, distance=int(1.5 * fps))
        self._events = ([(c, rot1[c] / TRICK_DEG, ('loop', 'spin', 'roll')[int(np.argmax(ax1[c]))]) for c in tp]
                        + [(c, down[c] / DIVE_DEG, 'dive') for c in dp]
                        + [(c, rise[c] / PULLUP_DEG, 'pull-up') for c in up_])
        self.delev = uniform_filter1d(np.gradient(self.elev) * fps, int(0.3 * fps))
        self.cu = csum(self.usable.astype(np.float64))
        self.cv = csum(self.vis)
        k = int(0.2 * fps)                                       # rotation just before / after a frame
        cl = csum(lp)
        i = np.arange(n)
        a, b = np.maximum(i - k, 0), np.minimum(i + k, n)
        self.w_back = (cl[i] - cl[a]) / np.maximum(i - a, 1)[:, None]
        self.w_fwd = (cl[b] - cl[i]) / np.maximum(b - i, 1)[:, None]

    def _usable(self):
        fps, n = self.fps, self.n
        u = np.ones(n, bool)
        u[:int(3 * fps)] = False
        u[n - int(5 * fps):] = False
        raw = np.linalg.norm(self.rates, axis=1)
        for c in np.flatnonzero(raw > 2000):                     # crash / impact
            if c > n - 15 * fps:
                u[max(c - int(1.5 * fps), 0):] = False
            else:
                u[max(c - int(1.0 * fps), 0):c + int(1.5 * fps)] = False
        # on the ground: no vibration and no rotation for 2 s; motion from a stabilized picture has no vibration
        # left, so there it is a picture standing still (STILL_FLOW)
        if self.motion == 'picture' and self.flow is not None:
            calm = self.flow['flow'] < STILL_FLOW
        else:
            hf = self.rates - uniform_filter1d(self.rates, int(fps / 2), axis=0)
            vib = np.sqrt(uniform_filter1d((hf ** 2).sum(1), int(2 * fps)))
            calm = (vib < 1.5) & (self.vis < 3.0)
        still = uniform_filter1d(calm.astype(float), int(2 * fps)) > 0.9
        u &= ~still
        # landings: the approach and the touchdown before each stretch on the ground, marked (W_LANDING), not cut
        self.landing = np.zeros(n, bool)
        for a in np.flatnonzero(np.diff(still.astype(int)) == 1) + 1:
            self.landing[max(a - int(LANDING_S * fps), 0):a] = True
        if self.flow is not None:                                # blown-out / black picture
            br = self.flow['bright']
            u &= (br > 25) & (br < 235)
            if 'props' in self.flow:                             # propellers in view, +-0.25 s
                bad = uniform_filter1d((self.flow['props'] > PROP_KEPT).astype(float), int(0.5 * fps)) > 1e-6
                u &= ~bad
        return u

    def raw_skill_parts(self):
        """Unranked per-frame ingredients of the piloting scores."""
        fps = self.fps
        lean = uniform_filter1d(5.0 - self.elev, int(fps))       # nose down: speed or a dive
        jerk = uniform_filter1d(np.abs(np.gradient(self.vis)) * fps, int(0.5 * fps))
        parts = dict(lean=lean, rotation=uniform_filter1d(self.vis, int(0.5 * fps)), jerk=jerk)
        if self.flow is not None:
            parts['parallax'] = uniform_filter1d(self.flow['resid90'], int(0.5 * fps))
        return parts

    def set_scales(self, ref, ranks):
        self.peak_strength = np.log1p(self.peak_heights / ref)
        a = np.zeros(self.n)
        a[self.peak_frames] = self.peak_strength
        self.accent_imp = a
        tri = 1 - np.abs(np.arange(-ACC_REACH, ACC_REACH + 1)) / (ACC_REACH + 1)
        self.accent_wide = np.convolve(a, tri, 'same')
        self.ca = csum(a)
        parts = self.raw_skill_parts()
        rk = {k: prank(v, ranks[k]) for k, v in parts.items() if k in ranks}
        rk['smooth'] = 1.0 - rk['jerk']

        def mix(w):
            return sum(w[k] * rk[k] for k in w if k in rk) / sum(w[k] for k in w if k in rk)
        self.skill = mix(SKILL_PARTS)                            # action
        self.cine = mix(CINE_PARTS)                              # cinematic
        self.cs, self.cc = csum(self.skill), csum(self.cine)
        ev = list(self._events)
        par = rk.get('parallax')
        self.par_rank = par
        if par is not None:                                      # close pass: top parallax, little rotation
            pk, _ = find_peaks(np.where(rk['rotation'] < 0.7, par, 0.0), height=PASS_RANK, distance=int(2 * self.fps))
            ev += [(c, 1.2, 'pass') for c in pk]
            # descent from height: looking well down while everything is far away
            low = np.flatnonzero(uniform_filter1d(self.elev, int(2 * self.fps)) < -DESCENT_DEG)
            for seg in (np.split(low, np.flatnonzero(np.diff(low) > 1) + 1) if low.size else []):
                if seg.size >= DESCENT_S * self.fps and par[seg].mean() < DESCENT_FAR:
                    ev.append((int(seg[seg.size // 2]), 1.2, 'descent'))
        tilt_ok = getattr(self, 'tilt_ok', None)                 # motion from the picture: see vis_motion.TRUST_INL
        ev = sorted(e for e in ev if self.usable[e[0]] and
                    (tilt_ok is None or e[2] not in ('dive', 'pull-up', 'descent') or tilt_ok[e[0]]))
        self.tricks = np.array([e[0] for e in ev], int)
        self.trick_deg = np.array([e[1] * TRICK_DEG for e in ev])
        self.trick_kind = [e[2] for e in ev]
        # movement signature per frame, one channel per entry of MOVES
        lp = self.lp
        sig = np.stack([
            np.clip((-self.elev - 25.0) / 30.0, 0, 1),
            np.clip((self.delev - 40.0) / 60.0, 0, 1),
            np.clip((-self.delev - 40.0) / 60.0, 0, 1),
            np.clip((lp[:, 1] - 40.0) / 80.0, 0, 1),
            np.clip((-lp[:, 1] - 40.0) / 80.0, 0, 1),
            np.clip((np.abs(lp[:, 2]) / ROLL_WEIGHT - 80.0) / 150.0, 0, 1),
            np.clip((par - 0.7) / 0.3, 0, 1) * (1 - rk['rotation']) if par is not None else np.zeros(self.n),
            np.full(self.n, 0.15),
        ], 1)
        self.msig = csum(sig)


def creation_time(path):
    import datetime
    with av.open(path) as c:
        t = c.metadata.get('creation_time') or c.streams.video[0].metadata.get('creation_time')
    try:
        return datetime.datetime.fromisoformat(t.replace('Z', '+00:00'))
    except (AttributeError, ValueError):
        return datetime.datetime.fromtimestamp(os.path.getmtime(path), datetime.timezone.utc)


def stabilized_before(path):
    """A clip montage.py's scan took in as already stabilized: a `_stabilized` export without its original."""
    return '_stabilized' in os.path.basename(path).lower()


def load_motion(src, cache_dir, render=None):
    """Per-frame camera orientation and where it comes from: Gyroflow's export if the cache has one ('gyro');
    for a clip stabilized before (a `_stabilized` export without its original, see montage.py scan) the
    orientation vis_motion.py found in its picture ('picture'); else DJI `djmd` ('gyro')."""
    stem = os.path.splitext(os.path.basename(src))[0]
    gj = os.path.join(cache_dir, stem + '_gyroflow_camera.json')
    if os.path.exists(gj):
        with open(gj, encoding='utf-8') as fh:
            d = json.load(fh)
        q = np.array([r['org_quat'] for r in d], float)
        return q / np.linalg.norm(q, axis=1, keepdims=True), frame_count(src)[1], 'gyro'
    if stabilized_before(src):
        vp = os.path.splitext(render or '')[0] + '_vis.npz'
        if not os.path.exists(vp):
            raise SystemExit(f'stabilized before, no motion from its picture yet: run vis_motion.py on {render}')
        z = np.load(vp)
        return z['q'].astype(np.float64), float(z['fps']), 'picture'
    return (*load_telemetry(src, cache_dir), 'gyro')


def find_flights_list(originals, cache_dir, proxy_dir):
    out = []
    for src in originals:
        stem = os.path.splitext(os.path.basename(src))[0]
        render = os.path.join(proxy_dir, stem + '_stabilized.mp4')
        if not (os.path.exists(render) and load_look(render, proxy_dir) is not None):
            print(f'  skip {stem}: no Gyroflow render + analysis in {proxy_dir}')
            continue
        try:
            q, fps, motion = load_motion(src, cache_dir, render)
        except SystemExit as e:
            print(f'  skip {stem}: {e}')
            continue
        n = frame_count(render)[0]
        off = 0
        if len(q) != n:
            off, c, c2 = render_offset(q, fps, load_flow(render, proxy_dir, n)['flow'])
            print(f'  {stem}: motion {len(q)} frames, render {n}: render starts at {off} (corr {c:.2f} vs {c2:.2f})')
        q = q[off:off + n]
        if len(q) < n:
            q = np.vstack([q, np.repeat(q[-1:], n - len(q), axis=0)])
        fl = Flight(len(out), src, render, q, fps, load_flow(render, proxy_dir, n), render, load_look(render, proxy_dir),
                    motion)
        if motion == 'picture':                                  # tilt manoeuvres only where the rotation fits
            import vis_motion
            z = np.load(os.path.splitext(render)[0] + '_vis.npz')
            tr = vis_motion.trusted(z['inl'], fps)[off:off + n]
            fl.tilt_ok = np.concatenate([tr, np.ones(n - len(tr), bool)]) if len(tr) < n else tr
        pe = load_people(render, proxy_dir, n)
        fl.people = pe if pe is not None else np.zeros(n)
        fl.cp = csum(fl.people)
        fl.cpa = csum(load_pilot_alone(render, proxy_dir, n))
        fl.csb = csum(load_subject(render, proxy_dir, n))
        fl.cimp = csum(load_stand_ins(render, proxy_dir, n))
        fl.can = csum(load_animals(os.path.join(proxy_dir, os.path.splitext(os.path.basename(render))[0]), n) > 0)
        fl.beauty = load_beauty(render, n)
        fl.offset, fl.when = off, creation_time(src)
        fl.day = fl.when.date().isoformat()
        out.append(fl)
    return finish_flights(out)


def find_flights(root, cache_dir, proxy_dir):
    srcs = sorted({os.path.normcase(p): p for p in glob.glob(os.path.join(root, '**', '*.mp4'), recursive=True)}.values())
    out = []
    for src in srcs:
        if '_stabilized' in os.path.basename(src).lower():
            continue
        n, _ = frame_count(src)
        stem = os.path.splitext(src)[0]
        renders = [p for p in srcs if os.path.normcase(p).startswith(os.path.normcase(stem + '_stabilized'))]
        full = [p for p in renders if frame_count(p)[0] == n]
        # a trimmed render (Gyroflow in/out points) is usable once its proxy pass says where it starts
        trimmed = [p for p in renders if p not in full and frame_count(p)[0] > 0.5 * n
                   and load_flow(p, proxy_dir, frame_count(p)[0]) is not None]
        if not full and not trimmed:
            print(f'  skip {os.path.basename(src)}: no Gyroflow render covering it')
            continue
        stab = max(full, key=os.path.getsize) if full else max(trimmed, key=lambda p: frame_count(p)[0])
        try:
            q, fps = load_telemetry(src, cache_dir)
        except SystemExit as e:
            print(f'  skip {os.path.basename(src)}: {e}')
            continue
        off = 0
        if not full:
            nr = frame_count(stab)[0]
            off, c, c2 = render_offset(q, fps, load_flow(stab, proxy_dir, nr)['flow'])
            ok = c > 0.3 and c - c2 > 0.05
            print(f'  {os.path.basename(stab)}: trimmed render, starts at original frame {off} '
                  f'({off / fps:.2f} s), correlation {c:.2f} vs next best {c2:.2f}{"" if ok else " -> skipped"}')
            if not ok:
                continue
            q = q[off:off + nr]
            n = nr
        fl = Flight(len(out), src, stab, q, fps, load_flow(stab, proxy_dir, n), proxy_path(stab, proxy_dir),
                    load_look(stab, proxy_dir))
        fl.offset = off
        out.append(fl)
    return finish_flights(out)


def finish_flights(out):
    """Scales shared by all flights: look-alike sigma, onset strength, piloting ranks. Files that continue
    one recording (the next starts within 3 s of the end of the previous) share a recording id (f.rec) and
    their start in it (f.rec_t0, seconds)."""
    prev = None
    for f in sorted(out, key=lambda f: (getattr(f, 'when', None) is None, getattr(f, 'when', None) or 0)):
        w, pw = getattr(f, 'when', None), getattr(prev, 'when', None)
        if w is not None and pw is not None and abs((w - pw).total_seconds() - prev.n / prev.fps) < 3.0:
            f.rec, f.rec_t0 = prev.rec, prev.rec_t0 + prev.n / prev.fps
        else:
            f.rec, f.rec_t0 = f.idx, 0.0
        prev = f
    if any(f.flow is None for f in out):
        print('  note: no proxy pass for some flights, piloting score without parallax')
    global LOOK_SIGMA
    looks = [f.look[1] for f in out if f.look is not None]
    if looks:
        rng = np.random.default_rng(0)                           # fixed: same footage, same scale
        ij = rng.integers(0, len(looks), (4000, 2))
        d = [np.sqrt(((looks[i][rng.integers(len(looks[i]))] - looks[j][rng.integers(len(looks[j]))]) ** 2).mean())
             for i, j in ij]
        LOOK_SIGMA = LOOK_SIGMA_FRAC * float(np.median(d))
    allh = np.concatenate([f.peak_heights for f in out])
    ref = np.percentile(allh, 75)
    ranks = {}
    for k in ('parallax', 'lean', 'rotation', 'jerk'):
        vals = [f.raw_skill_parts().get(k) for f in out]
        if all(v is not None for v in vals):
            ranks[k] = np.sort(np.concatenate([v[f.usable] for v, f in zip(vals, out)]))
    for f in out:
        if 'parallax' not in ranks and f.flow is not None:
            f.flow = None                                        # keep the score comparable
        f.set_scales(ref, ranks)
    return out


# ===================================================================== music
def beat_grid(env, times, bpm0):
    """Constant-tempo grid (studio tracks): bpm and phase maximising onset strength on beats."""
    def score(bpm, ph):
        g = np.arange(times[0] + ph, times[-1], 60.0 / bpm)
        return np.interp(g, times, env).mean()
    best = (-1, bpm0, 0.0)
    for bpm in np.arange(bpm0 * 0.97, bpm0 * 1.03, 0.05):
        for ph in np.arange(0, 60.0 / bpm, 0.01):
            s = score(bpm, ph)
            if s > best[0]:
                best = (s, bpm, ph)
    _, b0, p0 = best
    for bpm in np.arange(b0 - 0.06, b0 + 0.06, 0.002):
        for ph in np.arange(p0 - 0.012, p0 + 0.012, 0.001):
            s = score(bpm, ph)
            if s > best[0]:
                best = (s, bpm, ph)
    return best[1], best[2]


def analyse_music(path, bpm_hint=None):
    y = mus.load_audio(path)
    sr = mus.SR
    env, low, times, rate = mus.onset_envelope(y)
    bpm0 = bpm_hint or mus.tempo(env, rate)[0]
    bpm, ph = beat_grid(env, times, bpm0)
    P = 60.0 / bpm
    beats = np.arange(times[0] + ph, y.size / sr - P * 0.5, P)
    beats = beats[beats >= 0]
    edges = np.append(beats, beats[-1] + P)
    loud = np.array([20 * np.log10(np.sqrt(np.mean(y[int(a * sr):max(int(b * sr), int(a * sr) + 1)] ** 2)) + 1e-9)
                     for a, b in zip(edges[:-1], edges[1:])])
    on = loud > loud.max() - 30                                  # drop silent beats at the ends
    first, last = np.argmax(on), len(on) - np.argmax(on[::-1])
    beats, loud = beats[first:last], loud[first:last]
    ti = lambda t: np.clip(np.searchsorted(times, t), 0, times.size - 1)
    near = lambda e, t: np.array([e[max(i - 3, 0):i + 4].max() for i in ti(t)])
    acc = near(env, beats)
    kick = near(low, beats)
    tb = ti(np.append(beats, beats[-1] + P))
    busy = np.array([env[a:max(b, a + 1)].mean() for a, b in zip(tb[:-1], tb[1:])])
    dphase = int(np.argmax([kick[p::4].mean() for p in range(4)]))   # downbeat: most kick
    return dict(path=path, duration=y.size / sr, bpm=bpm, period=P, beats=beats, loud=loud,
                acc=acc, kick=kick, busy=busy, downbeat=dphase, env=env, env_t=times)


PATTERNS = {        # beats per shot inside a 16-beat phrase: one-minute people edit, 8-beat shots
    0: [[16]],
    1: [[8, 8]],
    2: [[8, 8]],
    3: [[8, 8]],
}
PATTERNS_LIVELY = {     # --lively: short lively edits (kids at play), never shorter than 4 beats
    0: [[8, 8]],
    1: [[8, 4, 4], [4, 4, 8]],
    2: [[4, 4, 4, 4], [8, 4, 4]],
    3: [[4, 4, 4, 4]],
}


def phrase_layout(m):
    """Phrase starts (beat index), energy class of each, pickup end, quiet threshold."""
    beats, nb, loud = m['beats'], len(m['beats']), m['loud']
    z = lambda x: (x - np.median(x)) / (x.std() + 1e-9)
    beat_energy = z(loud) + 0.5 * z(m['busy'])
    d0 = m['downbeat']
    nbars = (nb - d0) // 4
    bar_loud = np.array([loud[d0 + 4 * b:d0 + 4 * b + 4].mean() for b in range(nbars)])
    jumps = np.abs(np.diff(bar_loud))
    # phrase = 4 bars; its start phase (in bars) is where loudness jumps line up best
    pphase = int(np.argmax([jumps[(np.arange(len(jumps)) + 1) % 4 == p].sum() for p in range(4)]))
    start = d0 + 4 * pphase
    phrases, b = [], start
    while b + 16 <= nb:
        phrases.append(b)
        b += 16
    quiet_db = np.percentile(loud, 90) - 8.0
    pl = np.array([loud[a:a + 16].mean() for a in phrases])
    pe = np.array([beat_energy[a:a + 16].mean() for a in phrases])
    cls = np.zeros(len(phrases), int)
    lp_ = np.flatnonzero(pl >= quiet_db)
    if lp_.size:
        q = np.argsort(np.argsort(pe[lp_])) / max(lp_.size - 1, 1)
        cls[lp_] = 1 + (q > 0.34).astype(int) + (q > 0.67).astype(int)
    return phrases, cls, start, quiet_db, beat_energy


def plan_scenes(m, rng):
    beats, nb, loud = m['beats'], len(m['beats']), m['loud']
    phrases, cls, start, quiet_db, beat_energy = phrase_layout(m)
    d0 = m['downbeat']
    tail = nb - (phrases[-1] + 16 if phrases else start)

    def shot(a, n, c):
        t0 = 0.0 if a == 0 else float(beats[a])
        t1 = float(beats[a + n]) if a + n < nb else m['duration']
        w = m['acc'][a:a + n] / (m['acc'].mean() + 1e-9)
        w = w + 0.5 * ((np.arange(a, a + n) - d0) % 4 == 0)       # downbeats matter more
        return dict(b0=a, nbeats=n, cls=int(c), t0=t0, t1=t1, speed=BASE_SPEED[int(c)],
                    energy=float(beat_energy[a:a + n].mean()), beat_w=w)

    scenes = []
    if start > 0:                                                # pickup before the first phrase
        c0 = 0 if loud[:start].mean() < quiet_db else 1
        scenes.append(dict(kind='pickup', cls=c0, shots=[shot(0, start, c0)]))
    for i, (a, c) in enumerate(zip(phrases, cls)):
        opts = PATTERNS[c]
        pat = opts[rng.integers(len(opts))] if rng is not None else opts[i % len(opts)]
        sh = []
        for L in pat:
            sh.append(shot(a, L, c))
            a += L
        scenes.append(dict(kind='phrase', cls=int(c), shots=sh))
    if tail > 0:
        sh, a = [], nb - tail
        while a < nb:
            L = min(nb - a, 8)
            sh.append(shot(a, L, cls[-1] if len(cls) else 1))
            a += L
        scenes.append(dict(kind='tail', cls=int(cls[-1]) if len(cls) else 1, shots=sh))
    if END_SET[0] and len(scenes[-1]['shots']) > 1:              # the editor's ending is a scene of its own
        last = scenes[-1]['shots'].pop()
        scenes.append(dict(kind='end', cls=scenes[-1]['cls'], shots=[last]))
    # hero: at most MAX_HEROES, where the music lifts the most (coming back after a breakdown first)
    lifts = []
    for i, s in enumerate(scenes):
        s['idx'], s['hero'], s['slow_hero'] = i, False, False
        prev = scenes[i - 1]['cls'] if i else 0
        if s['kind'] == 'phrase' and s['cls'] >= 2 and s['shots'][0]['nbeats'] >= 4 and i > 0:
            lift = s['cls'] - prev + (2 if prev == 0 else 0)
            if lift > 0:
                lifts.append((lift, s['cls'], -i))
    for _, _, ni in sorted(lifts, reverse=True)[:MAX_HEROES]:
        scenes[-ni]['hero'] = True
    # flying scenes: the biggest lifts, else the loudest phrases; not the opening, not side by side
    for s in scenes:
        s['fly'] = False
    cand = sorted(lifts, reverse=True) + sorted(((0, s['cls'], -i) for i, s in enumerate(scenes)
                                                 if s['kind'] == 'phrase' and i > 0), reverse=True)
    for _, _, ni in cand:
        if sum(s['fly'] for s in scenes) >= MAX_FLY:
            break
        if END_SET[0] and -ni == len(scenes) - 1:
            continue
        if not any(scenes[j]['fly'] for j in (-ni - 1, -ni, -ni + 1) if 0 <= j < len(scenes)):
            scenes[-ni]['fly'] = True
    for sc in scenes:                                            # a flight to something is shown whole: one shot
        if sc['fly'] and len(sc['shots']) > 1:
            x0 = sc['shots'][0]
            sc['shots'] = [shot(x0['b0'], sum(x['nbeats'] for x in sc['shots']), sc['cls'])]
    allshots = [x for s in scenes for x in s['shots']]
    e = np.array([x['energy'] + 2.0 * x['cls'] for x in allshots])
    q = np.argsort(np.argsort(e)) / max(len(e) - 1, 1)
    for x, qq in zip(allshots, q):
        x['q_music'] = float(qq)
    return scenes


# ================================================================== matching
class Pools:
    """Distribution of a per-frame feature averaged over every usable window of a length."""
    def __init__(self, flights):
        self.flights, self.cache = flights, {}

    def get(self, n, sp, what):
        key = (n, sp, what)
        if key not in self.cache:
            parts = []
            for f in self.flights:
                if n >= f.n:
                    continue
                c = {'vis': f.cv, 'skill': f.cs, 'cine': f.cc}[what]
                s = np.arange(f.n - n)
                ok = (f.cu[s + n] - f.cu[s]) == n
                v = (c[s + n] - c[s])[ok] / n
                parts.append(sp * v if what == 'vis' else v)
            self.cache[key] = np.sort(np.concatenate(parts))
        return self.cache[key]


def hero_offset(sh, f):
    """Source frames from the shot start to the manoeuvre in the hero ramp."""
    bf = 60.0 / sh['bpm'] * f.fps
    return (HERO_RUSH + HERO_SLOW * (HERO_AT - 1.0)) * bf


def look_penalty(f, seen):
    """Per thumbnail of f: how much it looks like a shot of an earlier-placed scene (0..1)."""
    if f.look is None or not seen:
        return None
    step, d = f.look
    S = np.array(seen)
    dist = np.sqrt(((d[:, None, :] - S[None, :, :]) ** 2).mean(-1))
    return step, np.exp(-(dist / LOOK_SIGMA) ** 2).max(1)


def add_ascents(flights, cache):
    """Ascents from the picture (see ASC_*), added to each flight's manoeuvres, with the whole way up and down
    (f.arcs: first frame, top reached, back near the ground, seconds up there, how far everything is up there).
    The cinematic piloting score then counts height (CINE_HEIGHT): a climb from height scored ~0.1 without it.
    Only when an edit asks for climbs (--want ascent, --arcs); flights without CLIP embeddings (places.py
    embed) or image motion get none."""
    import places
    mdl, temb = None, None
    for f in flights:
        if getattr(f, 'par_rank', None) is None or not os.path.exists(os.path.splitext(f.stab)[0] + '_clip.npz'):
            continue
        if mdl is None:
            mdl = places.model_of(cache)
            temb = places.text_emb(mdl, ASC_HIGH_TEXTS + ASC_LOW_TEXTS)
        fr, _, emb = places.embed(f.stab, mdl)
        fr = np.asarray(fr)
        ps = uniform_filter1d(places.probs(emb.astype(np.float32), temb, len(ASC_HIGH_TEXTS)).sum(1), 3)
        step = max(int(fr[1] - fr[0]), 1) if fr.size > 1 else int(f.fps / 2)
        within, keep = int(ASC_WITHIN_S * f.fps / step), int(2.5 * f.fps / step)
        found, arcs, i = [], [], 0
        while i < ps.size:
            j = next((k for k in range(i + 1, min(i + within + 1, ps.size)) if ps[k] > ASC_HIGH), None) \
                if ps[i] < ASC_LOW else None
            if j is None:
                i += 1
                continue
            a = max(k for k in range(i, j) if ps[k] < ASC_LOW)
            lo, hi = int(fr[j]), min(int(fr[min(j + keep, ps.size - 1)]), f.n - 1)
            d = j                                                    # back near the ground
            while d < ps.size and ps[d] > ASC_LOW:
                d += 1
            if ps[j:j + keep].mean() >= ASC_KEEP and f.par_rank[lo:hi + 1].mean() <= ASC_FAR:
                c = (int(fr[a]) + int(fr[j])) // 2                   # the way up, the place opens at its end
                if f.usable[c]:
                    found.append(c)
                    end = int(fr[min(d, ps.size - 1)])
                    arcs.append((int(fr[a]), int(fr[j]), end, (end - fr[j]) / f.fps,
                                 float(f.par_rank[int(fr[j]):end + 1].mean())))
            i = d
        f.arcs = arcs
        f.high = np.interp(np.arange(f.n), fr, ps)
        if found:
            ev = sorted(list(zip(f.tricks, f.trick_deg, f.trick_kind)) +
                        [(c, 1.2 * TRICK_DEG, 'ascent') for c in found])
            f.tricks = np.array([e[0] for e in ev], int)
            f.trick_deg = np.array([e[1] for e in ev])
            f.trick_kind = [e[2] for e in ev]
            print(f'  F{f.idx} {f.name}: up and down ' + ', '.join(
                f'{a / f.fps:.1f}-{e / f.fps:.1f}s (up {u:.0f}s, far {p:.2f})' for a, _, e, u, p in arcs))
    seen_high = [f for f in flights if getattr(f, 'high', None) is not None]
    if seen_high:                                                # height in the cinematic score
        pool = np.sort(np.concatenate([f.high[f.usable] for f in seen_high]))
        for f in seen_high:
            f.cine = (1 - CINE_HEIGHT) * f.cine + CINE_HEIGHT * prank(f.high, pool)
            f.cc = csum(f.cine)


def pick_arcs(flights, m, n, taken):
    """--arcs N: the highest climbs-and-descents (up there long, everything far) as editor's pieces, from
    different flights first. One longer than a phrase at PIECE_MAX_SPEED keeps its end: the top and the way
    down. Pieces that overlap the editor's own (taken) are skipped."""
    longest = 16 * 60.0 / m['bpm'] * PIECE_MAX_SPEED - PIECE_AIR_S
    cand = sorted(((u * (1 - p), f, a, e) for f in flights for a, _, e, u, p in getattr(f, 'arcs', [])),
                  key=lambda c: -c[0])
    out, used = [], set()
    for unique in (True, False):
        for _, f, a, e in cand:
            if len(out) >= n or (unique and f.idx in used):
                continue
            hi = min(e + int(0.5 * f.fps), f.n - 1)
            lo = max(a - int(0.5 * f.fps), hi - int(longest * f.fps), 0)
            if any(fi == f.idx and lo < h and hi > l for fi, l, h, *_ in taken + out):
                continue
            out.append((f.idx, lo, hi, None, False))
            used.add(f.idx)
            f.usable[lo:hi] = True
            f.cu = csum(f.usable.astype(np.float64))
    if out:
        print('  arcs: ' + ', '.join(f'{flights[fi].name} {lo / flights[fi].fps:.1f}-{hi / flights[fi].fps:.1f}s'
                                     for fi, lo, hi, *_ in out))
    return out


def load_themes(flights, cache):
    """Per flight: frames of the CLIP samples (f.theme_fr) and the probability of each --theme there (f.theme_p)."""
    import places
    mdl = places.model_of(cache)
    texts = [t for t, _ in THEMES]
    temb = places.text_emb(mdl, texts + [t for t in places.GENERIC_NOT if t not in texts])
    for f in flights:
        fr, _, emb = places.embed(f.stab, mdl)
        f.theme_fr = np.asarray(fr)
        f.theme_p = places.probs(emb.astype(np.float32), temb, len(texts))


def theme_penalty(f, shown):
    """(sample frames, cumulative per-sample penalty) for the themes that have their scenes already, or None."""
    if getattr(f, 'theme_p', None) is None:
        return None
    full = np.array([shown[k] >= cap for k, (_, cap) in enumerate(THEMES)], float)
    if not full.any():
        return None
    pen = (np.clip((f.theme_p - THEME_LOW) / (THEME_FULL - THEME_LOW), 0, 1) * full).sum(1)
    return f.theme_fr, np.concatenate([[0.0], np.cumsum(pen)])


def window_theme(f, a, n):
    """Mean probability of each theme over the frames [a, a + n) of f."""
    i, j = np.searchsorted(f.theme_fr, [a, a + n])
    i = min(i, len(f.theme_fr) - 1)
    return f.theme_p[i:max(j, i + 1)].mean(0)


def shot_look(f, a, n):
    if f.look is None:
        return None
    step, d = f.look
    return d[min((a + n // 2) // step, len(d) - 1)]


def window_moves(f, s, n):
    """Unit movement signature of the windows [s, s+n) of flight f, (len(s), len(MOVES))."""
    v = (f.msig[s + n] - f.msig[s]) / n
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


def shot_quality(sh, f, s, n, pools):
    """Piloting percentile of the windows: cinematic for quiet music, action for loud, mixed by energy."""
    e = sh['q_music']
    qa = prank((f.cs[s + n] - f.cs[s]) / n, pools.get(n, sh['speed'], 'skill'))
    qc = prank((f.cc[s + n] - f.cc[s]) / n, pools.get(n, sh['speed'], 'cine'))
    return e * qa + (1 - e) * qc


def shot_nodes(sh, f, n, pools, used, hero, slow_hero, rng, look=None, moves=(), hero_used=(), opening=False,
               fly=False, must=None, theme=None, ending=False):
    """Score of starting this shot at every frame of flight f (-inf where impossible)."""
    N = f.n
    out = np.full(N, -np.inf)
    if n >= N:
        return out
    s = np.arange(N - n)
    sp = sh['speed']
    ok = (f.cu[s + n] - f.cu[s]) == n
    mg = int(1.0 * f.fps)
    near = np.zeros(s.size)
    for a, b in used:
        ok &= (s + n <= a - mg) | (s >= b + mg)
        gap = np.maximum(np.maximum(a - (s + n), s - b), 0) / f.fps
        near = np.maximum(near, np.exp(-gap / NEAR_TAU_S))
    qw = prank(sp * (f.cv[s + n] - f.cv[s]) / n, pools.get(n, sp, 'vis'))
    qs = shot_quality(sh, f, s, n, pools)
    offs = np.round((np.arange(sh['nbeats']) * (60.0 / sh['bpm'])) * sp * f.fps).astype(int)
    sync = np.zeros(s.size)
    on_beat = np.zeros(s.size)
    R = ACC_REACH
    for o, w in zip(offs, sh['beat_w']):
        sync += w * f.accent_wide[np.minimum(s + o, N - 1)]
        on_beat += f.ca[np.clip(s + o + R + 1, 0, N)] - f.ca[np.clip(s + o - R, 0, N)]
    sync /= sh['nbeats']
    off = np.maximum(f.ca[s + n] - f.ca[s] - on_beat, 0) / sh['nbeats']
    sc = (W_SKILL * qs - W_INT * np.abs(qw - sh['q_music']) + W_SYNC * sync - W_OFF * off - W_NEAR * near)
    vw = (f.cvw[s + n] - f.cvw[s]) / n if W_BEAUTY and hasattr(f, 'cvw') else np.zeros(s.size)
    sc += W_BEAUTY * vw
    if fly:
        sc += W_FLY * qs
    elif W_PEOPLE and hasattr(f, 'cp'):
        pv = (f.cp[s + n] - f.cp[s]) / n
        # flights to nowhere belong to the flying scenes, unless the view itself is the point (--beauty)
        sc += W_PEOPLE * pv - W_EMPTY * (pv < EMPTY_BELOW) * (1 - vw)
        if FOCUS[0]:
            sc -= W_AWAY * ((f.csb[s + n] - f.csb[s]) / n < FOCUS_PRESENT)
            sc -= W_IMPOSTOR * (f.cimp[s + n] - f.cimp[s]) / n
        if PILOT_USED[0] >= PILOT_BUDGET * PILOT_USED[1]:
            sc -= W_PILOT_OVER * (f.cpa[s + n] - f.cpa[s]) / n
    if opening:
        sc += W_OPEN * qs
    if not ending and getattr(f, 'landing', None) is not None:  # a landing is for the ending only
        cl = f.cland if hasattr(f, 'cland') else csum(f.landing.astype(np.float64))
        f.cland = cl
        sc -= W_LANDING * (cl[s + n] - cl[s]) / n
    if moves:
        sig = window_moves(f, s, n)
        pen = np.zeros(s.size)
        for v, t in moves:
            cos = sig @ v
            pen += np.maximum(cos - MOVE_SIM0, 0) / (1 - MOVE_SIM0) * np.exp(-abs(t - sh['t0']) / MOVE_TAU_S)
        sc -= W_MOVE * pen
    h = np.zeros(N)
    if hero:
        # the manoeuvre has to sit where the hero ramp puts it (+-0.3 beat)
        c0 = hero_offset(sh, f)
        tol = 0.3 * 60.0 / sh['bpm'] * f.fps
        for c, dg, kind in zip(f.tricks, f.trick_deg, f.trick_kind):
            a, b = int(max(c - c0 - tol, 0)), int(max(c - c0 + tol, 0))
            w = HERO_REUSE if kind in hero_used else 1.0
            h[a:b + 1] = np.maximum(h[a:b + 1], w * dg / TRICK_DEG)
    elif slow_hero or fly:
        # a manoeuvre in the middle half of the shot (flying scenes: of a kind not shown yet, if possible)
        for c, dg, kind in zip(f.tricks, f.trick_deg, f.trick_kind):
            a, b = int(max(c - 0.75 * n, 0)), int(max(c - 0.25 * n, 0))
            w = HERO_REUSE if fly and kind in hero_used else 1.0
            h[a:b + 1] = np.maximum(h[a:b + 1], w * dg / TRICK_DEG)
    sc += W_HERO * h[s]
    if WANT:
        wv = np.zeros(N)
        for c, kind in zip(f.tricks, f.trick_kind):
            if kind in WANT:
                wv[int(max(c - 0.75 * n, 0)):int(max(c - 0.25 * n, 0)) + 1] = 1.0
        sc += W_WANT * wv[s]
    if look is not None:
        step, pen = look
        sc -= W_LOOK * pen[np.minimum((s + n // 2) // step, pen.size - 1)]
    if theme is not None:
        fr, cs = theme
        i, j = np.searchsorted(fr, s), np.searchsorted(fr, s + n)
        sc -= W_THEME * (cs[j] - cs[i]) / np.maximum(j - i, 1)
    if must is not None:                                         # the editor's moment: (first, last frame, end at)
        lo, hi, end_at = must
        if end_at is not None:                                   # the ending: the shot stops at end_at
            sc += W_MUST * ((s <= lo) & (np.abs(s + n - end_at) <= 0.3 * f.fps))
        else:
            sc += W_MUST * ((s <= lo) & (s + n >= hi))
    if rng is not None:
        sc = sc + NOISE * rng.gumbel(size=sc.size)
    sc[~ok] = -np.inf
    out[s] = sc
    return out


def rec_used(f, used, flights):
    """Footage taken so far, in f's frames: f's own windows and those of the other files of its recording
    (outside f's frame range, so they count only as close in time)."""
    out = list(used[f.idx])
    for g in flights:
        if g.idx != f.idx and g.rec == f.rec:
            out += [(int((g.rec_t0 + a / g.fps - f.rec_t0) * f.fps), int((g.rec_t0 + b / g.fps - f.rec_t0) * f.fps))
                    for a, b in used[g.idx]]
    return out


def merged(shots):
    """The shots of a scene as one shot: an editor's piece longer than any of them."""
    x = dict(shots[0])
    x.update(nbeats=sum(s['nbeats'] for s in shots), t1=shots[-1]['t1'],
             energy=float(np.mean([s['energy'] for s in shots])),
             beat_w=np.concatenate([s['beat_w'] for s in shots]))
    return x


def continuity(a, b):
    """1 = the camera keeps turning the same way across the cut, 0 = opposite / unrelated."""
    return 1.0 - np.linalg.norm(a - b, axis=-1) / (np.linalg.norm(a, axis=-1) + np.linalg.norm(b, axis=-1) + 20.0)


def source_frames(x, f):
    """Source frames a shot consumes at its base speed (hero shots: the hero ramp)."""
    L, P = x['nbeats'], 60.0 / x['bpm']
    if x.get('hero_ramp'):
        bf = P * f.fps
        return int(round((HERO_RUSH + HERO_SLOW * 2 + x['speed'] * (L - 3)) * bf))
    return int(round((x['t1'] - x['t0']) * x['speed'] * f.fps))


def scene_dp(scene, f, pools, used, rng, look=None, moves=(), hero_used=(), theme=None):
    """Best start frames of all shots of a scene inside flight f, shots in forward order."""
    shots = scene['shots']
    ns = [source_frames(x, f) for x in shots]
    mu = scene.get('must')                                       # (flight, first frame, last frame, end at or None)
    nodes = [shot_nodes(x, f, n, pools, used, scene['hero'] and k == 0, scene['slow_hero'], rng, look,
                        moves, hero_used, scene['idx'] <= 1, scene.get('fly', False),
                        mu[1:] if mu and mu[0] == f.idx and (mu[3] is None or k == len(shots) - 1) else None,
                        theme, scene.get('last', False))
             for k, (x, n) in enumerate(zip(shots, ns))]
    N = f.n
    idx = np.arange(N)
    gaps = np.arange(int(GAP_MIN_S * f.fps), int(GAP_MAX_S * f.fps) + 1, max(int(GAP_STEP_S * f.fps), 1))
    dp, backs = nodes[0], []
    for k in range(1, len(shots)):
        pn, psp, sp = ns[k - 1], shots[k - 1]['speed'], shots[k]['speed']
        best = np.full(N, -np.inf)
        arg = np.zeros(N, int)
        for g in gaps:
            ps = idx - g - pn                                    # start of the previous shot
            okp = ps >= 0
            val = np.full(N, -np.inf)
            e = idx[okp] - g                                     # its end frame
            val[okp] = dp[ps[okp]] + W_CONT * continuity(psp * f.w_back[e], sp * f.w_fwd[idx[okp]])
            bt = val > best
            best[bt] = val[bt]
            arg[bt] = g
        dp = best + nodes[k]
        backs.append(arg)
    j = int(np.argmax(dp))
    if not np.isfinite(dp[j]):
        return None
    starts = [j]
    for k in range(len(shots) - 1, 0, -1):
        starts.append(starts[-1] - backs[k - 1][starts[-1]] - ns[k - 1])
    return float(dp[j]), starts[::-1], ns


def chapters(scenes, flights, good=0.75):
    """Recording days in date order; each gets consecutive scenes in proportion to its good flying
    (seconds of footage whose piloting score, cinematic or action, is above `good`) and, with --beauty, its
    beautiful views (f.view above `good`): a day of calm flights over the most beautiful place (Dagestan, an old
    stone village on a mountain: 94-98 % scenic) had 71 good seconds of flying against 143 of a beach day."""
    days = sorted({f.day for f in flights})
    w = []
    for d in days:
        fs = [f for f in flights if f.day == d]
        w.append(sum(float(((np.maximum(np.maximum(f.skill, f.cine), getattr(f, 'view', 0.0) if W_BEAUTY else 0.0)
                             > good) & f.usable).sum()) / f.fps for f in fs))
    w = np.array(w)
    wall = dict(zip(days, w))
    keep = w >= 15.0                                             # a day needs ~15 s of good flying
    if not keep.any():                                           # little footage: the best day it is
        keep = w == w.max()
    days = [d for d, k in zip(days, keep) if k]
    w = w[keep]
    n = len(scenes)
    share = w / w.sum() * n
    cnt = np.maximum(np.floor(share).astype(int), 1)
    while cnt.sum() > n:
        cnt[np.argmax(cnt)] -= 1
    while cnt.sum() < n:
        cnt[np.argmax(share - cnt)] += 1
    if CHAPTERS:                                                 # the editor's own split (--chapters)
        every = sorted({f.day for f in flights})
        if len(CHAPTERS) != len(every) or sum(CHAPTERS) != n:
            raise SystemExit(f'--chapters {",".join(map(str, CHAPTERS))}: {len(every)} days ({", ".join(every)}) '
                             f'and {n} scenes to share')
        days, cnt = [d for d, c in zip(every, CHAPTERS) if c], np.array([c for c in CHAPTERS if c])
        w = np.array([wall[d] for d in days])
    out, k = [], 0
    for d, c in zip(days, cnt):
        for _ in range(c):
            scenes[k]['day'] = d
            k += 1
    return dict(zip(days, w.round(0).tolist())), dict(zip(days, cnt.tolist()))


def shows(r, lo, hi, end_at, fps):
    """Does a scene_dp result show frames lo..hi in one shot (the ending: its last shot, stopping at end_at)?"""
    if end_at is not None:
        return r[1][-1] <= lo and abs(r[1][-1] + r[2][-1] - end_at) <= 0.3 * fps
    return any(a <= lo and a + n >= hi for a, n in zip(r[1], r[2]))


def place_moments(scenes, flights, musts, pools, used, rng):
    """Editor's moments: each goes to the free scene where it scores best without the bonus (the last scene
    for --end, the earliest that can show it for --start); the scene is then planned in that clip only.
    A piece longer than the scene's shots gets the whole phrase as one shot, played faster (up to
    PIECE_MAX_SPEED) when even the phrase is shorter than the piece."""
    for fi, lo, hi, end_at, first in sorted(musts, key=lambda x: (x[3] is None, not x[4])):
        f, last = flights[fi], end_at is not None
        free = [len(scenes) - 1] if last else [i for i, sc in enumerate(scenes)     # a flying scene may host a flight
                                               if not (sc['hero'] or sc.get('must'))]
        # chapters by day: a moment goes to a scene of its own day (assign looks for footage of the scene's day
        # and the days next to it only: a 9 Sep moment placed in a 5 Sep scene left that scene with nothing)
        own = [i for i in free if scenes[i].get('day') in (None, getattr(f, 'day', None))]
        if own and not first:
            free = own
        best = None
        for i in free:
            for whole in (False, True):
                if whole and best is not None:
                    continue
                shots = [merged(scenes[i]['shots'])] if whole else scenes[i]['shots']
                if whole:
                    x = shots[0]
                    need, have = hi - lo + int(PIECE_AIR_S * f.fps), source_frames(x, f)
                    if len(scenes[i]['shots']) == 1 and need <= have:
                        continue                                 # the same single shot again
                    if need > have:
                        x['speed'] = min(x['speed'] * need / have, max(PIECE_MAX_SPEED, x['speed']))
                sc = dict(scenes[i], shots=shots) if whole else scenes[i]
                sc['must'] = (fi, lo, hi, end_at)
                r = scene_dp(sc, f, pools, rec_used(f, used, flights), rng)
                del sc['must']
                if r is not None and shows(r, lo, hi, end_at, f.fps) and (best is None or r[0] > best[0]):
                    best = (r[0], i, shots[0] if whole else None)
            if first and best is not None:
                break
        what = f'moment {f.name} {lo / f.fps:.1f}-{hi / f.fps:.1f}s'
        if best is None:
            print(f'  {what}: no scene can show it' +
                  (f' (longer than a phrase at {PIECE_MAX_SPEED}x)' if hi - lo > 8 * f.fps else ''))
            continue
        if best[2] is not None:
            scenes[best[1]]['shots'] = [best[2]]
        scenes[best[1]]['must'] = (fi, lo, hi, end_at)
        if scenes[best[1]].get('day') and getattr(f, 'day', None):
            scenes[best[1]]['day'] = f.day                       # an opening or an ending of another day
        sp = f' at {best[2]["speed"]:.2f}x' if best[2] is not None and best[2]['speed'] > BASE_SPEED[0] + 0.01 else ''
        print(f'  {what} -> scene {best[1]}{" (the end)" if last else " (the opening)" if first else ""}'
              f'{", the phrase as one shot" + sp if best[2] is not None else ""}')


def avoided(paths, flights):
    """Footage windows (flight idx -> [(first, last frame)]) that earlier edits of the series used."""
    by_name = {f.name.lower(): f for f in flights}
    out = {f.idx: [] for f in flights}
    for p in paths:
        with open(p, encoding='utf-8') as fh:
            e = json.load(fh)
        names = {g['idx']: os.path.splitext(os.path.basename(g['original']))[0].lower() for g in e['flights']}
        for sc in e['scenes']:
            for x in sc['shots']:
                f = by_name.get(names[x['flight']])
                if f is not None:
                    out[f.idx].append((int(x['src_frame']), int(x['src_frame'] + x['src_frames'])))
    return out


def assign(scenes, flights, m, rng, musts=(), avoid=None):
    pools = Pools(flights)
    used = {f.idx: list((avoid or {}).get(f.idx, [])) for f in flights}
    PILOT_USED[:] = [0.0, m['duration']]
    for sc in scenes:
        for k, x in enumerate(sc['shots']):
            x['bpm'] = m['bpm']
            x['hero_ramp'] = bool(sc['hero'] and k == 0)
    place_moments(scenes, flights, musts, pools, used, rng)
    # the editor's moments first (a hero took the footage next to a piece and the piece went missing: village
    # drive), then heroes, flying scenes, the opening, then the rest by energy
    order = sorted(range(len(scenes)), key=lambda i: (not scenes[i].get('must'), not scenes[i]['hero'],
                                                     not scenes[i]['fly'], i > 1, -scenes[i]['cls'],
                                                     -np.mean([x['energy'] for x in scenes[i]['shots']]), i))
    seen = []                                                    # thumbnails of placed shots
    moves = []                                                   # (movement signature, music time) of placed shots
    hero_used = set()
    shown = [0] * len(THEMES)                                    # scenes that show each --theme
    for i in order:
        sc = scenes[i]
        nbr = [scenes[j].get('flight') for j in (i - 1, i + 1) if 0 <= j < len(scenes)]
        best = None
        # the scene's own recording day first; if nothing fits there, the next day, then the previous one
        days = sorted({getattr(f, 'day', None) for f in flights} - {None})
        tries = [sc.get('day')]
        if sc.get('day') in days:
            d = days.index(sc['day'])
            tries += [days[j] for j in (d + 1, d - 1, d + 2, d - 2) if 0 <= j < len(days)]
        for day in tries:
            for f in flights:
                if day and getattr(f, 'day', None) != day:
                    continue
                if sc.get('must') and sc['must'][0] != f.idx:
                    continue
                r = scene_dp(sc, f, pools, rec_used(f, used, flights), rng, look_penalty(f, seen), moves, hero_used,
                             theme_penalty(f, shown))
                if r is None:
                    continue
                score = r[0] - W_SAME_SCENE * sum(1 for x in nbr if x is not None and flights[x].rec == f.rec)
                if best is None or score > best[0]:
                    best = (score, f, r[1], r[2])
            if best is not None:
                if day != sc.get('day'):
                    print(f'  scene {i}: nothing fits on {sc.get("day")}, took {day}')
                    sc['day'] = day
                break
        if best is None:
            usable = sum(float(f.usable.sum()) / f.fps for f in flights)
            raise SystemExit(f'ran out of footage at scene {i}: {usable:.0f} s usable for a {m["duration"]:.0f} s edit '
                             '(shots keep ~1 s apart and away from each other): a shorter --target, or more footage')
        score, f, starts, ns = best
        sc['score'] = float(score)
        take(sc, f, starts, ns, pools, used, seen, moves, hero_used, shown)
    return scenes


def take(sc, f, starts, ns, pools, used, seen, moves, hero_used, shown):
    """Scene sc becomes these windows of flight f: its shots get their footage and what is in it, and what the
    next scenes are scored against (used footage, looks, movements, manoeuvres, themes shown) is updated."""
    sc['flight'] = f.idx
    for k, (x, a, n) in enumerate(zip(sc['shots'], starts, ns)):
        used[f.idx].append((a, a + n))
        sv = window_moves(f, np.array([a]), n)[0]
        if hasattr(f, 'cp'):
            x['people'] = float((f.cp[a + n] - f.cp[a]) / n)
            x['pilot_alone'] = float((f.cpa[a + n] - f.cpa[a]) / n)
            x['subject'] = float((f.csb[a + n] - f.csb[a]) / n)
            x['stand_in'] = float((f.cimp[a + n] - f.cimp[a]) / n)
            x['animal'] = float((f.can[a + n] - f.can[a]) / n)
            PILOT_USED[0] += x['pilot_alone'] * (x['t1'] - x['t0'])
        x.update(flight=f.idx, src_frame=int(a), src_frames=int(n),
                 skill=float(shot_quality(x, f, np.array([a]), n, pools)[0]),
                 move=MOVES[int(np.argmax(sv[:-1])) if sv[:-1].max() > 0.35 else len(MOVES) - 1])
        moves.append((sv, x['t0']))
        if sc.get('fly'):
            kinds = [kind for c, kind in zip(f.tricks, f.trick_kind) if a <= c < a + n]
            if kinds:
                x['fly_kind'] = kinds[0]
                hero_used.update(kinds)
        if sc['hero'] and k == 0:
            want = a + hero_offset(x, f)
            near_ = [(abs(c - want), kind) for c, kind in zip(f.tricks, f.trick_kind)]
            if near_:
                x['hero_kind'] = min(near_)[1]
                hero_used.add(x['hero_kind'])
        lk = shot_look(f, a, n)
        if lk is not None:
            seen.append(lk)
    if getattr(f, 'theme_p', None) is not None:
        hit = np.zeros(len(THEMES), bool)
        for x in sc['shots']:
            if x['t1'] - x['t0'] < THEME_MIN_S:                  # a glimpse (a pickup) does not show it
                continue
            p = window_theme(f, x['src_frame'], x['src_frames'])
            if p.max() >= THEME_HIT:
                x['theme'] = THEMES[int(np.argmax(p))][0]
            hit |= p >= THEME_HIT
        for k in np.flatnonzero(hit):
            shown[k] += 1


# ================================================================ speed ramps
def snap(sh, f, lo, hi):
    """Bend playback speed inside a shot so motion onsets land on beats.

    Anchors at every beat; anchor 0 is the cut (fixed), the others pick either
    their nominal frame or a motion onset in reach, first-order DP over beats.
    Returns source frame of every anchor (len nbeats + 1), or None when the
    shot is not a whole number of beats (pickup, last shot) or cannot bend."""
    L, sp, fps = sh['nbeats'], sh['speed'], f.fps
    bf = 60.0 / sh['bpm'] * fps                                  # source frames per beat at 1x
    vmin, vmax = max(ABS_LO, RAMP_LO * sp), min(ABS_HI, RAMP_HI * sp)
    s = sh['src_frame']
    nominal = s + sp * bf * np.arange(L + 1)
    if sh['t0'] == 0.0 or vmax <= vmin or abs(sh['t1'] - sh['t0'] - L * 60.0 / sh['bpm']) > 1e-3:
        return None
    pk = f.peak_frames
    cands = [[(float(s), 0.0)]]
    for j in range(1, L + 1):
        c = [(nominal[j], 0.0)]
        if j < L:
            r0, r1 = s + vmin * bf * j, s + vmax * bf * j
            for p, h in zip(pk, f.peak_strength):
                if r0 <= p <= r1:
                    c.append((float(p), W_SNAP * h * sh['beat_w'][j]))
        else:
            c += [(s + v * bf * L, 0.0) for v in np.linspace(vmin, vmax, 9)]
        cands.append(c)
    score = [np.array([0.0])]
    back = []
    for j in range(1, L + 1):
        prev = np.array([x for x, _ in cands[j - 1]])
        cur = np.array([x for x, _ in cands[j]])
        bon = np.array([b for _, b in cands[j]])
        v = (cur[None, :] - prev[:, None]) / bf                   # speed of the segment
        ok = (v >= vmin - 1e-9) & (v <= vmax + 1e-9)
        cost = W_DEV * np.log(np.clip(v, 1e-3, None) / sp) ** 2
        tot = np.where(ok, score[-1][:, None] - cost, -np.inf)
        bi = np.argmax(tot, axis=0)
        score.append(tot[bi, np.arange(cur.size)] + bon)
        back.append(bi)
    j = int(np.argmax(score[-1]))
    if not np.isfinite(score[-1][j]):
        return None
    path = [j]
    for k in range(L - 1, -1, -1):
        path.append(back[k][path[-1]])
    path = path[::-1]
    x = np.array([cands[k][path[k]][0] for k in range(L + 1)])
    a, b = int(np.floor(x[0])), int(np.ceil(x[-1]))
    if a < lo or b > hi or b > f.n - 1 or (f.cu[b] - f.cu[a]) < (b - a):
        return None
    return x


def hero_ramp(sh, f):
    """Rush in (beat 0), the manoeuvre in slow motion (beats 1-2, on beat 2), then base speed.

    The manoeuvre is the strongest one near where shot_nodes asked for it."""
    bf = 60.0 / sh['bpm'] * f.fps
    s, L, sp = sh['src_frame'], sh['nbeats'], sh['speed']
    want = s + hero_offset(sh, f)
    near = [(abs(c - want), c) for c in f.tricks if abs(c - want) <= 0.4 * bf]   # nodes ask for 0.3
    if not near:
        return None                                              # nothing to show off: a normal shot
    c = min(near)[1]
    x = [c - HERO_RUSH * bf - HERO_SLOW * (HERO_AT - 1.0) * bf]
    for j in range(1, L + 1):
        v = HERO_RUSH if j == 1 else HERO_SLOW if j <= 3 else sp
        x.append(x[-1] + v * bf)
    sh['hero_frame'] = int(c)
    return np.array(x)


def ramp_all(scenes, flights):
    shots = [x for s in scenes for x in s['shots']]
    by = {}
    for x in shots:
        by.setdefault(x['flight'], []).append(x)
    for fi, lst in by.items():
        f = flights[fi]
        lst.sort(key=lambda x: x['src_frame'])
        mg = int(0.3 * f.fps)
        for k, x in enumerate(lst):
            lo = lst[k - 1]['src_frame'] + lst[k - 1]['src_frames'] + mg if k else 0
            hi = lst[k + 1]['src_frame'] - mg if k + 1 < len(lst) else f.n - 1
            anchors = hero_ramp(x, f) if x['hero_ramp'] else None
            if anchors is None:
                x['hero_ramp'] = False
                anchors = snap(x, f, lo, hi)
            if anchors is None:
                constant_speed(x)
                continue
            x['anchors'] = [float(v) for v in anchors]
            x['anchor_t'] = np.linspace(0, x['t1'] - x['t0'], len(anchors)).tolist()
            x['src_frame'] = int(np.floor(anchors[0]))
            x['src_frames'] = int(np.ceil(anchors[-1] - anchors[0]))


def constant_speed(x):
    x['anchors'] = [float(x['src_frame']), float(x['src_frame'] + x['src_frames'])]
    x['anchor_t'] = [0.0, x['t1'] - x['t0']]


def time_map(x):
    """Output time inside the shot -> source frame (monotone, smooth speed)."""
    return PchipInterpolator(np.array(x['anchor_t']), np.array(x['anchors']))


# ================================================================== reports
def sync_stats(scenes, flights, m):
    on = tot = 0
    for x in (x for s in scenes for x in s['shots']):
        f = flights[x['flight']]
        a, b = x['anchors'][0], x['anchors'][-1]
        pk = f.peak_frames[(f.peak_frames >= a) & (f.peak_frames < b)]
        uu = np.linspace(0, x['t1'] - x['t0'], 400)
        tt = x['t0'] + np.interp(pk, time_map(x)(uu), uu)
        d = np.abs(tt[:, None] - m['beats'][None, :]).min(1) if tt.size else tt
        tot += tt.size
        on += int((d < 0.04).sum())
    return on, tot


def plot_plan(scenes, flights, m, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (a1, a2, a4, a3) = plt.subplots(4, 1, figsize=(18, 10), sharex=True,
                                         gridspec_kw=dict(height_ratios=[2, 3, 1.5, 1.2]))
    a1.plot(m['env_t'], m['env'], color='0.6', lw=0.4)
    b = m['beats']
    a1.vlines(b[(np.arange(b.size) - m['downbeat']) % 4 == 0], 0, 8, color='r', lw=0.4)
    cols = plt.cm.tab20(np.arange(max(20, len(flights))) % 20)    # more flights than colours: they repeat
    for s in scenes:
        t0, t1 = s['shots'][0]['t0'], s['shots'][-1]['t1']
        for a in (a1, a2, a4, a3):
            a.axvspan(t0, t1, color=cols[s['flight']], alpha=0.08)
        tag = 'HERO ' if s['hero'] else 'SLOW ' if s['slow_hero'] else 'FLY ' if s.get('fly') else ''
        a1.text((t0 + t1) / 2, 9.2, tag + f"c{s['cls']}", ha='center', fontsize=7)
        for x in s['shots']:
            a1.axvline(x['t0'], color='k', lw=1.2 if x is s['shots'][0] else 0.5)
    a1.set_ylim(0, 10)
    a1.set_ylabel('music onsets\nred: downbeats\nblack: cuts')
    for s in scenes:
        for x in s['shots']:
            f = flights[x['flight']]
            uu = np.linspace(0, x['t1'] - x['t0'], max(int((x['t1'] - x['t0']) * 50), 3))
            tm = time_map(x)
            fr = np.clip(np.round(tm(uu)).astype(int), 0, f.n - 1)
            spd = tm.derivative()(uu) / f.fps
            a2.plot(x['t0'] + uu, spd * f.vis[fr], color=cols[f.idx], lw=1)
            pk = f.peak_frames[(f.peak_frames >= x['anchors'][0]) & (f.peak_frames < x['anchors'][-1])]
            a2.plot(x['t0'] + np.interp(pk, tm(uu), uu), np.full(pk.size, -10), '|', color='k', ms=8)
            tr = f.tricks[(f.tricks >= x['anchors'][0]) & (f.tricks < x['anchors'][-1])]
            a2.plot(x['t0'] + np.interp(tr, tm(uu), uu), np.full(tr.size, 215), 'v', color='r', ms=6)
            a4.plot(x['t0'] + uu, f.skill[fr], color=cols[f.idx], lw=1)
            a3.plot(x['t0'] + uu, spd, color=cols[f.idx], lw=1)
    for f in flights:
        a2.plot([], [], color=cols[f.idx], label=f'F{f.idx} {f.name}')
    a2.legend(fontsize=7, ncol=3, loc='upper left')
    a2.set_ylim(-20, 230)
    a2.set_ylabel('visible rotation deg/s\n(ticks: onsets, v: manoeuvres)')
    a4.set_ylim(0, 1)
    a4.set_ylabel('piloting score\n(percentile)')
    a3.set_ylim(0.3, 1.8)
    a3.set_ylabel('playback speed')
    a3.set_xlabel('music time, s')
    a3.set_xlim(0, m['duration'])
    for a in (a1, a2, a4, a3):
        a.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(path, dpi=80)
    plt.close(fig)


# ================================================================== render
def levels(px):
    """[black in, r, g, b gains] for pixels (N, 3) after the LUT: out = (in - black in) * gain + GRADE_BLACK."""
    y = px @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    lo, hi = np.percentile(y, 0.5), np.percentile(y, 99.5)
    gain = float(np.clip((GRADE_WHITE - GRADE_BLACK) / max(hi - lo, 1.0), *GRADE_GAIN))
    wb = np.ones(3)
    wh = px[(y > np.percentile(y, 90)) & (y < hi) & (px.max(1) - px.min(1) < 40)]
    if len(wh) > 0.005 * len(px):
        m = wh.mean(0)
        wb = np.clip(1 + GRADE_WB * (m.mean() / m - 1), *GRADE_WB_MAX)
    return [float(lo)] + [gain * float(v) for v in wb]


def shot_grade(f, a, n, lut):
    """Levels and white balance through one shot, from 9 of its frames: one key per third
    -> [[place in the shot 0..1, black in, r, g, b gains], ...]; a shot whose thirds agree gets one key.
    A shot that flies from a dark staircase onto a bright court changes levels as it goes."""
    c = av.open(f.proxy or f.stab)
    st = c.streams.video[0]
    w = 320
    h = int(round(w * st.codec_context.height / st.codec_context.width))
    want = list(np.linspace(a, a + max(n, 1) - 1, 9).astype(int))
    c.seek(int(want[0] / f.fps / st.time_base), stream=st, backward=True)
    px = []
    for fr in c.decode(st):
        i = int(round(float(fr.pts * st.time_base) * f.fps))
        while want and i >= want[0]:
            img = fr.reformat(width=w, height=h, format='rgb24', src_colorspace='itu709', dst_colorspace='itu709',
                              src_color_range='MPEG', dst_color_range='JPEG').to_ndarray()
            px.append((lut(img) if lut is not None else img).reshape(-1, 3).astype(np.float32))
            want.pop(0)
        if not want:
            break
    c.close()
    if not px:
        return [[0.5, 0.0, 1.0, 1.0, 1.0]]
    keys = [[rel] + levels(np.concatenate(px[k:k + 3])) for k, rel in zip((0, 3, 6), (1 / 6, 0.5, 5 / 6))
            if px[k:k + 3]]
    g = np.array([k[1:] for k in keys])
    if len(keys) < 3 or (np.ptp(g[:, 0]) < GRADE_STEADY[0] and np.ptp(g[:, 1:], axis=0).max() < GRADE_STEADY[1]):
        return [[0.5] + levels(np.concatenate(px))]
    return keys


def grade_at(keys, rel):
    """The grade at a place in the shot (0..1), linear between keys; old EDLs hold one flat list."""
    if not keys:
        return None
    if not isinstance(keys[0], (list, tuple)):
        return list(keys)
    if len(keys) == 1:
        return list(keys[0][1:])
    k = np.array(keys, np.float64)
    return [float(np.interp(rel, k[:, 0], k[:, c])) for c in range(1, 5)]


def apply_grade(img, g):
    return np.clip((img.astype(np.float32) - g[0]) * np.array(g[1:], np.float32) + GRADE_BLACK, 0, 255).astype(np.uint8)


def grade_all(scenes, flights, lut):
    """--grade auto: every shot's levels and white balance (x['grade'], keys through the shot)."""
    for x in (x for s in scenes for x in s['shots']):
        x['grade'] = shot_grade(flights[x['flight']], x['src_frame'], x['src_frames'], lut)
    g = np.array([k[1:] for s in scenes for x in s['shots'] for k in x['grade']])
    moving = sum(1 for s in scenes for x in s['shots'] if len(x['grade']) > 1)
    print(f'  grade: black in {g[:, 0].min():.0f}..{g[:, 0].max():.0f}, gain r {g[:, 1].min():.2f}..{g[:, 1].max():.2f} '
          f'g {g[:, 2].min():.2f}..{g[:, 2].max():.2f} b {g[:, 3].min():.2f}..{g[:, 3].max():.2f}; '
          f'{moving} shots change levels as they go')


def render(scenes, flights, m, out, height, fps_out, debug=False, crf=23, lut=None, zoom=1.0):
    with av.open(flights[0].proxy or flights[0].stab) as c0:             # keep the footage's aspect (4:3 or 16:9)
        cc = c0.streams.video[0].codec_context
        aspect = cc.width / cc.height
    W = int(round(height * aspect / 2)) * 2
    H = height
    Wz, Hz = max(int(round(W * zoom / 2)) * 2, W), max(int(round(H * zoom / 2)) * 2, H)   # --zoom: centre cut
    zx, zy = (Wz - W) // 2, (Hz - H) // 2
    dur = m['duration']
    oc = av.open(out, 'w')
    vs = oc.add_stream('libx264', rate=fps_out)
    vs.width, vs.height, vs.pix_fmt = W, H, 'yuv420p'
    vs.options = {'crf': str(crf), 'preset': 'veryfast'}
    for k, v in (('colorspace', 1), ('color_primaries', 1), ('color_trc', 1), ('color_range', 1)):
        try:                                                     # tag BT.709, limited range
            setattr(vs.codec_context, k, v)
        except (AttributeError, TypeError, ValueError):
            pass
    to_rgb = dict(width=Wz, height=Hz, format='rgb24', src_colorspace='itu709', dst_colorspace='itu709',
                  src_color_range='MPEG', dst_color_range='JPEG')
    as_ = oc.add_stream('aac', rate=48000)
    as_.layout = 'stereo'
    as_.bit_rate = 160000
    with av.open(m['path']) as ic:                               # audio first: the muxer interleaves
        rs = av.AudioResampler(format='fltp', layout='stereo', rate=48000)
        for fr in ic.decode(audio=0):
            for o in rs.resample(fr):
                for p in as_.encode(o):
                    oc.mux(p)
        for o in rs.resample(None):
            for p in as_.encode(o):
                oc.mux(p)
    for p in as_.encode(None):
        oc.mux(p)

    beats, d0 = m['beats'], m['downbeat']
    shots = [(s, x) for s in scenes for x in s['shots']]
    readers = {}
    nout = int(np.floor(dur * fps_out))
    k = 0
    t_start = time.time()
    for si, (sc, x) in enumerate(shots):
        f = flights[x['flight']]
        if f.idx not in readers:
            c = av.open(f.proxy or f.stab)
            st = c.streams.video[0]
            st.thread_type = 'AUTO'
            readers[f.idx] = (c, st)
        c, st = readers[f.idx]
        j0 = int(np.ceil(x['t0'] * fps_out - 1e-9))
        j1 = nout if si == len(shots) - 1 else int(np.ceil(x['t1'] * fps_out - 1e-9))
        tm = time_map(x)
        want = np.clip(np.round(tm(np.arange(j0, j1) / fps_out - x['t0'])).astype(int), 0, f.n - 1)
        spd = tm.derivative()(np.arange(j0, j1) / fps_out - x['t0']) / f.fps
        c.seek(int(want[0] / f.fps / st.time_base), stream=st, backward=True)
        it = c.decode(st)
        cur, img = -1, None
        for jj, w, v in zip(range(j0, j1), want, spd):
            while cur < w:
                fr = next(it)
                cur = int(round(float(fr.pts * st.time_base) * f.fps))
                if cur >= w:
                    img = fr.reformat(**to_rgb).to_ndarray()[zy:zy + H, zx:zx + W]
                    if lut is not None:
                        img = lut(img)
                    if x.get('grade'):
                        img = apply_grade(img, grade_at(x['grade'], (w - x['src_frame']) / max(x['src_frames'], 1)))
            t = jj / fps_out
            g = min(1.0, t / FADE_IN_S, (dur - t) / FADE_OUT_S)
            o = img if g >= 1.0 else (img * max(g, 0.0)).astype(np.uint8)
            if debug:
                o = np.ascontiguousarray(o)
                tag = 'H' if sc['hero'] else 'S' if sc['slow_hero'] else ''
                txt = (f'{si} sc{sc["idx"]}{tag} F{f.idx} {w / f.fps:6.1f}s x{v:.2f} '
                       f'q{x["q_music"]:.2f} sk{x["skill"]:.2f}')
                cv2.putText(o, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
                bi = np.searchsorted(beats, t + 0.5 / fps_out) - 1
                if bi >= 0 and t - beats[bi] < 0.08:
                    cv2.circle(o, (W - 24, 24), 14 if (bi - d0) % 4 == 0 else 7, (255, 0, 0), -1)
            vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(o), format='rgb24').reformat(
                format='yuv420p', src_colorspace='itu709', dst_colorspace='itu709',
                src_color_range='JPEG', dst_color_range='MPEG')
            vf.pts = k
            for p in vs.encode(vf):
                oc.mux(p)
            k += 1
        print(f'\r  render {k / fps_out:6.1f}/{dur:.1f}s  ({time.time() - t_start:.0f}s)', end='', flush=True)
    for p in vs.encode(None):
        oc.mux(p)
    oc.close()
    for c, _ in readers.values():
        c.close()
    print()


def make_parser():
    """The command line (montage.py checks every edit's arguments with it before anything runs)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('music')
    ap.add_argument('footage', nargs='?', default=None, help='footage folder (or --originals)')
    ap.add_argument('--originals', default=None, help='text file: one original clip per line')
    ap.add_argument('--ramps', action='store_true', help='bend speed inside shots (off: every frame at 1x)')
    ap.add_argument('--people', type=float, default=0.0, help='weight of people in the picture (YOLO); 4 = as much as piloting')
    ap.add_argument('--fly', type=int, default=0, help='flying scenes: this many phrases on the biggest lifts go to the best piloting')
    ap.add_argument('--must', action='append', default=[],
                    help='CLIP:SECONDS or CLIP:START-END, a moment or a piece the edit has to show (repeatable); '
                         '--must, --start and --end also take find:TEXT, e.g. "find:a corridor -> a staircase -> '
                         'a tennis court" (places.py: the best stretch, or the best route through the places)')
    ap.add_argument('--end', default=None, help='CLIP:SECONDS or CLIP:START-END, what the last shot shows')
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--target', type=float, default=None, help='shorten the track to about this many seconds')
    ap.add_argument('--seed', type=int, default=None, help='variation number; omitted = deterministic best')
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--fps', type=int, default=25)
    ap.add_argument('--no-ramps', action='store_true', help='constant speed inside shots')
    ap.add_argument('--debug', action='store_true', help='overlay shot info and a beat flash')
    ap.add_argument('--plan-only', action='store_true')
    ap.add_argument('--cache', default=None, help='telemetry cache folder (default: next to the output)')
    ap.add_argument('--proxy-dir', default=None, help='proxy.py output: 480p proxies and image motion features')
    ap.add_argument('--lut', default=None, help='.cube LUT for the picture, e.g. D-Log M -> Rec.709')
    ap.add_argument('--bpm', type=float, default=None, help='tempo hint when the detector picks the wrong one')
    ap.add_argument('--avoid', action='append', default=[], help='EDL of an earlier edit of the series: its footage is taken')
    ap.add_argument('--subject', default='', help='main character tag: people.py subject --tag NAME')
    ap.add_argument('--minor', default='', help="another edit's main character (subject tag): counts like the pilot")
    ap.add_argument('--start', default=None, help='CLIP:START-END, the piece that opens the edit')
    import paths
    ap.add_argument('--clip-cache', default=os.path.join(paths.MODELS, 'clip'),
                    help='CLIP model folder for find:TEXT pieces')
    ap.add_argument('--grade', choices=('none', 'auto'), default='none',
                    help='auto: levels and white balance per shot after the LUT, alike across the edit')
    ap.add_argument('--peers', default=None,
                    help='--focus: comma-separated subject tags of look-alikes that must not stand in for the main '
                         'character (other children of the series); <subject>_not is always one')
    ap.add_argument('--focus', action='store_true', help='an edit about the --subject: others count FOCUS_OTHERS')
    ap.add_argument('--lively', action='store_true', help='4-beat shots in loud phrases (PATTERNS_LIVELY)')
    ap.add_argument('--heroes', type=int, default=0,
                    help='hero ramps (with --ramps) on this many of the biggest music lifts, never the same manoeuvre kind twice')
    ap.add_argument('--pace', action='store_true', help='loud phrases play 1.1-1.2x (BASE_SPEED_PACE): the flying looks faster')
    ap.add_argument('--want', action='append', default=[], choices=EVENT_KINDS,
                    help='a manoeuvre kind the edit asks for (repeatable), e.g. ascent: shots with one in the middle '
                         'get W_WANT')
    ap.add_argument('--zoom', type=float, default=1.0,
                    help='cut the centre 1/ZOOM of the picture (kept in the EDL for final.py): 1.16 hides the edges of '
                         'a lens guard that the stabilized frame still shows')
    ap.add_argument('--beauty', type=float, default=0.0,
                    help='weight of beautiful views (beauty.py, in the words of the job); 4 = as much as piloting. A '
                         'beautiful view also excuses a shot with nobody in it in a people edit')
    ap.add_argument('--chapters', default=None,
                    help='scenes per recording day in date order, e.g. 3,2,4,2 (0 leaves a day out); default: in '
                         'proportion to good flying (and beautiful views with --beauty)')
    ap.add_argument('--theme', action='append', default=[],
                    help='"TEXT[:N]", what the edit shows in words (repeatable): at most N scenes of it (default 1), '
                         'e.g. "a man on a ladder repairing a roof" when one activity fills the footage')
    ap.add_argument('--animals', type=float, default=None,
                    help='an animal against a person of the same size (ANIMAL_WORTH); 0 when the footage has none (in a garden '
                         'YOLO sees them in cabbage beds and clothes on a porch)')
    ap.add_argument('--arcs', type=int, default=0,
                    help='this many scenes each show a whole climb and the way down (ascents), the highest that fit '
                         'a phrase at PIECE_MAX_SPEED, from different flights where possible')
    return ap


def apply_options(a):
    """The weights and switches of the command line, set where the scoring reads them; -> the noise generator."""
    global MAX_HEROES, HERO_REUSE, ANIMAL_WORTH, W_PEOPLE, MAX_FLY, W_BEAUTY
    if a.lively:
        PATTERNS.update(PATTERNS_LIVELY)
    if a.pace:
        BASE_SPEED.update(BASE_SPEED_PACE)
    if a.animals is not None:
        ANIMAL_WORTH = a.animals
    WANT[:] = a.want
    THEMES.clear()
    for t in a.theme:
        text, _, cap = t.rpartition(':')
        THEMES.append((text, int(cap)) if text and cap.isdigit() else (t, 1))
    if a.heroes:
        MAX_HEROES, HERO_REUSE = a.heroes, 0.0
    SUBJECT_TAG[0] = a.subject
    MINOR_TAG[0] = a.minor
    FOCUS[0] = a.focus and bool(a.subject)
    PEER_TAGS[:] = ([t for t in (a.peers or '').split(',') if t and t != a.subject] +
                    ([a.subject + '_not'] if a.subject else []))
    W_PEOPLE, MAX_FLY, W_BEAUTY = a.people, a.fly, a.beauty
    CHAPTERS[:] = [int(v) for v in a.chapters.split(',')] if a.chapters else []
    END_SET[0] = a.end is not None
    return np.random.default_rng(a.seed) if a.seed is not None else None


def prepare_flights(flights, a):
    """What the options ask of the footage beyond find_flights: climbs, beautiful views, themes."""
    if 'ascent' in a.want or a.arcs:
        add_ascents(flights, a.clip_cache)
    if W_BEAUTY:
        set_beauty(flights)
    if THEMES:
        load_themes(flights, a.clip_cache)


def main():
    a = make_parser().parse_args()
    rng = apply_options(a)
    out = os.path.abspath(a.out)
    base = os.path.splitext(out)[0]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t0 = time.time()
    print('footage:')
    cache = a.cache or os.path.join(os.path.dirname(out), '.telemetry_cache')
    if a.originals:
        with open(a.originals, encoding='utf-8') as fh:
            flights = find_flights_list([x.strip() for x in fh if x.strip()], cache, a.proxy_dir)
    else:
        flights = find_flights(a.footage, cache, a.proxy_dir)
    prepare_flights(flights, a)
    for f in flights:
        kinds = ', '.join(f'{f.trick_kind.count(k)} {k}' for k in EVENT_KINDS
                          if f.trick_kind.count(k))
        print(f'  F{f.idx} {f.name}: {f.n / f.fps:.0f}s, usable {f.usable.mean() * 100:.0f}%, '
              f'{f.peak_frames.size} motion onsets, {kinds}, '
              f'proxy {"yes" if f.proxy else "no"}, flow {"yes" if f.flow is not None else "no"}, '
              f'looks {"yes" if f.look is not None else "no"}')
    print('music:')
    music_path, edit_info = a.music, None
    m = analyse_music(a.music, a.bpm)
    if a.target:
        import music_edit
        phrases, cls, _, _, _ = phrase_layout(m)
        music_path = base + '_music.wav'
        edit_info = music_edit.shorten(a.music, music_path, m, [(p, 16, c == 0, int(c)) for p, c in zip(phrases, cls)],
                                       a.target)
        print(f'  shortened {m["duration"]:.1f}s -> {edit_info["seconds"]:.1f}s, phrases {edit_info["phrases"]}, '
              f'join cost {edit_info["jump_cost"]:.2f}')
        m = analyse_music(music_path, a.bpm)
    print(f'  {os.path.basename(music_path)}: {m["duration"]:.1f}s, {m["bpm"]:.2f} BPM, {m["beats"].size} beats, '
          f'downbeat phase {m["downbeat"]}')
    scenes = plan_scenes(m, rng)
    scenes[-1]['last'] = True
    if all(hasattr(f, 'day') for f in flights):
        w, cnt = chapters(scenes, flights)
        print(f'  chapters (good seconds -> scenes): {", ".join(f"{d}: {w[d]:.0f}s -> {cnt[d]}" for d in cnt)}')
    musts = []
    for spec, last, first in ([(x, False, False) for x in a.must] + ([(a.end, True, False)] if a.end else [])
                              + ([(a.start, False, True)] if a.start else [])):
        if spec.startswith('find:'):                             # the piece in words (places.py, CLIP on a GPU)
            import places
            hit = places.find([f.stab for f in flights], spec[5:], a.clip_cache)
            if hit is None:
                raise SystemExit(f'{spec}: nothing in the footage looks like that')
            r, t0_, t1_, pr = hit
            f = next(f for f in flights if os.path.normcase(f.stab) == os.path.normcase(r))
            spec = f'{f.name}:{t0_:.1f}-{t1_:.1f}'
            print(f'  found {spec} (p {pr:.2f})')
        stem, sec = spec.rsplit(':', 1)
        f = next((f for f in flights if f.name.lower() == os.path.splitext(os.path.basename(stem))[0].lower()), None)
        if f is None:
            raise SystemExit(f'{spec}: no such clip among the flights')
        if '-' in sec:                                           # a piece, shown whole
            lo, hi = (int(round(float(v) * f.fps)) for v in sec.split('-'))
            end_at = hi if last else None
        else:                                                    # a moment: 0.5 s before to 1 s after it
            fr = int(round(float(sec) * f.fps))
            lo, hi = fr - int(0.5 * f.fps), fr + int(1.0 * f.fps)
            end_at = fr + int(END_AFTER_S * f.fps) if last else None
        # the last shot runs on while the music rings out, so the ending opens more footage before the moment
        o0 = max(lo - int((END_OPEN_S if last else MUST_OPEN_S) * f.fps), 0)
        o1 = min(max(hi, end_at or 0) + int(MUST_OPEN_S * f.fps), f.n)
        f.usable[o0:o1] = True                                   # the editor's call beats the automatic trimming
        f.cu = csum(f.usable.astype(np.float64))
        musts.append((f.idx, lo, hi, end_at, first))
    if a.arcs:
        musts += pick_arcs(flights, m, a.arcs, musts)
    if any(x[4] for x in musts) and len(scenes) > 1 and scenes[0]['kind'] == 'pickup' and \
            not (scenes[1]['hero'] or scenes[1]['fly']):
        # the opening piece opens the edit: the pickup before the first phrase joins that phrase's first shot
        # (it took the opening shot otherwise, and the piece came second)
        p, s1 = scenes[0]['shots'][0], scenes[1]['shots']
        s1[0] = dict(merged([p, s1[0]]), cls=s1[0]['cls'], speed=s1[0]['speed'], q_music=s1[0]['q_music'])
        scenes = scenes[1:]
        for i, s in enumerate(scenes):
            s['idx'] = i
    avoid = avoided(a.avoid, flights)
    if a.avoid:
        print(f'  footage taken by {len(a.avoid)} earlier edits: '
              f'{sum(b - a_ for v in avoid.values() for a_, b in v) / flights[0].fps:.0f}s')
    scenes = assign(scenes, flights, m, rng, musts, avoid)
    if a.no_ramps or not a.ramps:
        for x in (x for s in scenes for x in s['shots']):
            constant_speed(x)
    else:
        ramp_all(scenes, flights)
    shots = [x for s in scenes for x in s['shots']]
    sw = sum(1 for p, q in zip(scenes, scenes[1:]) if p['flight'] != q['flight'])
    on, tot = sync_stats(scenes, flights, m)
    heroes = [x for x in shots if x.get('hero_frame') is not None]
    print(f'  {len(scenes)} scenes ({sum(s["hero"] for s in scenes)} hero, {sum(s["slow_hero"] for s in scenes)} slow), '
          f'{len(shots)} shots, flight changes at {sw}/{len(scenes) - 1} scene boundaries')
    if any('people' in x for x in shots):
        pp = np.array([x.get('people', 0.0) for x in shots])
        dur_p = sum(x['t1'] - x['t0'] for x in shots if x.get('people', 0) > 0.15)
        print(f'  people: in {np.mean(pp > 0.15) * 100:.0f} % of shots ({dur_p:.0f} s of {m["duration"]:.0f} s), mean presence {pp.mean():.2f}')
    fly = [x['skill'] for s in scenes if s.get('fly') for x in s['shots']]
    print(f'  flying scenes: {sum(s.get("fly", False) for s in scenes)}, piloting there {np.median(fly) if fly else 0:.2f}, '
          f'manoeuvres {[x["fly_kind"] for x in shots if x.get("fly_kind")]}; '
          f'subject in {sum(1 for x in shots if x.get("subject", 0) > 0.1)}/{len(shots)} shots, '
          f'animals in {sum(1 for x in shots if x.get("animal", 0) > 0.1)}')
    print(f'  piloting score of the edit: median {np.median([x["skill"] for x in shots]):.2f} (footage median 0.50); '
          f'hero ramps with a manoeuvre {len(heroes)}/{sum(s["hero"] for s in scenes)}')
    mv = [x['move'] for x in shots]
    run = best_run = 1
    for a_, b_ in zip(mv, mv[1:]):
        run = run + 1 if a_ == b_ else 1
        best_run = max(best_run, run)
    print(f'  movements: {dict((k, mv.count(k)) for k in MOVES if mv.count(k))}, longest run of one kind {best_run}; '
          f'hero manoeuvres: {[x.get("hero_kind") for x in shots if x.get("hero_ramp")]}; '
          f'opening piloting {np.mean([x["skill"] for x in shots[:3]]):.2f}')
    lk = [(s['idx'], shot_look(flights[x['flight']], x['src_frame'], x['src_frames'])) for s in scenes for x in s['shots']]
    lk = [(i, d) for i, d in lk if d is not None]
    close = sum(1 for k, (i, a) in enumerate(lk) for j, b in lk[k + 1:]
                if i != j and np.sqrt(((a - b) ** 2).mean()) < LOOK_SIGMA)
    print(f'  look-alike shot pairs across scenes (RMS < {LOOK_SIGMA:.1f}): {close}')
    if WANT:
        got = [kind for x in shots for c, kind in zip(flights[x['flight']].tricks, flights[x['flight']].trick_kind)
               if kind in WANT and x['src_frame'] <= c < x['src_frame'] + x['src_frames']]
        print(f'  wanted manoeuvres in the edit: {dict((k, got.count(k)) for k in WANT)}')
    if THEMES:
        print('  themes, scenes / allowed: ' + '; '.join(
            f'{t} {sum(any(x.get("theme") == t for x in s["shots"]) for s in scenes)}/{c}' for t, c in THEMES))
    print(f'  motion onsets in the edit: {tot}, on a beat (+-40 ms): {on} ({on / max(tot, 1) * 100:.0f}%)  '
          f'({time.time() - t0:.0f}s)')
    lut = CubeLUT(a.lut) if a.lut else None
    if a.grade == 'auto':
        grade_all(scenes, flights, lut)
    edl = dict(music=os.path.abspath(music_path), music_edit=edit_info, bpm=m['bpm'], seed=a.seed,
               subject=a.subject or None, focus=bool(FOCUS[0]), zoom=a.zoom,
               lut=a.lut, flights=[dict(idx=f.idx, original=f.src, render=f.stab, render_offset=f.offset,
                                        proxy=f.proxy, fps=f.fps, day=getattr(f, 'day', None),
                                        recorded=getattr(f, 'when', None) and f.when.isoformat()) for f in flights],
               scenes=[dict(idx=s['idx'], kind=s['kind'], cls=s['cls'], hero=s['hero'], slow_hero=s['slow_hero'], fly=s['fly'],
                            must=bool(s.get('must')),
                            flight=s['flight'], day=s.get('day'),
                            shots=[{k: v for k, v in x.items() if k not in ('beat_w', 'bpm')}
                                   | dict(src_t0=x['src_frame'] / flights[x['flight']].fps) for x in s['shots']])
                       for s in scenes])
    with open(base + '_edl.json', 'w', encoding='utf-8') as fh:
        json.dump(edl, fh, ensure_ascii=False, indent=1)
    plot_plan(scenes, flights, m, base + '_plan.png')
    if not a.plan_only:
        m['path'] = music_path
        render(scenes, flights, m, out, a.height, a.fps, a.debug, lut=lut, zoom=a.zoom)
    print(f'done in {time.time() - t0:.0f}s -> {out}')


if __name__ == '__main__':
    main()
