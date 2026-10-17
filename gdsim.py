"""Geometry Dash physics simulation: plays a level and measures what the route demands.

Physics follows GD 2.2 as documented by the camila314/gdp decompilation: fixed 240 Hz steps (dt = 0.25 of a
60 Hz frame), the per-speed jump velocities and gravities, and the per-gamemode acceleration rules of
PlayerObject::updateJump. A planner (dynamic programming over reachable player states, 60 Hz decisions)
then finds the cheapest way through, and reruns with the hazards fattened to see how much room there is.

Everything hot is numba-compiled; geometry is flat numpy arrays with a 30-unit column index.
"""
import json, math, os
import numpy as np
from numba import njit
import gdlevel
from numba.typed import List

HERE = os.path.dirname(os.path.abspath(__file__))
SHAPES = json.load(open(os.path.join(HERE, "objshapes.json")))
FLAT = SHAPES.get("flat", {})                          # ground-hugging hazards: id -> sprite height
GAME = SHAPES.get("source") == "game"                  # shapes dumped from the game (tools/build_hitboxes.py): they also decide what is solid
PAD_K, RING_K = float(os.environ.get("GDSIM_PAD_K", 1.0)), float(os.environ.get("GDSIM_RING_K", 1.0))                                # strength scales for pads and orbs (calibration knobs)
TS = float(os.environ.get("GDSIM_TS", 0.9))            # GD moves the player vertically by velocity x dt x 0.9 (and applies gravity the same way)
HZ_K = float(os.environ.get("GDSIM_HZ_K", 1.0))        # hazards are tested against the player's full box (x this factor), as in GD
PIT_A, PIT_B, PIT_OY = 0.85, 0.75, 0.0                # their hitbox: fractions of the 30 x height sprite box, offset as a fraction of the height

# ----------------------------------------------------------------- constants (per speed index 0..4 = 1x, .5x, 2x, 3x, 4x)
DT = 0.25
SP_PS = np.array([0.9, 0.7, 1.1, 1.3, 1.6])
SP_MULT = np.array([5.77000189, 5.980002, 5.870002, 6.000002, 6.000002])
VX = SP_PS * SP_MULT                                  # units per 60 Hz frame: 5.193 / 4.186 / 6.457 / 7.8 / 9.6
YSTART = np.array([11.1800318, 10.620032, 11.420032, 11.230032, 11.230032])
GRAV = np.array([0.958199024, 0.940199, 0.957199, 0.961199, 0.961199])
G_FLY = 0.9582                                        # gravity used by ship / ufo / ball / spider / swing
CUBE, SHIP, BALL, UFO, WAVE, ROBOT, SPIDER, SWING = range(8)

# state vector layout
Y, VY, UP, GND, BOOST, ACC, RACC, BUF, PAGE, RING, PAD, INV, SLP, TEL, NS = range(15)

# ring (orb) and pad types
ORB_TYPE = {36: 1, 141: 2, 1333: 3, 84: 4, 1022: 5, 1330: 6}
PAD_TYPE = {35: 1, 140: 2, 1332: 3, 67: 4}
# strength as a multiple of the speed's jump velocity (placeholders until calibrated against real levels)
# Orb launch as a multiple of the speed's jump velocity, per orb type (1 yellow, 2 pink, 3 red, 4 blue, 5 green) and
# gamemode group (cube/robot, ship/swing, ball/spider, ufo). Pad launch in units per frame, per pad type (1 yellow, 2 pink,
# 3 red, 4 blue) and group. Values: the GD 2.x tables of the camila314/pathfinder simulator (velocity / 54).
RING_MULT = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.700, 1.0], [0.720, 0.370, 0.539, 0.420], [1.3607, 0.9938, 0.9264, 1.0085],
                      [0.400, 0.400, 0.280, 0.400], [0.9807, 0.7077, 0.6885, 0.7156], [0.0, 0.0, 0.0, 0.0]])
PAD_V = np.array([[0.0, 0.0, 0.0, 0.0], [16.0, 16.0, 9.6, 8.0], [10.4, 5.6, 6.72, 6.4], [20.0, 10.08, 12.0, 9.6], [6.4, 6.4, 3.84, 6.4]])
BLACK_V = 15.0                                         # black orb: slam toward the floor (810 / 54)
GROUP = np.array([0, 1, 2, 3, -1, 0, 2, 1])            # gamemode -> orb / pad group (-1: wave, no launches)
# ship / ufo switch between their light and heavy acceleration at this velocity, per speed (pathfinder's thresholds / 54;
# the ufo's 0.101 vs 0.152 per tick at low speed confirmed it in the game)
VTHR = np.array([1.91640, 1.88040, 1.91440, 1.92240, 1.92240])
RING_R = 18.0

EV_MODE, EV_SIZE, EV_GRAV, EV_SPEED, EV_DUAL = 0, 1, 2, 3, 4
YMAX = 1500.0                                          # above this (or 600 above the highest object) the player has left the level
CUBE_CEIL = int(os.environ.get("GDSIM_CUBE_CEIL", 0))    # 0: a cube / robot portal removes the ceiling (as in the game), 1: the previous ceiling stays
CUBE_RESET = True                                      # a cube / robot portal drops the floor back to the ground line
NOCOLL = ("121",)                                         # "No Touch" removes the hitbox; 135 ("Hide") only hides the sprite: still deadly in the game


PARSE_KEYS = frozenset(gdlevel.AUXK) | {"128", "129",       # + 2.2 scale X / scale Y
                                         "30", "85", "11", "62", "63", "56", "58", "59", "100", "394", "68", "69", "71", "54"}   # trigger easing / spawn / toggle / lock keys
TELEPORT = (747,) if os.environ.get("GDSIM_NO_TELE") != "1" else ()   # (2902 did not teleport in the game)                                 # teleport portals: key 54 = how far above the entrance the exit sits
TELE_BOX = (25.0, 90.0, 12.0)                          # their trigger box and its x offset from the object (game dump; checked in the game);
                                                       # the box turns with the portal, the exit stays straight above (both seen in the game)
DYN_RES = 2.0                                          # x resolution of the trigger-driven offsets (units of player x)
DYN_CAP = 12_000_000                                   # most floats spent on offset tracks per level (else coarser resolution)


def _ease(f, e, r):
    """GD easing (cocos2d actions) of a 0..1 progress array: 0 none, 1-3 ease in-out / in / out (rate r), 4-6 elastic,
    7-9 bounce, 10-12 exponential, 13-15 sine, 16-18 back."""
    f = np.clip(f, 0.0, 1.0)
    r = r if r > 0 else 2.0
    if e in (1, 2, 3):
        if e == 2:
            return f ** r
        if e == 3:
            return f ** (1.0 / r)
        g = 2.0 * f
        return np.where(g < 1.0, 0.5 * g ** r, 1.0 - 0.5 * np.abs(2.0 - g) ** r)
    if e in (4, 5, 6):
        per = r if 0 < r <= 1.0 else 0.3
        sh = per / 4.0
        out = lambda t: np.where(t >= 1.0, 1.0, np.power(2.0, -10.0 * t) * np.sin((t - sh) * 2.0 * np.pi / per) + 1.0)
        if e == 6:
            return np.where(f <= 0, 0.0, out(f))
        if e == 5:
            return 1.0 - out(1.0 - f)
        return np.where(f < 0.5, 0.5 * (1.0 - out(1.0 - 2.0 * f)), 0.5 * out(2.0 * f - 1.0) + 0.5)
    if e in (7, 8, 9):
        def bout(t):
            return np.where(t < 1 / 2.75, 7.5625 * t * t, np.where(t < 2 / 2.75, 7.5625 * (t - 1.5 / 2.75) ** 2 + 0.75,
                            np.where(t < 2.5 / 2.75, 7.5625 * (t - 2.25 / 2.75) ** 2 + 0.9375, 7.5625 * (t - 2.625 / 2.75) ** 2 + 0.984375)))
        if e == 9:
            return bout(f)
        if e == 8:
            return 1.0 - bout(1.0 - f)
        return np.where(f < 0.5, 0.5 * (1.0 - bout(1.0 - 2.0 * f)), 0.5 * bout(2.0 * f - 1.0) + 0.5)
    if e in (10, 11, 12):
        if e == 11:
            return np.where(f <= 0, 0.0, np.power(2.0, 10.0 * (f - 1.0)))
        if e == 12:
            return np.where(f >= 1, 1.0, 1.0 - np.power(2.0, -10.0 * f))
        return np.where(f < 0.5, 0.5 * np.power(2.0, 10.0 * (2.0 * f - 1.0)), 0.5 * (2.0 - np.power(2.0, -10.0 * (2.0 * f - 1.0))))
    if e in (13, 14, 15):
        if e == 14:
            return 1.0 - np.cos(f * np.pi / 2.0)
        if e == 15:
            return np.sin(f * np.pi / 2.0)
        return -0.5 * (np.cos(np.pi * f) - 1.0)
    if e in (16, 17, 18):
        k = 1.70158
        if e == 17:
            return f * f * ((k + 1) * f - k)
        if e == 18:
            g = f - 1.0
            return g * g * ((k + 1) * g + k) + 1.0
        k2 = k * 1.525
        g = 2.0 * f
        return np.where(g < 1.0, 0.5 * g * g * ((k2 + 1) * g - k2), 0.5 * ((g - 2) ** 2 * ((k2 + 1) * (g - 2) + k2) + 2.0))
    return f


def _fnum(a, k, d=0.0):
    try:
        return float(a.get(k, d) or d)
    except ValueError:
        return d


def trigger_groups(objs, aux, gd):
    """Read the move / toggle / rotate / spawn triggers. -> (moves, toggles, rotations, spawns, bad groups). A rotate trigger
    without a center group turns each object in place, which the game does not apply to hitboxes (checked), so it is
    ignored; one with a center group swings the objects around that group (checked). Groups driven by anything else the
    simulator cannot predict (follow, lock to player, move to a target, scale) are 'bad': left out."""
    moves, toggles, rots, spawns, bad = [], [], [], [], set()
    for i, a in aux.items():
        oid, t = objs[i][0], a.get("51")
        if t is None:
            continue
        tg = frozenset(a["57"].split(".")) if "57" in a else frozenset()
        spawned, x = a.get("62") == "1", objs[i][1]
        if oid == 901:
            if any(a.get(k) not in (None, "0", "") for k in ("58", "59", "100", "394")):
                bad.add(t)
                continue
            moves.append((x, tg, spawned, t, _fnum(a, "28"), _fnum(a, "29"), max(_fnum(a, "10"), 0.0), int(_fnum(a, "30")), _fnum(a, "85", 2.0)))
        elif oid == 1049:
            toggles.append((x, tg, spawned, t, a.get("56") == "1"))
        elif oid == 1268:
            spawns.append((x, tg, spawned, t, max(_fnum(a, "63"), 0.0)))
        elif oid == 1346:
            deg, cg = _fnum(a, "68") + 360.0 * _fnum(a, "69"), a.get("71", "0") or "0"
            if cg != "0" and deg != 0.0:
                rots.append((x, tg, spawned, t, deg, max(_fnum(a, "10"), 0.0), int(_fnum(a, "30")), _fnum(a, "85", 2.0), cg))
        elif oid in gd.MOVERS:
            bad.add(t)
    return moves, toggles, rots, spawns, bad


def dyn_tracks(tracks, moves, toggles, spawns, sch_x, x_lo, x_hi, rots=(), centers=None):
    """Offsets (dx, dy) of each track (a tuple of trigger-driven groups) at every DYN_RES of player x. In a classic level
    the player's x is a fixed function of time, so every trigger's start time and every move's progress is known in
    advance. A track switched off by a toggle trigger is parked far away (dy = 1e7). Triggers fire when the player's x
    reaches theirs (checked in the game); spawn-triggered ones when a spawn trigger reaches their group, after its delay."""
    T = len(sch_x) - 1
    tick_of = lambda x: float(np.searchsorted(sch_x, x))
    fire = {}                                                     # trigger key -> list of fire ticks
    allt = [("m", k, m) for k, m in enumerate(moves)] + [("t", k, m) for k, m in enumerate(toggles)] + [("s", k, m) for k, m in enumerate(spawns)] \
        + [("r", k, m) for k, m in enumerate(rots)]
    frontier = []
    for kind, k, m in allt:
        if not m[2]:
            fire[(kind, k)] = [tick_of(m[0])]
            if kind == "s":
                frontier.append((k, fire[(kind, k)][0]))
    for _ in range(6):                                            # spawn chains, a few levels deep
        nxt = []
        for k, t0 in frontier:
            sp = spawns[k]
            for kind, j, m in allt:
                if m[2] and sp[3] in m[1]:
                    lst = fire.setdefault((kind, j), [])
                    if len(lst) < 8:
                        t1 = t0 + sp[4] * 240.0
                        lst.append(t1)
                        if kind == "s":
                            nxt.append((j, t1))
        frontier = nxt
    ncell = int((x_hi - x_lo) / DYN_RES) + 2
    res = DYN_RES
    while len(tracks) * ncell * 3 > DYN_CAP and res < 64:
        res *= 2.0
        ncell = int((x_hi - x_lo) / res) + 2
    cx = x_lo + np.arange(ncell) * res
    ticks = np.interp(cx, sch_x, np.arange(T + 1, dtype=np.float64))
    per = dict(m={}, t={}, r={})
    for (kind, k), lst in fire.items():
        if kind == "s":
            continue
        m = (moves if kind == "m" else toggles if kind == "t" else rots)[k]
        for t0 in lst:
            per[kind].setdefault(m[3], []).append((t0, m))
    per_group_moves, per_group_tog = per["m"], per["t"]
    TRK = np.zeros((max(len(tracks), 1), ncell, 3), np.float32)
    TCEN = np.zeros((max(len(tracks), 1), 2), np.float64)
    for ti, groups in enumerate(tracks):
        dx, dy, on, th = np.zeros(ncell), np.zeros(ncell), np.ones(ncell, bool), np.zeros(ncell)
        pivot = None
        for g in groups:
            for t0, m in per["r"].get(g, ()):
                c = (centers or {}).get(m[8])
                if c is None or (pivot is not None and abs(c[0] - pivot[0]) + abs(c[1] - pivot[1]) > 1.0):
                    continue                                      # one pivot per track (a second, different one is dropped)
                pivot = c
                dur = m[5] * 240.0
                f = (ticks >= t0).astype(np.float64) if dur <= 0 else (ticks - t0) / dur
                th += np.radians(m[4]) * _ease(f, m[6], m[7])
            if pivot is not None:
                TCEN[ti] = pivot
            for t0, m in per_group_moves.get(g, ()):
                dur = m[6] * 240.0
                f = (ticks >= t0).astype(np.float64) if dur <= 0 else (ticks - t0) / dur
                e = _ease(f, m[7], m[8])
                dx += m[4] * e
                dy += m[5] * e
            tg = sorted(per_group_tog.get(g, ()), key=lambda v: v[0])
            if tg:
                state = np.ones(ncell, bool)
                for t0, m in tg:
                    state = np.where(ticks >= t0, m[4], state)
                on &= state
        TRK[ti, :, 0] = dx
        TRK[ti, :, 1] = np.where(on, dy, 1e7)
        TRK[ti, :, 2] = th
    return TRK, np.array([x_lo, res, ncell], np.float64), TCEN


def parse_level(text):
    return gdlevel.parse(text, PARSE_KEYS)


def _rot_cw(a, b, quarter):
    """Rotate vector (a, b) clockwise by `quarter` x 90 degrees (GD rotation is clockwise)."""
    for _ in range(quarter % 4):
        a, b = b, -a
    return a, b


# ----------------------------------------------------------------- level -> world arrays
def build_world(hdr, objs, aux, gd):
    """Turn a parsed level into flat arrays + the event list. `gd` is the gdlevel module (ID tables)."""
    CATS, H_, S_, O_, P_, X_ = gd.CAT, gd.H_, gd.S_, gd.O_, gd.P_, gd.X_
    hz_sh, so_sh = SHAPES["hazard"], SHAPES["solid"]

    def cat(oid):
        c = CATS.get(oid)
        if not GAME or c in (O_, P_, X_):
            return c
        k = str(oid)
        return S_ if k in so_sh else H_ if k in hz_sh else None      # the game calls everything else decoration
    play = [o for o in objs if cat(o[0]) in (H_, S_, O_, P_, X_)]
    ends = [x for oid, x, y, _ in objs if oid == gd.END_TRIGGER]
    x_end = min(ends) if ends else max((o[1] for o in play), default=300.0)
    x_end = max(x_end, 300.0)

    # groups moved or toggled by triggers follow those triggers (a track per combination of such groups); groups driven by
    # something unpredictable (rotate, follow, lock to player) are left out, as is anything an orb or pad sits in
    moves, toggles, rots, spawns, bad = trigger_groups(objs, aux, gd)
    dynset = ({m[3] for m in moves} | {t[3] for t in toggles} | {r[3] for r in rots}) - bad
    want = {r[8] for r in rots}
    acc = {}
    for i, a in aux.items():                                          # pivot = mean position of the center group's objects
        if "57" in a and want:
            for g in want.intersection(a["57"].split(".")):
                v = acc.setdefault(g, [0.0, 0.0, 0])
                v[0] += objs[i][1]; v[1] += objs[i][2]; v[2] += 1
    centers = {g: (v[0] / v[2], v[1] / v[2]) for g, v in acc.items()}
    Tt = dict(x0=[], x1=[], cx=[], cy=[], hw=[], hh=[], ang=[], dy=[], trk=[])
    tracks, track_of = [], {}
    S = dict(x0=[], x1=[], y0=[], y1=[], kind=[], dir=[], id=[], trk=[])
    Hh = dict(kind=[], x0=[], x1=[], y0=[], y1=[], cx=[], cy=[], r=[], id=[], trk=[])
    Oo = dict(t=[], x=[], y=[], r=[])
    Pp = dict(t=[], x0=[], x1=[], y0=[], y1=[])
    events, unsupported = [], 0
    for i, (oid, x, y, sc) in enumerate(objs):
        if oid in TELEPORT and x <= x_end + 120:
            a = aux.get(i) or {}
            d54 = _fnum(a, "54")
            gs = a["57"].split(".") if "57" in a else ()
            if d54 and not (gs and bad and not bad.isdisjoint(gs)):
                dg = tuple(sorted(dynset.intersection(gs))) if gs and dynset else ()
                if dg and dg not in track_of:                         # a portal in a moved / toggled group follows it
                    track_of[dg] = len(tracks); tracks.append(dg)
                Tt["trk"].append(track_of[dg] if dg else -1)
                bw, bh, bo = TELE_BOX
                th = math.radians(_fnum(a, "6"))
                ct, st = abs(math.cos(th)), abs(math.sin(th))
                hw2, hh2 = (bw * ct + bh * st) / 2, (bw * st + bh * ct) / 2          # box of the turned trigger box
                ox, oy = bo * math.cos(th), -bo * math.sin(th)                        # its offset turns clockwise with it
                Tt["x0"].append(x + ox - hw2); Tt["x1"].append(x + ox + hw2)             # column index and quick reject
                Tt["cx"].append(x + ox); Tt["cy"].append(y + oy); Tt["hw"].append(bw / 2); Tt["hh"].append(bh / 2)
                Tt["ang"].append(th); Tt["dy"].append(d54)
            continue
        c = cat(oid)
        if c is None or x > x_end + 120:
            continue
        a = aux.get(i)
        trk = -1
        if a is not None and c in (S_, H_, O_, P_):
            if any(a.get(k) == "1" for k in NOCOLL):
                continue                                              # "No Touch": no hitbox
            gs = a["57"].split(".") if "57" in a else ()
            if gs and bad and not bad.isdisjoint(gs):
                continue                                              # driven by something we cannot predict
            dg = tuple(sorted(dynset.intersection(gs))) if gs and dynset else ()
            if dg:
                if c in (O_, P_):
                    continue
                if dg not in track_of:
                    track_of[dg] = len(tracks); tracks.append(dg)
                trk = track_of[dg]
        rot = float(a["6"]) if a and "6" in a else 0.0
        try:
            sx = float(a["128"]) if a and "128" in a else sc
            sy = float(a["129"]) if a and "129" in a else sc
        except ValueError:
            sx = sy = sc
        sm = math.sqrt(abs(sx * sy))
        fx, fy = (a is not None and a.get("4") == "1"), (a is not None and a.get("5") == "1")
        q = int(round(rot / 90.0))
        exact = abs(rot - 90.0 * q) < 1.0
        if c == S_:
            kind, w, h = so_sh.get(str(oid), (0, 30.0, 30.0))
            w, h = w * sx, h * sy
            if exact and q % 2:
                w, h = h, w
            elif not exact:
                th = math.radians(rot)
                w, h = abs(w * math.cos(th)) + abs(h * math.sin(th)), abs(w * math.sin(th)) + abs(h * math.cos(th))
            d = 0
            if kind == 1:
                corner = 0                                            # base sprite: right angle at bottom-right
                if fx: corner = {0: 1, 1: 0, 2: 3, 3: 2}[corner]
                if fy: corner = {0: 3, 3: 0, 1: 2, 2: 1}[corner]
                d = (corner + q) % 4
            S["x0"].append(x - w / 2); S["x1"].append(x + w / 2); S["y0"].append(y - h / 2); S["y1"].append(y + h / 2)
            S["kind"].append(kind); S["dir"].append(d); S["id"].append(oid); S["trk"].append(trk)
        elif c == H_:
            k, a1, b1, ox, oy = hz_sh.get(str(oid), (0, 6, 12, 0, -1.5))
            fh = FLAT.get(str(oid))
            if fh:
                k, a1, b1, ox, oy = 0, PIT_A * 30.0, PIT_B * fh, 0.0, PIT_OY * fh
            ox, oy = ox * sx, oy * sy
            if fy:
                oy = -oy
                q += 2
            ox, oy = _rot_cw(ox, oy, q) if exact else (ox, oy)
            cx, cy = x + ox, y + oy
            if k == 1:
                r = a1 * sm
                Hh["kind"].append(1); Hh["cx"].append(cx); Hh["cy"].append(cy); Hh["r"].append(r)
                Hh["x0"].append(cx - r); Hh["x1"].append(cx + r); Hh["y0"].append(cy - r); Hh["y1"].append(cy + r); Hh["id"].append(oid)
                Hh["trk"].append(trk)
            else:
                w, h = a1 * sx, b1 * sy
                if exact and q % 2:
                    w, h = h, w
                elif not exact:
                    th = math.radians(rot)
                    w, h = abs(w * math.cos(th)) + abs(h * math.sin(th)), abs(w * math.sin(th)) + abs(h * math.cos(th))
                Hh["kind"].append(0); Hh["cx"].append(cx); Hh["cy"].append(cy); Hh["r"].append(0.0)
                Hh["x0"].append(cx - w / 2); Hh["x1"].append(cx + w / 2); Hh["y0"].append(cy - h / 2); Hh["y1"].append(cy + h / 2); Hh["id"].append(oid)
                Hh["trk"].append(trk)
        elif c == O_:
            t = ORB_TYPE.get(oid)
            if t:
                Oo["t"].append(t); Oo["x"].append(x); Oo["y"].append(y)
                Oo["r"].append(SHAPES.get("orb", {}).get(str(oid), (2 * RING_R,))[0] / 2 * sm)     # half size of the square trigger box
            else:
                unsupported += 1
        elif c == P_:
            t = PAD_TYPE.get(oid)
            if t:
                pw, ph = SHAPES.get("pad", {}).get(str(oid), (30.0, 10.0))      # the game's trigger box (yellow pad 25 x 4)
                hx, hy = (ph / 2 * sy, pw / 2 * sx) if q % 2 else (pw / 2 * sx, ph / 2 * sy)
                Pp["t"].append(t); Pp["x0"].append(x - hx); Pp["x1"].append(x + hx)
                Pp["y0"].append(y - hy); Pp["y1"].append(y + hy)
            else:
                unsupported += 1
        elif c == X_ and x <= x_end and x + SHAPES.get("portal", {}).get(str(oid), (0.0, 0.0))[0] / 2.0 > -15.0:   # left of x = 0 still counts if the starting box reaches it
            pw, ph = SHAPES.get("portal", {}).get(str(oid), (0.0, 0.0))
            x = x - 15.0 - (ph if (exact and q % 2) else pw) / 2.0    # it fires when the player's box first touches it
            if oid in gd.MODE_PORTAL: events.append((x, EV_MODE, gd.MODE_PORTAL[oid], y))
            elif oid in gd.SIZE_PORTAL: events.append((x, EV_SIZE, gd.SIZE_PORTAL[oid], y))
            elif oid in gd.GRAV_PORTAL: events.append((x, EV_GRAV, {0: 1, 1: -1, 2: 0}[gd.GRAV_PORTAL[oid]], y))
            elif oid in gd.SPEED_PORTAL: events.append((x, EV_SPEED, gd.SPEED_PORTAL[oid], y))
            elif oid in gd.DUAL_PORTAL: events.append((x, EV_DUAL, gd.DUAL_PORTAL[oid], y))
    events.sort()
    f = lambda d, k: np.array(d[k], dtype=np.float64)
    xs = [v for v in (S["x0"], Hh["x0"], Oo["x"], Pp["x0"], Tt["x0"]) if len(v)]
    col0 = int(math.floor(min((min(v) for v in xs), default=0.0) / 30.0)) - 2
    ncols = int(math.ceil((x_end + 400) / 30.0)) - col0 + 2
    sx0, sx1, sy0, sy1 = (f(S, k) for k in ("x0", "x1", "y0", "y1"))
    hx0, hx1, hy0, hy1 = (f(Hh, k) for k in ("x0", "x1", "y0", "y1"))
    ox_, oy_ = f(Oo, "x"), f(Oo, "y")
    px0, px1 = f(Pp, "x0"), f(Pp, "x1")
    tops = [v for v in (S["y1"], Hh["y1"], Oo["y"], Pp["y1"], [e[3] for e in events]) if len(v)]
    top = max((max(v) for v in tops), default=0.0)
    start = dict(mode=int(hdr.get("kA2", 0) or 0) % 8, mini=hdr.get("kA3") == "1", speed=min(max(int(hdr.get("kA4", 0) or 0), 0), 4),
                 up=-1 if hdr.get("kA11") == "1" else 1, dual=hdr.get("kA8") == "1")
    strk, htrk = np.array(S["trk"], np.int64), np.array(Hh["trk"], np.int64)
    if tracks:
        sch0 = build_schedule(dict(start=start, events=events, x_end=x_end))
        TRK, DP, TCEN = dyn_tracks(tracks, moves, toggles, spawns, sch0["x"], -60.0, x_end + 120.0, rots, centers)

        def sweep(x0, x1, y0, y1, trk):                           # moving objects are indexed over every column they visit
            a0, a1 = x0.copy(), x1.copy()
            for j in np.where(trk >= 0)[0]:
                k = trk[j]
                th = TRK[k, :, 2].astype(np.float64)
                ox = TRK[k, :, 0].astype(np.float64)
                if np.any(th != 0.0):
                    mx, my = (x0[j] + x1[j]) / 2, (y0[j] + y1[j]) / 2
                    cx, cy = TCEN[k]
                    ox = ox + cx + (mx - cx) * np.cos(th) + (my - cy) * np.sin(th) - mx
                a0[j] += ox.min(); a1[j] += ox.max()
            return a0, a1
        csr_s0, csr_s1 = sweep(sx0, sx1, sy0, sy1, strk)
        csr_h0, csr_h1 = sweep(hx0, hx1, hy0, hy1, htrk)
        csr_t0, csr_t1 = sweep(f(Tt, "x0"), f(Tt, "x1"), f(Tt, "cy"), f(Tt, "cy"), np.array(Tt["trk"], np.int64))
    else:
        TRK, DP, TCEN = np.zeros((1, 1, 3), np.float32), np.array([0.0, 1.0, 1.0]), np.zeros((1, 2))
        csr_t0, csr_t1 = f(Tt, "x0"), f(Tt, "x1")
        csr_s0, csr_s1, csr_h0, csr_h1 = sx0, sx1, hx0, hx1
    scp, sci = build_csr(csr_s0, csr_s1, col0, ncols)
    hcp, hci = build_csr(csr_h0, csr_h1, col0, ncols)
    ocp, oci = build_csr(ox_ - 40, ox_ + 40, col0, ncols)
    pcp, pci = build_csr(px0, px1, col0, ncols)
    GT = (sx0, sx1, sy0, sy1, np.array(S["kind"], np.int64), np.array(S["dir"], np.int64), scp, sci,
          np.array(Hh["kind"], np.int64), hx0, hx1, hy0, hy1, f(Hh, "cx"), f(Hh, "cy"), f(Hh, "r"), hcp, hci,
          np.array(Oo["t"], np.int64), ox_, oy_, f(Oo, "r"), ocp, oci,
          np.array(Pp["t"], np.int64), px0, px1, f(Pp, "y0"), f(Pp, "y1"), pcp, pci, np.array([HZ_K, max(YMAX, top + 600.0), PAD_K, RING_K, TS]),
          strk, htrk, TRK, DP, TCEN,
          f(Tt, "x0"), f(Tt, "x1"), f(Tt, "cx"), f(Tt, "cy"), f(Tt, "hw"), f(Tt, "hh"), f(Tt, "ang"), f(Tt, "dy"),
          *build_csr(csr_t0, csr_t1, col0, ncols), np.array(Tt["trk"], np.int64))
    return dict(GT=GT, col0=col0, ncols=ncols, x_end=x_end, events=events, start=start, unsupported=unsupported,
                sid=np.array(S["id"]), hid=np.array(Hh["id"]),
                counts=dict(solid=len(sx0), hazard=len(hx0), orb=len(ox_), pad=len(px0), moving=int((strk >= 0).sum() + (htrk >= 0).sum()),
                            tracks=len(tracks), rotating=int(sum(1 for k in range(len(tracks)) if np.any(TRK[k, :, 2] != 0))),
                            teleports=len(Tt["dy"])))


@njit(cache=True)
def build_csr(x0, x1, col0, ncols):
    n = len(x0)
    cnt = np.zeros(ncols + 1, np.int64)
    for i in range(n):
        a = int(math.floor(x0[i] / 30.0)) - col0
        b = int(math.floor(x1[i] / 30.0)) - col0
        a, b = max(a, 0), min(b, ncols - 1)
        for c in range(a, b + 1):
            cnt[c + 1] += 1
    for c in range(ncols):
        cnt[c + 1] += cnt[c]
    idx = np.empty(cnt[ncols], np.int64)
    fill = cnt[:-1].copy()
    for i in range(n):
        a = int(math.floor(x0[i] / 30.0)) - col0
        b = int(math.floor(x1[i] / 30.0)) - col0
        a, b = max(a, 0), min(b, ncols - 1)
        for c in range(a, b + 1):
            idx[fill[c]] = i
            fill[c] += 1
    return cnt, idx


def build_schedule(world):
    """Global per-tick timeline shared by every candidate: x, speed, mode, size, floor/ceiling, event pointers."""
    st, events, x_end = world["start"], world["events"], world["x_end"]
    mode, mini, spd = st["mode"], st["mini"], st["speed"]
    floor, ceil = 0.0, (1e9 if mode in (CUBE, ROBOT) else 300.0)
    xs, spds, modes, minis, floors, ceils = [0.0], [], [], [], [], []
    ev_ptr, ev_type, ev_val, ev_y = [], [], [], []
    ei, x, tick = 0, 0.0, 0
    while x < x_end and tick < 160000:
        ev_ptr.append(len(ev_type))
        while ei < len(events) and events[ei][0] <= x:
            ex, et, val, ey = events[ei]
            ei += 1
            if et == EV_MODE:
                prev_mode, mode = mode, val
                val = mode + 8 * prev_mode
                if mode in (CUBE, ROBOT):
                    if CUBE_CEIL == 0:
                        ceil = 1e9
                    if CUBE_RESET:
                        floor = 0.0
                else:
                    span = 270.0 if mode == SPIDER else (240.0 if mode == BALL else 300.0)    # ball: 240 (seen in the game)
                    floor = max(0.0, 30.0 * math.floor((ey - span / 2.0 - 15.0) / 30.0))
                    ceil = floor + span
            elif et == EV_SIZE: mini = bool(val)
            elif et == EV_SPEED: spd = val
            ev_type.append(et); ev_val.append(val); ev_y.append(ey)
        spds.append(spd); modes.append(mode); minis.append(mini); floors.append(floor); ceils.append(ceil)
        x += VX[spd] * DT
        xs.append(x)
        tick += 1
    ev_ptr.append(len(ev_type))
    return dict(T=tick, x=np.array(xs), spd=np.array(spds, np.int64), mode=np.array(modes, np.int64), mini=np.array(minis, np.int64),
                floor=np.array(floors), ceil=np.array(ceils), ev_ptr=np.array(ev_ptr, np.int64), ev_type=np.array(ev_type, np.int64),
                ev_val=np.array(ev_val, np.int64), ev_y=np.array(ev_y))


def start_state(world, sch):
    st = world["start"]
    p = np.zeros(NS)
    p[UP] = st["up"]
    p[PAD] = -1.0
    p[TEL] = -1.0
    p[PAGE] = 99.0
    mode = int(sch["mode"][0]) if sch["T"] else st["mode"]
    hw = 15.0 * (0.6 if st["mini"] else 1.0)
    if mode in (CUBE, ROBOT, BALL, SPIDER):
        p[Y] = (hw if st["up"] > 0 else sch["ceil"][0] - hw) if sch["ceil"][0] < 1e8 or st["up"] > 0 else hw
        p[GND] = 2.0                                      # standing still at the start
    else:
        p[Y] = sch["floor"][0] + 150.0
    return p


# ----------------------------------------------------------------- physics core (one 240 Hz tick of one player)
# All vertical maths is done in *relative* coordinates (gravity pulls toward -y), so inverted gravity needs no second code path.
@njit(cache=True)
def _doff(GT, k, x, mx, my):
    """Offset (dx, dy) of an object centred at (mx, my) on track k when the player is at x (0, 0 for static objects):
    the track's move plus, when it rotates, the swing of (mx, my) around the track's pivot (clockwise, as in GD)."""
    if k < 0:
        return 0.0, 0.0
    D = GT[35]
    c = int((x - D[0]) / D[1])
    c = min(max(c, 0), int(D[2]) - 1)
    dx, dy, th = float(GT[34][k, c, 0]), float(GT[34][k, c, 1]), float(GT[34][k, c, 2])
    if th != 0.0:
        cx, cy = GT[36][k, 0], GT[36][k, 1]
        rx, ry = mx - cx, my - cy
        ct, st = math.cos(th), math.sin(th)
        dx += cx + rx * ct + ry * st - mx
        dy += cy - rx * st + ry * ct - my
    return dx, dy


@njit(cache=True)
def _box_obb(px, py, h, cx, cy, bw, bh, th):
    """Does the player's square box (centre px, py, half size h) touch a box of half size bw x bh turned clockwise by th
    around (cx, cy)? Separating-axis test (the game tests turned trigger boxes as turned boxes)."""
    ux, uy = math.cos(th), -math.sin(th)
    vx, vy = math.sin(th), math.cos(th)
    dx, dy = cx - px, cy - py
    for k in range(4):
        if k == 0:
            ax, ay = 1.0, 0.0
        elif k == 1:
            ax, ay = 0.0, 1.0
        elif k == 2:
            ax, ay = ux, uy
        else:
            ax, ay = vx, vy
        rp = h * (abs(ax) + abs(ay))
        rq = bw * abs(ux * ax + uy * ay) + bh * abs(vx * ax + vy * ay)
        if abs(dx * ax + dy * ay) > rp + rq:
            return False
    return True


@njit(cache=True)
def _overlap(a0, a1, b0, b1):
    return a1 > b0 and a0 < b1


@njit(cache=True)
def solid_lift(p, GT, col0, ncols, x, yr, hw, up, snap, rising):
    """Highest surface (relative y) the player can step up onto among overlapping solids, or very negative.
    Rectangles give their top; floor-type slopes give the ramp height under the player's leading corner."""
    sx0, sx1, sy0, sy1, skind, sdir, scp, sci = GT[0], GT[1], GT[2], GT[3], GT[4], GT[5], GT[6], GT[7]
    c0 = max(int(math.floor((x - hw) / 30.0)) - col0, 0)
    c1 = min(int(math.floor((x + hw) / 30.0)) - col0, ncols - 1)
    best = -1e18
    for c in range(c0, c1 + 1):
        for q in range(scp[c], scp[c + 1]):
            j = sci[q]
            ddx, ddy = _doff(GT, GT[32][j], x, 0.5 * (sx0[j] + sx1[j]), 0.5 * (sy0[j] + sy1[j]))
            a0, a1, b0, b1 = sx0[j] + ddx, sx1[j] + ddx, sy0[j] + ddy, sy1[j] + ddy
            if not _overlap(x - hw, x + hw, a0, a1):
                continue
            if up > 0:
                r0, r1 = b0, b1
            else:
                r0, r1 = -b1, -b0
            if not (r1 > yr - hw and r0 < yr + hw):
                continue
            if skind[j] == 0:
                if rising:                                     # the game only snaps onto a block top when not moving up
                    continue
                lift = r1 - (yr - hw)
                if 0.0 < lift <= snap:
                    best = max(best, r1)
            else:
                d = sdir[j] if up > 0 else (3 - sdir[j])
                if d > 1:
                    continue                                   # ceiling-type slope: handled by the inner-box death test
                w = a1 - a0
                h = r1 - r0
                xc = x + hw if d == 0 else x - hw
                fr = min(max((xc - a0) / w, 0.0), 1.0)
                surf = (r0 + fr * h) if d == 0 else (r1 - fr * h)
                lift = surf - (yr - hw)
                if 0.0 < lift <= max(snap, 12.0):
                    best = max(best, surf)
    return best


@njit(cache=True)
def tri_hit(x0, x1, y0, y1, d, px, py, h):
    """Box (centre px,py half-size h) vs right triangle filling the bounding box with its right angle at corner d."""
    w, hh = x1 - x0, y1 - y0
    cx = px + h if (d == 0 or d == 3) else px - h
    cy = py - h if (d == 0 or d == 1) else py + h
    u = (cx - x0) / w if w > 0 else 0.0
    v = (cy - y0) / hh if hh > 0 else 0.0
    if d == 0:
        return v < u
    if d == 1:
        return v < 1.0 - u
    if d == 2:
        return v > u
    return v > 1.0 - u


@njit(cache=True)
def inner_hits_solid(GT, col0, ncols, x, yw, ih):
    """Does the small 'solid' hitbox (world coordinates) touch any solid block?"""
    sx0, sx1, sy0, sy1, skind, sdir, scp, sci = GT[0], GT[1], GT[2], GT[3], GT[4], GT[5], GT[6], GT[7]
    c0 = max(int(math.floor((x - ih) / 30.0)) - col0, 0)
    c1 = min(int(math.floor((x + ih) / 30.0)) - col0, ncols - 1)
    for c in range(c0, c1 + 1):
        for q in range(scp[c], scp[c + 1]):
            j = sci[q]
            ddx, ddy = _doff(GT, GT[32][j], x, 0.5 * (sx0[j] + sx1[j]), 0.5 * (sy0[j] + sy1[j]))
            a0, a1, b0, b1 = sx0[j] + ddx, sx1[j] + ddx, sy0[j] + ddy, sy1[j] + ddy
            if _overlap(x - ih, x + ih, a0, a1) and _overlap(yw - ih, yw + ih, b0, b1):
                if skind[j] == 0:
                    return True
                if tri_hit(a0, a1, b0, b1, sdir[j], x, yw, ih):
                    return True
    return False


@njit(cache=True)
def hazard_hit(GT, col0, ncols, x, y, hw, eps):
    hkind, hx0, hx1, hy0, hy1, hcx, hcy, hr, hcp, hci = GT[8], GT[9], GT[10], GT[11], GT[12], GT[13], GT[14], GT[15], GT[16], GT[17]
    c0 = max(int(math.floor((x - hw - eps) / 30.0)) - col0, 0)
    c1 = min(int(math.floor((x + hw + eps) / 30.0)) - col0, ncols - 1)
    for c in range(c0, c1 + 1):
        for q in range(hcp[c], hcp[c + 1]):
            j = hci[q]
            ddx, ddy = _doff(GT, GT[33][j], x, hcx[j], hcy[j])
            if hkind[j] == 0:
                if _overlap(x - hw, x + hw, hx0[j] + ddx - eps, hx1[j] + ddx + eps) and _overlap(y - hw, y + hw, hy0[j] + ddy - eps, hy1[j] + ddy + eps):
                    return True
            else:
                dx = max(abs(hcx[j] + ddx - x) - hw, 0.0)
                dy = max(abs(hcy[j] + ddy - y) - hw, 0.0)
                if dx * dx + dy * dy <= (hr[j] + eps) * (hr[j] + eps):
                    return True
    return False


@njit(cache=True)
def apply_event(p, et, val):
    if et == 0:                                           # entering a gamemode (val = new + 8 x previous): clear latches, scale velocity
        new, old = val % 8, val // 8
        p[GND] = 0.0; p[BOOST] = 0.0; p[ACC] = 0.0; p[RACC] = 0.0; p[SLP] = 0.0
        if new != old:
            f = 1.0
            if new == 0 or new == 5:
                f = 1.0 if old == 2 else (0.25 if old == 4 else 0.5)
            elif new == 1:
                f = 0.25 if (old == 3 or old == 4) else 0.5
            elif new == 2:
                f = 0.5 if (old == 1 or old == 3) else 1.0
            elif new == 3:
                f = 0.25 if old == 1 else 0.5
            p[VY] *= f
    elif et == 2:                                         # gravity portal: an actual flip halves the velocity
        nu = float(val) if val != 0 else -p[UP]
        if nu != p[UP]:
            p[VY] *= 0.5
        p[UP] = nu
        p[GND] = 0.0


@njit(cache=True)
def _r3(v):
    """GD rounds every velocity change to 1/1000 (seen in the game's logs: gravity 0.21555 per tick arrives as 0.216)."""
    return math.floor(v * 1000.0 + 0.5) / 1000.0


@njit(cache=True)
def step_tick(p, GT, col0, ncols, xn, mode, mini, spd, flo, cei, hold, press, eps):
    """Advance one player by one 240 Hz tick. Returns 0 if alive, else a death code (1 hazard, 2 solid, 3 bounds)."""
    up = p[UP]
    yr = up * p[Y]
    vr = up * p[VY]
    small = 0.6 if mini else 1.0
    hw = 15.0 * small
    if mode == 4:
        hw = 5.0 * small
    elif mode == 6:
        hw = 13.5 * small                                 # the spider rests 13.5 from the floor and ceiling (measured in the game)
    ih = hw if mode == 4 else 4.5 * small
    vs = 0.8 if mini else 1.0                             # v16 in the decompiled jump code
    g = GRAV[spd]
    dtv = DT * GT[31][4]                                  # vertical time step (the 0.9 factor)
    gf = G_FLY
    ground = p[GND] > 0.5                                 # GND: 0 airborne, 1 landed this tick, 2 standing for longer
    settled = p[GND] > 1.5
    # ---- input latches
    if press:
        p[BUF] = 1.0; p[PAGE] = 0.0; p[RING] = 0.0
    elif hold == 0:
        p[BUF] = 0.0; p[PAGE] = 99.0; p[RING] = 0.0
    else:
        p[PAGE] += 1.0
    buf = p[BUF] > 0.5
    # ---- an orb already in reach when clicking fires before the move (launch, gravity, move); one reached during the
    # move fires at the end of the tick, like pads (both orders seen in the game)
    if buf and p[RING] < 0.5:
        if pad_orb_effects(p, GT, col0, ncols, xn - VX[spd] * DT, p[Y], hw, mode, mini, spd, False, True):
            up = p[UP]; yr = up * p[Y]; vr = up * p[VY]
            ground = False; settled = False
            buf = p[BUF] > 0.5
    jumped = False
    flo_r = flo if up > 0 else -cei
    cei_r = cei if up > 0 else -flo
    # ---- per-gamemode velocity update (PlayerObject::updateJump)
    if mode == 0:                                         # cube
        if ground and hold == 1:
            vr = YSTART[spd] * vs; jumped = True; p[BOOST] = 1.0
        if settled or not jumped:                         # a jump from standing still also takes this tick's gravity, one taken
            vr -= _r3(g * dtv)                            # on the tick of landing does not (both measured in the game)
        vr = max(vr, -15.0)
    elif mode == 5:                                       # robot: short hop, then a held rise that cancels gravity for 15 frames
        if ground and buf:
            vr = 0.5 * YSTART[spd] * vs; jumped = True; p[BOOST] = 1.0; p[RACC] = 0.0; p[BUF] = 0.0; buf = False
        if not jumped:
            if hold == 0:
                p[BOOST] = 0.0                                # once released, the rise cannot resume until the next jump
            if p[BOOST] > 0.5 and hold == 1 and p[RACC] + dtv / 10.0 <= 1.5:      # 66 boosted ticks at most, as in the game
                p[RACC] += dtv / 10.0
            else:
                vr -= _r3(g * 0.9 * dtv)
            vr = max(vr, -15.0)
    elif mode == 2:                                       # ball: gravity 0.6x, a click on a surface flips gravity
        if ground and buf:
            p[UP] = -up; up = -up
            yr = -yr; flo_r, cei_r = -cei_r, -flo_r
            vr = -0.3 * YSTART[spd] * vs; jumped = True; p[BUF] = 0.0; buf = False
        if not jumped:
            vr -= _r3(gf * 0.6 * dtv)
            vr = max(vr, -15.0)
    elif mode == 6:                                       # spider: click teleports to the far surface and flips gravity
        if ground and buf:
            best = cei_r - hw if cei_r < 1e8 else 1e18
            c0 = max(int(math.floor((xn - hw) / 30.0)) - col0, 0)
            c1 = min(int(math.floor((xn + hw) / 30.0)) - col0, ncols - 1)
            for c in range(c0, c1 + 1):
                for q in range(GT[6][c], GT[6][c + 1]):
                    j = GT[7][q]
                    ddx, ddy = _doff(GT, GT[32][j], xn, 0.5 * (GT[0][j] + GT[1][j]), 0.5 * (GT[2][j] + GT[3][j]))
                    if not _overlap(xn - hw, xn + hw, GT[0][j] + ddx, GT[1][j] + ddx) or GT[4][j] != 0:
                        continue
                    r0 = (GT[2][j] + ddy) if up > 0 else -(GT[3][j] + ddy)
                    if r0 >= yr + hw and r0 - hw < best:
                        best = r0 - hw
            if best < 1e17 and best - yr < 520.0:
                p[UP] = -up; up = -up
                yr = -best; flo_r, cei_r = -cei_r, -flo_r
                vr = 0.0; jumped = True; p[BUF] = 0.0; buf = False; p[GND] = 1.0
        if not jumped:
            vr -= _r3(gf * 0.6 * dtv)
            vr = max(vr, -15.0)
    elif mode == 1 or mode == 3:                          # ship / ufo (flying vehicles share caps)
        v16 = 0.85 if mini else 1.0
        vtop = 8.0 / v16
        vbot = -6.4 / v16
        falling = vr < 0.0
        thr = VTHR[spd]
        if mode == 1:
            if hold == 1:
                a = (0.5 if vr <= thr else 0.4) * gf / v16
                if p[ACC] > 0.5 and vr >= 0.0:
                    a = -0.8 * 0.4 * gf / v16
            else:
                a = -(1.2 * 0.4 if vr >= thr else 0.8 * 0.4) * gf / v16
            vr += _r3(a * dtv)
        else:
            if press:                                     # a ufo click sets the launch speed; no gravity on that tick
                jv = 6.648 if mini else 6.871
                if jv > vr:
                    vr = jv
            else:
                vr -= _r3(gf * (1.2 if vr > thr else 0.8) * 0.5 / v16 * dtv)
        if p[ACC] > 0.5:
            if vbot <= vr <= vtop:
                p[ACC] = 0.0
        else:
            vr = min(max(vr, vbot), vtop)
    elif mode == 4:                                       # wave: constant slope; mini is twice as steep
        sp = VX[spd] * (2.0 if mini else 1.0)
        vr = (sp if hold == 1 else -sp) * DT / dtv            # 45 degrees (x2 mini) whatever the vertical time factor
    else:                                                 # swing: click flips gravity and keeps 0.8 of the world velocity
        if press:
            p[UP] = -up; up = -up; vr = -0.8 * vr
            yr = -yr; flo_r, cei_r = -cei_r, -flo_r
        vr -= _r3(gf * (0.6 if mini else 0.4) * dtv)
        vr = min(max(vr, -8.0), 8.0)
    # ---- horizontal move: step up onto blocks and ramps
    snap = 0.0 if mode == 4 else (6.0 if (mode == 1 or mode == 3 or mode == 7) else 10.0)
    landed = False
    on_slope = False
    if snap > 0.0:
        top = solid_lift(p, GT, col0, ncols, xn, yr, hw, up, snap, vr > 0.0)
        if top > -1e17:
            if top - (yr - hw) > 0.0:
                on_slope = top - (yr - hw) < 30.0
            yr = top + hw
            if vr < 0.0:
                vr = 0.0
            landed = True
    # ---- vertical move
    yn = yr + vr * dtv
    sx0, sx1, sy0, sy1, skind, scp, sci = GT[0], GT[1], GT[2], GT[3], GT[4], GT[6], GT[7]
    c0 = max(int(math.floor((xn - hw) / 30.0)) - col0, 0)
    c1 = min(int(math.floor((xn + hw) / 30.0)) - col0, ncols - 1)
    if vr <= 0.0:
        bx = hw if mode == 4 else 0.0                     # the wave stops a full size (10) from the floor and ceiling
        best = (flo_r + bx) if (yn - hw - bx) < flo_r else -1e18
        for c in range(c0, c1 + 1):
            for q in range(scp[c], scp[c + 1]):
                j = sci[q]
                ddx, ddy = _doff(GT, GT[32][j], xn, 0.5 * (sx0[j] + sx1[j]), 0.5 * (sy0[j] + sy1[j]))
                if skind[j] != 0 or not _overlap(xn - hw, xn + hw, sx0[j] + ddx, sx1[j] + ddx):
                    continue
                r1 = (sy1[j] + ddy) if up > 0 else -(sy0[j] + ddy)
                if (yr - hw) >= r1 - 0.01 and (yn - hw) < r1:
                    best = max(best, r1)
        if best > -1e17:
            yn = best + hw
            if vr < 0.0 and mode != 4:
                vr = 0.0
            landed = True
    else:
        bx = hw if mode == 4 else 0.0
        best = (cei_r - bx) if (yn + hw + bx) > cei_r else 1e18
        for c in range(c0, c1 + 1):
            for q in range(scp[c], scp[c + 1]):
                j = sci[q]
                ddx, ddy = _doff(GT, GT[32][j], xn, 0.5 * (sx0[j] + sx1[j]), 0.5 * (sy0[j] + sy1[j]))
                if skind[j] != 0 or not _overlap(xn - hw, xn + hw, sx0[j] + ddx, sx1[j] + ddx):
                    continue
                r0 = (sy0[j] + ddy) if up > 0 else -(sy1[j] + ddy)
                if (yr + hw) <= r0 + 0.01 and (yn + hw) > r0:
                    best = min(best, r0)
        if best < 1e17:
            yn = best - hw
            if mode != 4:
                vr = 0.0
    # leaving a ramp's top edge launches the player along the slope
    if (not landed) and p[SLP] > 0.0 and p[GND] > 0.5:
        vr = max(vr, p[SLP])
    p[SLP] = (VX[spd] * 0.5 if on_slope else 0.0)
    p[GND] = (2.0 if ground else 1.0) if landed else 0.0
    p[Y] = up * yn
    p[VY] = up * vr
    yw = p[Y]
    inv = p[INV]
    if inv > 0.0:
        p[INV] = inv - 1.0
    # ---- pads and orbs
    pad_orb_effects(p, GT, col0, ncols, xn, yw, hw, mode, mini, spd, True, True)   # an orb reached during the move fires now
    yw = p[Y]
    # ---- teleport portals: touching the entrance moves the player up by the exit offset; velocity is kept (seen in the game)
    tx0, tx1, tcx, tcy, thw, thh, tang, tdy, tcp, tci = GT[37], GT[38], GT[39], GT[40], GT[41], GT[42], GT[43], GT[44], GT[45], GT[46]
    if len(tdy):
        touched = -1
        c0 = max(int(math.floor((xn - hw) / 30.0)) - col0, 0)
        c1 = min(int(math.floor((xn + hw) / 30.0)) - col0, ncols - 1)
        for c in range(c0, c1 + 1):
            for q in range(tcp[c], tcp[c + 1]):
                j = tci[q]
                ddx, ddy = _doff(GT, GT[47][j], xn, tcx[j], tcy[j])
                if _overlap(xn - hw, xn + hw, tx0[j] + ddx, tx1[j] + ddx) and _box_obb(xn, yw, hw, tcx[j] + ddx, tcy[j] + ddy, thw[j], thh[j], tang[j]):
                    touched = j
                    if int(p[TEL]) != j:
                        p[Y] = yw + tdy[j]
                        yw = p[Y]
                        p[GND] = 0.0
                    break
            if touched >= 0:
                break
        p[TEL] = float(touched)
    # ---- death checks
    if yw > GT[31][1] or yw < -300.0:
        if inv > 0.0:                                     # a forced pass keeps the player inside the level
            p[Y] = min(max(yw, -300.0), GT[31][1])
            p[VY] = 0.0
            return 0
        return 3
    if inv <= 0.0:
        if inner_hits_solid(GT, col0, ncols, xn, yw, ih + eps):
            return 2
        if hazard_hit(GT, col0, ncols, xn, yw, hw * GT[31][0], eps):
            return 1
    return 0


@njit(cache=True)
def pad_orb_effects(p, GT, col0, ncols, x, y, hw, mode, mini, spd, do_pads, do_orbs):
    """Pads (do_pads) and orbs (do_orbs) touching the player at (x, y). Returns True when an orb fired."""
    otype, ox, oy, orad, ocp, oci = GT[18], GT[19], GT[20], GT[21], GT[22], GT[23]
    ptype, px0, px1, py0, py1, pcp, pci = GT[24], GT[25], GT[26], GT[27], GT[28], GT[29], GT[30]
    up = p[UP]
    vs = 0.8 if mini else 1.0
    c0 = max(int(math.floor((x - hw) / 30.0)) - col0, 0)
    c1 = min(int(math.floor((x + hw) / 30.0)) - col0, ncols - 1)
    fly = (mode == 1 or mode == 3 or mode == 7)
    g = GROUP[mode]
    touching = -1
    for c in range(c0, c1 + 1 if do_pads else c0):
        for q in range(pcp[c], pcp[c + 1]):
            j = pci[q]
            if _overlap(x - hw, x + hw, px0[j], px1[j]) and _overlap(y - hw, y + hw, py0[j], py1[j]):
                touching = j
                if int(p[PAD]) != j:
                    p[PAD] = float(j)
                    t = ptype[j]
                    if t == 4:                             # blue pad: flip gravity and push toward the new floor
                        p[UP] = -up; up = -up
                        if g >= 0:
                            p[VY] = -up * PAD_V[4, g] * vs * GT[31][2]
                        else:
                            p[VY] = -p[VY]
                        p[GND] = 0.0
                    elif g >= 0:
                        p[VY] = up * PAD_V[t, g] * vs * GT[31][2]
                        if fly:
                            p[ACC] = 1.0
                        p[GND] = 0.0
    if do_pads and touching < 0:
        p[PAD] = -1.0
    fired = False
    if do_orbs and p[BUF] > 0.5 and p[RING] < 0.5 and p[PAGE] <= 6.0:
        best = -1
        bd = 1e18
        for c in range(c0, c1 + 1):
            for q in range(ocp[c], ocp[c + 1]):
                j = oci[q]
                dx = abs(ox[j] - x)
                dy = abs(oy[j] - y)
                d2 = dx * dx + dy * dy
                if dx <= hw + orad[j] and dy <= hw + orad[j] and d2 < bd:      # box overlap, as in the game
                    bd = d2
                    best = j
        if best >= 0:
            t = otype[best]
            fired = True
            if t == 4 or t == 5:                           # blue / green flip gravity (the wave too)
                p[UP] = -up; up = -up
            if g >= 0:
                p[RING] = 1.0
                if t == 6:                                 # black: slam toward the floor
                    p[VY] = -up * BLACK_V * GT[31][3]
                else:
                    v = RING_MULT[t, g] * YSTART[spd] * vs * GT[31][3]
                    p[VY] = (-up * v) if t == 4 else (up * v)   # blue pushes toward the new floor, the others launch away from it
                    if fly:
                        p[ACC] = 1.0
                p[GND] = 0.0
                if mode == 2 or mode == 5 or mode == 6:
                    p[BUF] = 0.0
            elif t == 4 or t == 5:
                p[RING] = 1.0
    return fired


# ----------------------------------------------------------------- planner: dynamic programming over reachable states
TABLE = 1 << 16
SUB = int(os.environ.get("GDSIM_SUB", 1))                # 1: inputs may change on any 240 Hz tick, 0: only on 60 Hz steps


@njit(cache=True)
def make_key(p, mode, qs, h):
    if mode == 1 or mode == 3 or mode == 7:
        dy, dv = 3.0 * qs, 0.7 * qs
    elif mode == 4:
        dy, dv = 1.5 * qs, 1.0
    else:
        dy, dv = 1.0 * qs, 0.25 * qs
    ky = int(math.floor(p[Y] / dy)) + 100000
    kv = int(math.floor(p[VY] / dv)) + 100000
    fl = (1 if p[UP] > 0 else 0) | ((1 if p[GND] > 0.5 else 0) << 1) | ((1 if p[BOOST] > 0.5 else 0) << 2) \
        | ((1 if p[ACC] > 0.5 else 0) << 3) | ((1 if p[RING] > 0.5 else 0) << 4) | ((1 if p[BUF] > 0.5 else 0) << 5) | (h << 6) | ((1 if p[GND] > 1.5 else 0) << 7)
    kr = int(p[RACC] * 10.0)
    return ((ky * 1000003 + kv) * 97 + fl) * 64 + kr


@njit(cache=True)
def near_cost(GT, col0, ncols, x, y, mode, mini, e1, e2):
    """0, 1 or 2: how close the player's small box is to a hazard or solid (within e2 units: 1, within e1: 2)."""
    small = 0.6 if mini else 1.0
    ih = (5.0 if mode == 4 else 4.5) * small
    hb = (5.0 if mode == 4 else 15.0) * small * GT[31][0]
    c = 0
    if hazard_hit(GT, col0, ncols, x, y, hb, e2) or inner_hits_solid(GT, col0, ncols, x, y, ih + e2):
        c = 1
        if hazard_hit(GT, col0, ncols, x, y, hb, e1) or inner_hits_solid(GT, col0, ncols, x, y, ih + e1):
            c = 2
    return c


@njit(cache=True)
def run_dp(GT, col0, ncols, T, SX, SSPD, SMODE, SMINI, SFLO, SCEI, EVP, EVT, EVV, p0, eps, Kmax, qs0, grace, stop, wc, wn, e1, e2, sub):
    """Forward DP over player states. Decisions (hold / release) every 4 ticks (60 Hz); equal-cell states merge keeping the
    fewest input changes. With `sub`, a step where every 60 Hz choice dies is retried with inputs that change on the 2nd,
    3rd or 4th physics tick (GD 2.2 reads clicks at physics-tick precision); the step's input code is then final hold |
    (switch tick << 1). Doing this everywhere costs 3x the time for little gain. When every state dies the step is replayed with temporary invulnerability (a 'forced pass')
    so the rest of the level can still be measured. Returns (inputs per step, alive count per step, forced-pass steps,
    best cost, dropped states, deaths by code)."""
    nsteps = (T + 3) // 4
    curP = np.zeros((Kmax, NS)); nxtP = np.zeros((Kmax, NS))
    curH = np.zeros(Kmax, np.uint8); nxtH = np.zeros(Kmax, np.uint8)
    curC = np.zeros(Kmax, np.int32); nxtC = np.zeros(Kmax, np.int32)
    par = np.zeros(Kmax, np.int32); inp = np.zeros(Kmax, np.uint8)
    tkey = np.zeros(TABLE, np.int64); tidx = np.zeros(TABLE, np.int32); tstamp = np.zeros(TABLE, np.int32)
    tmp = np.zeros(NS)
    pars = List(); inps = List()
    counts = np.zeros(nsteps, np.int32)
    fails = np.zeros(nsteps, np.uint8)
    deaths = np.zeros(4, np.int64)
    for k in range(NS):
        curP[0, k] = p0[k]
    cur_n = 1
    stamp = 0
    qs = qs0
    dropped = 0
    for s in range(nsteps):
        t0 = s * 4
        t1 = min(t0 + 4, T)
        attempt = 0
        fine = False                                          # 240 Hz input switches, tried only where 60 Hz inputs all die
        while True:
            stamp += 1
            nxt_n = 0
            drop_here = 0
            for i in range(cur_n):
                h0 = curH[i]
                for a in range(5 if fine else 2):                     # 0 keep, 1 switch on tick 0, 2..4 switch on tick 1..3
                    u = h0 if a == 0 else 1 - h0
                    ks = a - 1 if a > 0 else 0
                    for k in range(NS):
                        tmp[k] = curP[i, k]
                    if attempt == 1:
                        tmp[INV] = grace
                    elif attempt >= 2:
                        tmp[INV] = 1e9
                    alive = True
                    for tk in range(t0, t1):
                        for e in range(EVP[tk], EVP[tk + 1]):
                            apply_event(tmp, EVT[e], EVV[e])
                        ut = h0 if tk - t0 < ks else u
                        press = 1 if (u == 1 and h0 == 0 and tk - t0 == ks) else 0
                        code = step_tick(tmp, GT, col0, ncols, SX[tk + 1], SMODE[tk], SMINI[tk], SSPD[tk], SFLO[tk], SCEI[tk], ut, press, eps)
                        if code != 0:
                            deaths[code] += 1
                            alive = False
                            break
                    if not alive:
                        continue
                    cost = curC[i] + (wc if u != h0 else 0)
                    if wn > 0:
                        cost += wn * near_cost(GT, col0, ncols, SX[t1], tmp[Y], SMODE[t1 - 1], SMINI[t1 - 1], e1, e2)
                    key = make_key(tmp, SMODE[t1 - 1], qs, u)
                    slot = ((key * 2654435761) >> 7) & (TABLE - 1)
                    found = -1
                    while tstamp[slot] == stamp:
                        if tkey[slot] == key:
                            found = tidx[slot]
                            break
                        slot = (slot + 1) & (TABLE - 1)
                    if found >= 0:
                        if cost < nxtC[found]:
                            for k in range(NS):
                                nxtP[found, k] = tmp[k]
                            nxtC[found] = cost; par[found] = i; inp[found] = u | (ks << 1); nxtH[found] = u
                    elif nxt_n < Kmax:
                        tstamp[slot] = stamp; tkey[slot] = key; tidx[slot] = nxt_n
                        for k in range(NS):
                            nxtP[nxt_n, k] = tmp[k]
                        nxtC[nxt_n] = cost; par[nxt_n] = i; inp[nxt_n] = u | (ks << 1); nxtH[nxt_n] = u
                        nxt_n += 1
                    else:
                        drop_here += 1
            if nxt_n > 0:
                break
            if sub and attempt == 0 and not fine:
                fine = True
                continue
            fine = False
            if attempt == 0:
                fails[s] = 1
                if stop == 1:                                  # debugging: hand back the last states (+ hold flag) before the first total death
                    snap = np.zeros((cur_n, NS + 1))
                    snap[:, :NS] = curP[:cur_n]
                    snap[:, NS] = curH[:cur_n]
                    return np.zeros(nsteps, np.uint8), counts, fails, s, dropped, deaths, snap
            attempt += 1
            if attempt > 2:
                return np.zeros(nsteps, np.uint8), counts, fails, -1, dropped, deaths, np.zeros((1, NS + 1))
        dropped += drop_here
        if drop_here > 0 or nxt_n > 0.85 * Kmax:
            qs = min(qs * 1.5, 12.0)
        elif nxt_n < 0.3 * Kmax and qs > qs0:
            qs = max(qs / 1.25, qs0)
        counts[s] = nxt_n
        pars.append(par[:nxt_n].copy()); inps.append(inp[:nxt_n].copy())
        curP, nxtP = nxtP, curP
        curH, nxtH = nxtH, curH
        curC, nxtC = nxtC, curC
        cur_n = nxt_n
    best = 0
    for i in range(cur_n):
        if curC[i] < curC[best]:
            best = i
    out = np.zeros(nsteps, np.uint8)
    idx = best
    for s in range(nsteps - 1, -1, -1):
        out[s] = inps[s][idx]
        idx = pars[s][idx]
    return out, counts, fails, curC[best], dropped, deaths, np.zeros((1, NS + 1))


# ----------------------------------------------------------------- driver
def plan(world, sch, eps=0.0, kmax=3000, grace=40.0, stop=0, wc=3, wn=1, e1=3.0, e2=8.0, sub=SUB):
    """Run the planner on a prepared level. Returns a dict describing the cheapest route and where it was forced."""
    p0 = start_state(world, sch)
    T = sch["T"]
    out, counts, fails, cost, dropped, deaths, snap = run_dp(
        world["GT"], world["col0"], world["ncols"], T, sch["x"], sch["spd"], sch["mode"], sch["mini"], sch["floor"], sch["ceil"],
        sch["ev_ptr"], sch["ev_type"], sch["ev_val"], p0, float(eps), int(kmax), 1.0, float(grace), int(stop), int(wc), int(wn), float(e1), float(e2), int(sub))
    if stop:                                                          # states / fail_step describe the first total death (-1: none)
        return dict(states=snap, fail_step=int(cost) if fails.any() else -1, counts=counts)
    return dict(inputs=out, counts=counts, fails=fails, cost=int(cost), dropped=int(dropped), deaths=deaths, steps=len(out))


# ----------------------------------------------------------------- timing-noise test of a finished route
@njit(cache=True)
def replay(GT, col0, ncols, T, SX, SSPD, SMODE, SMINI, SFLO, SCEI, EVP, EVT, EVV, p0, inputs, forced):
    """Play a recorded route (one hold flag per 60 Hz step) and return the player state at the start of every step.
    Steps flagged in `forced` are invulnerable, exactly like in the planner."""
    nsteps = len(inputs)
    S = np.zeros((nsteps + 1, NS))
    p = p0.copy()
    S[0, :] = p
    prev = 0
    for s in range(nsteps):
        u = inputs[s] & 1
        ks = inputs[s] >> 1
        t0 = s * 4
        t1 = min(t0 + 4, T)
        if forced[s]:
            p[INV] = 1e9
        for tk in range(t0, t1):
            for e in range(EVP[tk], EVP[tk + 1]):
                apply_event(p, EVT[e], EVV[e])
            ut = prev if tk - t0 < ks else u
            press = 1 if (u == 1 and prev == 0 and tk - t0 == ks) else 0
            step_tick(p, GT, col0, ncols, SX[tk + 1], SMODE[tk], SMINI[tk], SSPD[tk], SFLO[tk], SCEI[tk], ut, press, 0.0)
        if forced[s]:
            p[INV] = 0.0
        prev = u
        S[s + 1, :] = p
    return S


@njit(cache=True)
def mc_core(GT, col0, ncols, T, SX, SSPD, SMODE, SMINI, SFLO, SCEI, EVP, EVT, EVV, S, inputs, forced, sigma, trials, seg, seed):
    """Replay the route with every input change shifted by a random number of frames (normal, std `sigma`, applied in
    240 Hz ticks) and count, per segment of `seg` steps, how often the player dies when starting that segment from the
    clean route's state."""
    np.random.seed(seed)
    nsteps = len(inputs)
    nseg = (nsteps + seg - 1) // seg
    fails = np.zeros(nseg)
    nt = nsteps * 4
    ntog = 0
    prev = 0
    for s in range(nsteps):
        if (inputs[s] & 1) != prev:
            ntog += 1
        prev = inputs[s] & 1
    tt = np.zeros(ntog, np.int64)                                   # tick of every input change on the clean route
    k = 0
    prev = 0
    for s in range(nsteps):
        if (inputs[s] & 1) != prev:
            tt[k] = s * 4 + (inputs[s] >> 1)
            k += 1
        prev = inputs[s] & 1
    H = np.zeros(nt, np.uint8)
    for trial in range(trials):
        for t in range(nt):
            H[t] = 0
        last = -1
        state = 1
        for i in range(ntog):
            t = tt[i] + int(np.round(np.random.normal(0.0, sigma) * 4.0))
            if t <= last:
                t = last + 1
            if t >= nt:
                break
            for q in range(t, nt):
                H[q] = state
            state = 1 - state
            last = t
        for sg in range(nseg):
            s0 = sg * seg
            s1 = min(s0 + seg, nsteps)
            p = S[s0].copy()
            prev = H[s0 * 4 - 1] if s0 > 0 else 0
            dead = False
            for s in range(s0, s1):
                t0 = s * 4
                t1 = min(t0 + 4, T)
                if forced[s]:
                    p[INV] = 1e9
                for tk in range(t0, t1):
                    for e in range(EVP[tk], EVP[tk + 1]):
                        apply_event(p, EVT[e], EVV[e])
                    u = H[tk]
                    press = 1 if (u == 1 and prev == 0) else 0
                    prev = u
                    if step_tick(p, GT, col0, ncols, SX[tk + 1], SMODE[tk], SMINI[tk], SSPD[tk], SFLO[tk], SCEI[tk], u, press, 0.0) != 0:
                        dead = True
                        break
                if dead:
                    break
                if forced[s]:
                    p[INV] = 0.0
            if dead:
                fails[sg] += 1.0
    return fails / trials


def jitter_profile(world, sch, plan_result, sigmas=(1.5, 3.0), trials=24, seg=30, seed=1):
    """-> {sigma: failure probability per segment} for a finished planner run."""
    inputs = plan_result["inputs"]
    forced = plan_result["fails"][:len(inputs)].astype(np.uint8)
    args = (world["GT"], world["col0"], world["ncols"], sch["T"], sch["x"], sch["spd"], sch["mode"], sch["mini"], sch["floor"], sch["ceil"],
            sch["ev_ptr"], sch["ev_type"], sch["ev_val"])
    S = replay(*args, start_state(world, sch), inputs, forced)
    return {s: mc_core(*args, S, inputs, forced, float(s), int(trials), int(seg), int(seed)) for s in sigmas}


# ----------------------------------------------------------------- how close does the finished route come to things?
CLR = np.array([1.0, 2.0, 4.0, 6.0, 9.0, 13.0])


@njit(cache=True)
def clearance_core(GT, col0, ncols, T, SX, SMODE, SMINI, S, nsteps, thr):
    """Per step: how many of the distance thresholds have a hazard or solid inside them (6 = within 1 unit ... 1 = within 13)."""
    out = np.zeros(nsteps, np.int8)
    for s in range(nsteps):
        t1 = min(s * 4 + 4, T)
        x, y, mode, mini = SX[t1], S[s + 1, Y], SMODE[t1 - 1], SMINI[t1 - 1]
        ih = (5.0 if mode == 4 else 4.5) * (0.6 if mini else 1.0)
        hb = (5.0 if mode == 4 else 15.0) * (0.6 if mini else 1.0) * GT[31][0]
        n = 0
        for j in range(len(thr)):
            if hazard_hit(GT, col0, ncols, x, y, hb, thr[j]) or inner_hits_solid(GT, col0, ncols, x, y, ih + thr[j]):
                n = len(thr) - j
                break
        out[s] = n
    return out


def route_clearance(world, sch, plan_result):
    inputs = plan_result["inputs"]
    forced = plan_result["fails"][:len(inputs)].astype(np.uint8)
    args = (world["GT"], world["col0"], world["ncols"], sch["T"], sch["x"], sch["spd"], sch["mode"], sch["mini"], sch["floor"], sch["ceil"],
            sch["ev_ptr"], sch["ev_type"], sch["ev_val"])
    S = replay(*args, start_state(world, sch), inputs, forced)
    return clearance_core(world["GT"], world["col0"], world["ncols"], sch["T"], sch["x"], sch["mode"], sch["mini"], S, len(inputs), CLR)
