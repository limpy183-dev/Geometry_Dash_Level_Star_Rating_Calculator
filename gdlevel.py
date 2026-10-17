"""Geometry Dash level string -> gameplay timeline + difficulty features.

No game physics engine: the level is read as a timeline (portals, speeds), hazards are grouped into the
inputs they demand, and fly sections (ship/wave/UFO/swing) are measured by how tight the corridor is.
The features feed the trained model in model.py; `timeline` and `stats` feed the UI.

ID tables: portal/orb/pad/trigger IDs follow the MIT-licensed gmdkit constants; hazard/solid classes
come from objclasses.json (see tools/build_objclasses.py).
"""
import bisect, json, math, os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
_cls = json.load(open(os.path.join(HERE, "objclasses.json")))
HAZ, SAW, FAKE, INVIS, SOLID = (frozenset(_cls[k]) for k in ("hazard", "saw", "fake", "invisible", "solid"))

MODES = ["cube", "ship", "ball", "ufo", "wave", "robot", "spider", "swing"]
FLY = {1, 3, 4, 7}                                    # modes where the corridor shape, not single obstacles, drives inputs
SPEED = [311.58, 251.16, 387.42, 468.0, 576.0]        # units/s for kA4 / speed-portal index: 1x, 0.5x, 2x, 3x, 4x
SPEED_NAMES = ["1x", "0.5x", "2x", "3x", "4x"]
SPEED_PORTAL = {201: 0, 200: 1, 202: 2, 203: 3, 1334: 4}
MODE_PORTAL = {12: 0, 13: 1, 47: 2, 111: 3, 660: 4, 745: 5, 1331: 6, 1933: 7}
ORB_KIND = {36: "jump", 141: "jump", 1333: "jump", 84: "grav", 1022: "grav", 1330: "grav", 1704: "dash", 1751: "dash",
            3004: "other", 1594: "other", 3027: "other"}
PADS = frozenset((35, 67, 140, 1332, 3005))
SIZE_PORTAL, GRAV_PORTAL, MIRROR_PORTAL, DUAL_PORTAL = {99: 0, 101: 1}, {10: 0, 11: 1, 2926: 2}, {45: 1, 46: 0}, {286: 1, 287: 0}
COIN, END_TRIGGER = frozenset((142, 1329, 1614)), 3600
_F = {  # trigger families (counts only: moving/hidden/logic gimmicks are not simulated)
    "move": {901, 1347, 1814, 3016, 3660, 3661, 3006, 3011, 3017, 3032, 3033, 2999, 3645, 2069, 3655},
    "rot": {1346, 2067, 3007, 3008, 3012, 3013, 3018, 3019},
    "color": {899, 1006, 1007, 3009, 3010, 3014, 3015, 3020, 3021, 2903, 1818, 1819, 3029, 3030, 3031, 3606, 3612} | set(range(2904, 2925)),
    "toggle": {1049, 1612, 1613},
    "logic": {1268, 1912, 2068, 3607, 1611, 1811, 1817, 3619, 3620, 3641, 3614, 3615, 3617, 3618, 1815, 3609, 1812, 1595,
              1616, 3604, 3613, 3024, 3023, 1585, 3608, 3640, 3643, 1816},
    "camera": {1913, 1914, 1916, 2015, 2016, 2062, 2901, 2925, 1520, 1917, 1935},
    "audio": {1934, 3602, 3603, 3605},
}
FAMILY = {i: f for f, ids in _F.items() for i in ids}
TRIGGER_ALL = frozenset(_cls["trigger"]) | frozenset(FAMILY) | {1932, 2066, 2063, 3022, 3642, 2900, 31, 32, 33}

# object id -> category code
H_, S_, O_, P_, X_, T_, C_, F_ = 1, 2, 3, 4, 5, 6, 7, 8
CAT = {}
for _i in SOLID: CAT[_i] = S_
for _i in FAKE: CAT[_i] = F_
for _i in HAZ: CAT[_i] = H_
for _i in ORB_KIND: CAT[_i] = O_
for _i in PADS: CAT[_i] = P_
for _i in (*MODE_PORTAL, *SIZE_PORTAL, *GRAV_PORTAL, *MIRROR_PORTAL, *DUAL_PORTAL, *SPEED_PORTAL): CAT[_i] = X_
for _i in TRIGGER_ALL: CAT.setdefault(_i, T_)
for _i in COIN: CAT[_i] = C_

COL = 30.0                                            # one block in level units
X_MAX, Y_MAX = 2_000_000.0, 1_000_000.0               # ~107 min at 1x; anything beyond is corrupt data and must not size our arrays
AUXK = frozenset(("57", "51", "28", "29", "10", "6", "4", "5", "121", "135"))   # groups, target, move x/y, duration; rotation, flips; no-collision flags
MOVERS = frozenset((901, 1346, 1347, 2067, 3016, 3006, 3007, 3008, 3011, 3012, 3013, 3017, 3018, 3019, 3032, 2999))  # trigger ids that move / rotate / scale


def _pct(a, q):
    if not a: return 0.0
    a = sorted(a)
    k = (len(a) - 1) * q
    f = int(k)
    return a[f] + (a[min(f + 1, len(a) - 1)] - a[f]) * (k - f)


def parse(text, keys=AUXK):
    """-> (header dict, [(id, x, y, scale)], aux). aux[i] = {key: value} for the property keys listed in `keys` (groups, trigger
    target, move x/y, duration, ...) on objects that have any of them. Tolerant: malformed objects are skipped."""
    head, *rest = text.split(";")
    hp = head.split(",")
    hdr = dict(zip(hp[0::2], hp[1::2]))
    objs, aux = [], {}
    for s in rest:
        p = s.split(",")
        if len(p) < 6:
            continue
        oid = x = y = None
        sc = "1"
        a = None
        for i in range(0, len(p) - 1, 2):
            k = p[i]
            if k == "1": oid = p[i + 1]
            elif k == "2": x = p[i + 1]
            elif k == "3": y = p[i + 1]
            elif k == "32": sc = p[i + 1]
            elif k in keys:
                a = a or {}
                a[k] = p[i + 1]
        try:
            ox, oy = float(x), float(y)
            if not (-X_MAX < ox < X_MAX and -Y_MAX < oy < Y_MAX):        # also rejects nan / inf
                continue
            objs.append((int(oid), ox, oy, float(sc)))
        except (TypeError, ValueError):
            continue
        if a:
            aux[len(objs) - 1] = a
    return hdr, objs, aux


def build_segments(hdr, objs, x_end):
    """Portal events -> [(x0, x1, mode, mini, grav, dual, mirror, speed)] covering 0..x_end."""
    ev = sorted((x, oid, y) for oid, x, y, _ in objs if CAT.get(oid) == X_ and 0 <= x <= x_end)
    s = dict(mode=int(hdr.get("kA2", 0) or 0) % 8, mini=hdr.get("kA3") == "1", grav=hdr.get("kA11") == "1",
             dual=hdr.get("kA8") == "1", mirror=hdr.get("kA28") == "1", speed=min(max(int(hdr.get("kA4", 0) or 0), 0), 4))
    segs, seg_y, cx, cy = [], [], 0.0, 165.0

    def push(x):
        nonlocal cx
        if x > cx:
            segs.append((cx, x, s["mode"], s["mini"], s["grav"], s["dual"], s["mirror"], s["speed"]))
            seg_y.append(cy)
            cx = x
    for x, oid, y in ev:
        push(x)
        if oid in MODE_PORTAL: cy = y
        if oid in MODE_PORTAL: s["mode"] = MODE_PORTAL[oid]
        elif oid in SIZE_PORTAL: s["mini"] = bool(SIZE_PORTAL[oid])
        elif oid in GRAV_PORTAL: s["grav"] = (not s["grav"]) if GRAV_PORTAL[oid] == 2 else bool(GRAV_PORTAL[oid])
        elif oid in MIRROR_PORTAL: s["mirror"] = bool(MIRROR_PORTAL[oid])
        elif oid in DUAL_PORTAL: s["dual"] = bool(DUAL_PORTAL[oid])
        elif oid in SPEED_PORTAL: s["speed"] = SPEED_PORTAL[oid]
    push(max(x_end, cx + 1))
    return segs, seg_y, len(ev)


def _track(cols, c0, c1, y0, speed, mode, mini):
    """Follow the free gap through columns c0..c1 -> per-column (gap_cells, mid_cells, dt)."""
    hb = {4: 0.15 if mini else 0.3, 1: 0.45 if mini else 0.75}.get(mode, 0.75)       # player hitbox height in cells
    dt, cur, out = COL / SPEED[speed], y0 / COL, []
    for c in range(c0, c1):
        occ = sorted(set(cols.get(c, ())))
        if occ:
            taken = set(occ)
            for d in range(8):                                         # inside an obstacle: hop to the nearest free cell
                if int(cur) + d not in taken: cur = int(cur) + d + 0.5; break
                if int(cur) - d not in taken: cur = int(cur) - d + 0.5; break
            i = bisect.bisect_left(occ, int(cur))
            lo = occ[i - 1] + 1 if i > 0 else cur - 6
            hi = occ[i] if i < len(occ) else cur + 6
            lo, hi = max(lo, cur - 10), min(hi, cur + 10)
        else:
            lo, hi = cur - 6, cur + 6
        gap = max(hi - lo, 0.5)
        if gap <= 6:
            cur = (lo + hi) / 2
        out.append((gap, cur, dt, max(gap - hb, 0.35)))
    return out


def analyze(text):
    """Level text -> dict(features, timeline, stats). Features always contain the same keys."""
    hdr, objs, aux = parse(text)
    if not objs:
        raise ValueError("No objects found in that level.")
    play = [o for o in objs if CAT.get(o[0]) in (H_, S_, O_, P_, X_)]
    end_x = [x for oid, x, y, _ in objs if oid == END_TRIGGER]
    x_end = (min(end_x) if end_x else max((o[1] for o in play), default=max(o[1] for o in objs)))
    x_end = max(x_end, 300.0)
    segs, seg_y, n_portal = build_segments(hdr, objs, x_end)
    starts, t0 = [s[0] for s in segs], []
    acc = 0.0
    for s in segs:
        t0.append(acc)
        acc += (s[1] - s[0]) / SPEED[s[7]]
    total_t = max(acc, 1.0)

    def seg_of(x):
        return max(bisect.bisect_right(starts, x) - 1, 0)

    def t_of(x):
        i = seg_of(x)
        return t0[i] + (min(x, segs[i][1]) - segs[i][0]) / SPEED[segs[i][7]] if x >= 0 else 0.0

    nb = int(total_t) + 2
    z = lambda: [0.0] * nb
    ground, orbs, pads, hazb, portalb, turn, dem, tight, objb = z(), z(), z(), z(), z(), z(), z(), z(), z()
    cnt = defaultdict(int)
    haz_t, orb_t, cols, hz_scaled = [], [], defaultdict(list), 0
    fam = defaultdict(int)
    for oid, x, y, sc in objs:
        c = CAT.get(oid, 0)
        if x > x_end + 60:
            continue
        b = min(int(t_of(x)), nb - 1)
        objb[b] += 1
        if c == H_:
            cnt["haz"] += 1; hazb[b] += 1
            cnt["saw"] += oid in SAW; cnt["invis"] += oid in INVIS
            hz_scaled += abs(sc - 1.0) > 0.05
            cc, yy = int(x // COL), int(y // COL)
            cols[cc].append(yy)
            if oid in SAW:
                cols[cc].extend((yy - 1, yy + 1))
            gm = segs[seg_of(x)][2]
            if gm not in FLY:
                haz_t.append((t_of(x), gm))
        elif c == S_:
            cnt["solid"] += 1
            cols[int(x // COL)].append(int(y // COL))
        elif c == O_:
            k = ORB_KIND[oid]; cnt["orb_" + k] += 1; orbs[b] += 1; orb_t.append(t_of(x))
        elif c == P_:
            cnt["pad"] += 1; pads[b] += 1
        elif c == X_:
            portalb[b] += 1
        elif c == T_:
            cnt["trig"] += 1; fam[FAMILY.get(oid, "other")] += 1
        elif c == F_:
            cnt["fake"] += 1
        elif c == C_:
            cnt["coin"] += 1
        else:
            cnt["deco"] += 1
    # ---- gimmicks: how many hazards / platforms do triggers move or toggle, and how far and fast?
    moved, toggled, hz_g, sl_g = set(), set(), set(), set()
    for i, a in aux.items():
        t = objs[i][0]
        if "51" in a and t in MOVERS: moved.add(a["51"])
        elif "51" in a and t == 1049: toggled.add(a["51"])
        elif "57" in a:
            c = CAT.get(t)
            if c == H_: hz_g.update(a["57"].split("."))
            elif c == S_: sl_g.update(a["57"].split("."))
    hz_moved = hz_toggled = sl_moved = 0
    for i, a in aux.items():
        if "57" in a:
            c, gs = CAT.get(objs[i][0]), a["57"].split(".")
            if c == H_:
                hz_moved += bool(moved.intersection(gs)); hz_toggled += bool(toggled.intersection(gs))
            elif c == S_:
                sl_moved += bool(moved.intersection(gs))
    mv_d, mv_fast = [], 0
    for i, a in aux.items():
        if objs[i][0] == 901 and a.get("51") in hz_g | sl_g:
            try:
                d = (abs(float(a.get("28", 0))) + abs(float(a.get("29", 0)))) / COL
                mv_d.append(d)
                mv_fast += d / max(float(a.get("10", 0)), 0.05) > 6
            except ValueError:
                pass
    # ---- ground sections: one input per cluster of hazards close together in time
    haz_t.sort()
    gaps, last, g_cube, g_other, widths, c0 = [], -9.0, z(), z(), [], 0.0
    for t, gm in haz_t:
        if t - last > 0.2:
            b = min(int(t), nb - 1)
            ground[b] += 1
            (g_cube if gm == 0 else g_other)[b] += 1
            if last > 0: gaps.append(t - last); widths.append(last - c0 + 0.1)
            c0 = t
        last = t
    orb_t.sort()
    orb_gaps = [b - a for a, b in zip(orb_t, orb_t[1:])]
    # ---- fly sections: corridor tightness + direction changes of the free path
    mode_t, flags_t = [0.0] * 8, dict(mini=0.0, dual=0.0, grav=0.0, mirror=0.0)
    speed_t = [0.0] * 5
    fly = defaultdict(list)                                   # mode -> list of (gap, h, dt, speed)
    dem_m, turn_m, gnd = {k: z() for k in "wsu"}, {k: z() for k in "wsu"}, []
    tag_of = {4: "w", 1: "s", 7: "s", 3: "u"}
    for i, (xa, xb, mode, mini, grav, dual, mir, spd) in enumerate(segs):
        dur = (xb - xa) / SPEED[spd]
        mode_t[mode] += dur; speed_t[spd] += dur
        for k, v in (("mini", mini), ("grav", grav), ("dual", dual), ("mirror", mir)):
            flags_t[k] += dur * v
        if xb - xa < 60:
            continue
        path = _track(cols, int(xa // COL), int(xb // COL), seg_y[i], spd, mode, mini)
        if mode not in FLY:
            gnd.extend((p[0], p[2]) for p in path)
            continue
        ext, direction, tcur = None, 0, t0[i]
        for gap, mid, dt, h in path:
            b = min(int(tcur), nb - 1)
            fly[mode].append((gap, h, dt, spd))
            if gap <= 8:
                k = {4: (2.0 if mini else 1.0)}.get(mode, 0.6)           # wave slope factor; ship/UFO react more slowly
                d = (SPEED[spd] / COL) * k / h * dt                    # zig-zag lower bound, turns per second (dt-weighted)
                dem[b] += d; dem_m[tag_of[mode]][b] += d
            if gap <= 3.5:
                tight[b] += dt
            if ext is None:
                ext = mid
            if direction >= 0 and mid > ext: ext = mid; direction = 1
            elif direction <= 0 and mid < ext: ext = mid; direction = -1
            elif direction == 1 and mid < ext - 0.6: turn[b] += 1; turn_m[tag_of[mode]][b] += 1; direction, ext = -1, mid
            elif direction == -1 and mid > ext + 0.6: turn[b] += 1; turn_m[tag_of[mode]][b] += 1; direction, ext = 1, mid
            tcur += dt
    # ---- per-second series
    def smooth(a, w=3):
        n = len(a)
        return [sum(a[max(0, j - w // 2): j + w // 2 + 1]) / len(a[max(0, j - w // 2): j + w // 2 + 1]) for j in range(n)]
    dem = [min(v, 25.0) for v in dem]
    inputs = [ground[b] + orbs[b] + max(turn[b], dem[b]) for b in range(nb)]
    in_m = {k: [max(turn_m[k][b], min(dem_m[k][b], 25.0)) for b in range(nb)] for k in "wsu"}
    series = dict(inputs=inputs, ground=ground, orb=orbs, turn=turn, dem=dem, tight=tight, haz=hazb, portal=portalb, obj=objb)
    spd_b = [1.0] * nb                                         # speed relative to 1x, per second
    for i, sg in enumerate(segs):
        for b in range(int(t0[i]), min(int(t0[i] + (sg[1] - sg[0]) / SPEED[sg[7]]) + 1, nb)):
            spd_b[b] = SPEED[sg[7]] / SPEED[0]
    strain = [inputs[b] * (1 + tight[b]) * spd_b[b] for b in range(nb)]       # hard in several ways at the same moment
    extra = dict(in_wave=in_m["w"], in_ship=in_m["s"], in_ufo=in_m["u"], g_cube=g_cube, g_other=g_other, strain=strain,
                 strain_spd=[inputs[b] * spd_b[b] for b in range(nb)])
    f = {}
    for name, a in {**series, **extra}.items():
        s = sorted(smooth(a))
        f[f"{name}_mean"] = sum(a) / nb
        f[f"{name}_p90"], f[f"{name}_max"] = _pct(s, .9), s[-1]
        if name in series:
            f[f"{name}_p75"] = _pct(s, .75)
    top = sorted(smooth(inputs), reverse=True)
    f["inputs_top10"] = sum(top[:max(1, nb // 10)]) / max(1, nb // 10)
    f["inputs_total"] = sum(inputs)
    ranked = sorted(smooth(strain), reverse=True)                # osu!-style: hardest sections count most, with decaying weight
    f["strain_osu"] = sum(v * 0.93 ** i for i, v in enumerate(ranked))
    f["strain_osu_norm"] = f["strain_osu"] / max(sum(0.93 ** i for i in range(len(ranked))), 1e-9)
    sm_s = smooth(strain)                                      # where, how fast and in which mode is the hardest moment?
    pk = sm_s.index(max(sm_s))
    pk_seg = max(bisect.bisect_right(t0, pk) - 1, 0)
    f["peak_speed"], f["peak_pos"] = spd_b[pk], pk / nb
    for i, m in enumerate(MODES):
        f["peak_" + m] = float(segs[pk_seg][2] == i)
    sm, h = smooth(inputs), nb // 2
    f["inputs_h1"], f["inputs_h2"] = sum(inputs[:h]) / max(h, 1), sum(inputs[h:]) / max(nb - h, 1)
    for lim in (6, 10, 14):                                    # seconds spent above a pressure level (endurance at the top end)
        f[f"hard_s{lim}"] = float(sum(v > lim for v in sm))
    # ---- composition
    n = len(objs)
    f.update(len_s=total_t, log_len=math.log(total_t), x_len=x_end, n_obj=n, log_obj=math.log(n), obj_per_s=n / total_t,
             deco_frac=cnt["deco"] / n, deco_per_s=cnt["deco"] / total_t, solid_frac=cnt["solid"] / n,
             n_haz=cnt["haz"], haz_per_s=cnt["haz"] / total_t, saw_frac=cnt["saw"] / max(cnt["haz"], 1),
             invis_haz=cnt["invis"], fake_haz=cnt["fake"], scaled_haz_frac=hz_scaled / max(cnt["haz"], 1),
             n_orb=sum(v for k, v in cnt.items() if k.startswith("orb_")), orb_jump=cnt["orb_jump"], orb_grav=cnt["orb_grav"],
             orb_dash=cnt["orb_dash"], orb_other=cnt["orb_other"], n_pad=cnt["pad"], n_coin=cnt["coin"],
             n_portal=n_portal, portals_per_min=n_portal / total_t * 60, n_trig=cnt["trig"], trig_per_s=cnt["trig"] / total_t,
             platformer=float(hdr.get("kA22") == "1"), two_player=float(hdr.get("kA10") == "1"),
             start_mode=int(hdr.get("kA2", 0) or 0), start_speed=int(hdr.get("kA4", 0) or 0))
    for k in ("move", "rot", "color", "toggle", "logic", "camera", "audio", "other"):
        f["trig_" + k] = fam[k]
        f["trig_" + k + "_ps"] = fam[k] / total_t
    for i, m in enumerate(MODES):
        f["share_" + m] = mode_t[i] / total_t
    for k, v in flags_t.items():
        f["share_" + k] = v / total_t
    for i in range(5):
        f[f"speed_{SPEED_NAMES[i]}"] = speed_t[i] / total_t
    f["speed_mean"] = sum(SPEED[i] * speed_t[i] for i in range(5)) / total_t
    # ---- ground spacing + fly corridors
    f["cluster_gap_p10"], f["cluster_gap_p50"] = (_pct(gaps, .1), _pct(gaps, .5)) if gaps else (9.0, 9.0)
    f["cluster_fast_frac"] = sum(g < 0.4 for g in gaps) / max(len(gaps), 1)
    f["n_ground_clusters"] = sum(ground)
    gw = sum(d for _, d in gnd) or 1e-9
    f["ground_gap_mean"] = sum(g * d for g, d in gnd) / gw if gnd else 10.0
    for lim in (3, 4):
        f[f"ground_le{lim}"] = sum(d for g, d in gnd if g <= lim) / gw if gnd else 0.0
    f["ground_width_p90"] = _pct(widths, .9) if widths else 0.0
    f["ground_longjump_frac"] = sum(w > 0.27 for w in widths) / max(len(widths), 1)
    f["ground_longjump_n"] = float(sum(w > 0.27 for w in widths))
    f.update(haz_moved_frac=hz_moved / max(cnt["haz"], 1), haz_moved_n=float(hz_moved), haz_toggled_frac=hz_toggled / max(cnt["haz"], 1),
             solid_moved_frac=sl_moved / max(cnt["solid"], 1), move_haz_trig=float(len(mv_d)), move_haz_trig_ps=len(mv_d) / total_t,
             move_dist_mean=sum(mv_d) / len(mv_d) if mv_d else 0.0, move_fast_n=float(mv_fast))
    f["orb_chain_frac"] = sum(g < 0.4 for g in orb_gaps) / max(len(orb_gaps), 1)
    f["orb_gap_p10"] = _pct(orb_gaps, .1) if orb_gaps else 9.0
    for tag, modes in (("wave", (4,)), ("ship", (1, 7)), ("ufo", (3,)), ("fly", tuple(FLY))):
        rows = [r for m in modes for r in fly.get(m, [])]
        w = sum(r[2] for r in rows) or 1e-9
        f[f"{tag}_time"] = sum(r[2] for r in rows)
        f[f"{tag}_gap_mean"] = sum(r[0] * r[2] for r in rows) / w if rows else 10.0
        f[f"{tag}_gap_p10"] = _pct([r[0] for r in rows], .1) if rows else 10.0
        for lim in (2, 3, 4):
            f[f"{tag}_le{lim}"] = sum(r[2] for r in rows if r[0] <= lim) / w if rows else 0.0
        f[f"{tag}_demand"] = sum((SPEED[r[3]] / COL) / r[1] * r[2] for r in rows if r[0] <= 8) / w if rows else 0.0
    # ---- UI payload
    step = max(1, math.ceil(nb / 480))
    tl_mode, ti = [], 0
    for b in range(0, nb, step):
        while ti + 1 < len(segs) and t0[ti + 1] <= b:
            ti += 1
        tl_mode.append(segs[ti][2])
    timeline = dict(step=step, inputs=[round(max(inputs[b:b + step]), 2) for b in range(0, nb, step)],
                    tight=[round(max(tight[b:b + step]), 2) for b in range(0, nb, step)], mode=tl_mode)
    stats = dict(length_s=round(total_t, 1), objects=n, hazards=cnt["haz"], orbs=f["n_orb"], pads=cnt["pad"],
                 portals=n_portal, triggers=cnt["trig"], deco=cnt["deco"], coins=cnt["coin"],
                 mode_share={m: round(mode_t[i] / total_t, 4) for i, m in enumerate(MODES) if mode_t[i] > 0},
                 speed_share={SPEED_NAMES[i]: round(speed_t[i] / total_t, 4) for i in range(5) if speed_t[i] > 0},
                 platformer=bool(f["platformer"]), two_player=bool(f["two_player"]))
    return dict(features=f, timeline=timeline, stats=stats)


def _demo():                                              # python gdlevel.py  -> self-check on a synthetic level
    objs = ";".join(f"1,8,2,{600 + 150 * i},3,15" for i in range(20))
    objs += ";1,13,2,4000,3,165;" + ";".join(f"1,1,2,{4300 + 30 * i},3,{15 if i % 2 else 315}" for i in range(60))
    r = analyze("kA2,0,kA4,0;" + objs)
    f = r["features"]
    assert f["n_haz"] == 20 and f["n_portal"] == 1 and 0.1 < f["share_ship"] < 0.7, f
    assert f["cluster_gap_p50"] > 0.3 and f["ship_gap_mean"] > 0, f
    f2 = analyze("kA2,0,kA4,4;" + objs)["features"]            # 4x speed must shorten the level
    assert f2["len_s"] < f["len_s"] * 0.7
    print("gdlevel self-check ok:", round(f["len_s"], 1), "s,", int(f["n_obj"]), "objects")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        import gdnet, time
        meta, s = gdnet.download_level(sys.argv[1])
        t = time.time(); r = analyze(gdnet.decode_level_string(s))
        print(meta["name"], meta["difficulty"], f"({time.time() - t:.1f}s)")
        print(json.dumps(r["stats"], indent=1))
        print({k: round(v, 3) for k, v in r["features"].items()})
    else:
        _demo()
