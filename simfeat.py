"""Simulation features: play the level with the physics engine in gdsim.py and measure what the route demands.

    feats = simfeat.extract(level_text)     # dict of numbers, or None if the level cannot be simulated

The planner finds the route with the fewest input changes, then repeats the run with every hazard fattened by a few
units to see how much room that route has. Where the model cannot find any way through (a mechanic it does not know,
a moving platform...) the run is forced past the spot so the rest of the level still counts, and the forced share is
reported so the model can discount the other numbers.
"""
import math

import numpy as np

import gdlevel
import gdsim

FPS = 60.0
GROUP = {0: 0, 5: 0, 6: 0, 2: 0, 7: 0, 1: 1, 3: 1, 4: 2}              # cube-like / ship+ufo / wave
KEEP = 20                                                             # frames around a forced pass that are not trusted


def _windows(ev, n, w):
    """Max and 90th percentile of the event count in sliding windows of w frames."""
    if n < w or not len(ev):
        return float(len(ev)), float(len(ev))
    c = np.convolve(ev.astype(np.float32), np.ones(w, np.float32), mode="valid")
    return float(c.max()), float(np.percentile(c, 90))


def _route_feats(prefix, r, sch, ok):
    """Features of one planner run. `ok` marks the frames that are not near a forced pass."""
    inp = (r["inputs"] & 1).astype(np.int8)                          # low bit: hold; the rest is the sub-frame switch tick
    n = len(inp)
    prev = np.concatenate(([0], inp[:-1]))
    press, release = (inp == 1) & (prev == 0), (inp == 0) & (prev == 1)
    toggle = press | release
    mode = sch["mode"][::4][:n]
    grp = np.array([GROUP.get(int(m), 0) for m in mode])
    cnt = np.log2(np.maximum(r["counts"][:n], 1).astype(np.float64) + 1.0)
    out = {}
    use = ok[:n] if ok[:n].any() else np.ones(n, bool)
    secs = max(use.sum() / FPS, 1e-3)
    out[prefix + "press_rate"] = float((press & use).sum() / secs)
    out[prefix + "toggle_rate"] = float((toggle & use).sum() / secs)
    out[prefix + "hold_frac"] = float(inp[use].mean())
    mx, p90 = _windows((toggle & use), n, int(FPS))
    out[prefix + "toggle_max1s"] = mx
    out[prefix + "toggle_p90_1s"] = p90
    mx5, p905 = _windows((toggle & use), n, int(5 * FPS))
    out[prefix + "toggle_p90_5s"] = p905 / 5.0
    idx = np.where(toggle & use)[0]
    gaps = np.diff(idx) if len(idx) > 3 else np.array([60.0])
    out[prefix + "gap_p10"] = float(np.percentile(gaps, 10))
    out[prefix + "gap_med"] = float(np.median(gaps))
    out[prefix + "gap_lt6"] = float((gaps < 6).mean())               # toggles within 0.1 s of the previous one
    out[prefix + "cnt_mean"] = float(cnt[use].mean())
    out[prefix + "cnt_p10"] = float(np.percentile(cnt[use], 10))
    out[prefix + "cnt_lt2"] = float((cnt[use] < 2.0).mean())         # frames with at most one surviving line
    for g, name in ((0, "cube"), (1, "fly"), (2, "wave")):
        m = use & (grp == g)
        sec = m.sum() / FPS
        out[f"{prefix}{name}_toggle_rate"] = float((toggle & m).sum() / sec) if sec > 1.0 else 0.0
        out[f"{prefix}{name}_cnt_mean"] = float(cnt[m].mean()) if sec > 1.0 else 0.0
        out[f"{prefix}{name}_cnt_lt2"] = float((cnt[m] < 2.0).mean()) if sec > 1.0 else 0.0
    return out


def replay_payload(w, sch, r0, max_rects=120000):
    """Everything the browser needs to play the bot's route back: the path per 60 Hz step plus the level's hitboxes."""
    inputs = r0["inputs"]
    n = len(inputs)
    forced = r0["fails"][:n].astype(np.uint8)
    args = (w["GT"], w["col0"], w["ncols"], sch["T"], sch["x"], sch["spd"], sch["mode"], sch["mini"], sch["floor"], sch["ceil"],
            sch["ev_ptr"], sch["ev_type"], sch["ev_val"])
    S = gdsim.replay(*args, gdsim.start_state(w, sch), inputs, forced)
    ticks = np.minimum(np.arange(n + 1) * 4, sch["T"])
    r1 = lambda a: np.round(np.asarray(a, np.float64), 1).tolist()
    i0 = lambda a: np.round(np.asarray(a, np.float64)).astype(int).tolist()
    GT = w["GT"]
    ys = S[:n + 1, gdsim.Y]
    lo, hi = float(ys.min()) - 400.0, float(ys.max()) + 400.0
    keep = lambda y0, y1: (np.asarray(y1) > lo) & (np.asarray(y0) < hi)
    ks = np.where(keep(GT[2], GT[3]))[0][:max_rects]
    kh = np.where(keep(GT[11], GT[12]))[0][:max_rects]
    ko = np.where((GT[20] > lo) & (GT[20] < hi))[0]
    kp = np.where(keep(GT[27], GT[28]))[0]
    mode = sch["mode"][ticks[:-1]]
    return dict(
        x=r1(sch["x"][ticks]), y=r1(ys), up=[int(v) for v in S[:n + 1, gdsim.UP]], mode=[int(v) for v in mode],
        mini=[int(v) for v in sch["mini"][ticks[:-1]]], hold=[int(v) & 1 for v in inputs], forced=[int(v) for v in forced],
        floor=i0(sch["floor"][ticks[:-1]]), ceil=[int(min(v, 99999)) for v in i0(sch["ceil"][ticks[:-1]])],
        solids=[[*i0([GT[0][j], GT[1][j], GT[2][j], GT[3][j]]), int(GT[4][j]), int(GT[5][j])] for j in ks],
        hazards=[[int(GT[8][j]), *i0([GT[9][j], GT[10][j], GT[11][j], GT[12][j], GT[13][j], GT[14][j], GT[15][j]])] for j in kh],
        orbs=[[*i0([GT[19][j], GT[20][j]]), int(GT[18][j])] for j in ko],
        pads=[[*i0([GT[25][j], GT[26][j], GT[27][j], GT[28][j]]), int(GT[24][j])] for j in kp])


def analyze(text, kmax=300, sigmas=(1.5, 3.0, 6.0), trials=24, seg=30, replay=False):
    """-> (features, detail). `detail` holds the per-segment arrays the interface draws (None when nothing was simulated)."""
    hdr, objs, aux = gdsim.parse_level(text)
    if hdr.get("kA22") == "1":                                       # platformer: no fixed route to play
        return dict(sim_ok=0.0), None
    w = gdsim.build_world(hdr, objs, aux, gdlevel)
    if w["x_end"] < 600 or not w["counts"]["hazard"] + w["counts"]["solid"]:
        return dict(sim_ok=0.0), None
    sch = gdsim.build_schedule(w)
    if sch["T"] < 240:
        return dict(sim_ok=0.0), None
    r0 = gdsim.plan(w, sch, eps=0.0, kmax=kmax)
    if r0["cost"] < 0:
        return dict(sim_ok=0.0), None
    n = len(r0["inputs"])
    fails = r0["fails"][:n].astype(bool)
    bad = np.zeros(n, bool)
    for i in np.where(fails)[0]:
        bad[max(0, i - KEEP):i + KEEP] = True
    ok = ~bad
    f = dict(sim_ok=1.0, sim_steps=float(n), sim_len_s=n / FPS, sim_forced=float(fails.sum()),
             sim_forced_frac=float(bad.mean()), sim_first_fail=float(np.argmax(fails)) / n if fails.any() else 1.0,
             sim_dropped=float(r0["dropped"]), sim_unsup=float(w["unsupported"]),
             sim_n_solid=float(w["counts"]["solid"]), sim_n_haz=float(w["counts"]["hazard"]))
    f.update(_route_feats("sim_", r0, sch, ok))
    cl = gdsim.route_clearance(w, sch, r0)
    use = ok[:n]
    secs_ok = max(use.sum() / FPS, 1.0)
    for k, name in ((1, "13"), (2, "9"), (3, "6"), (4, "4"), (5, "2"), (6, "1")):
        f["sim_clr" + name] = float(((cl >= k) & use).sum() / max(use.sum(), 1))              # share of the route with something this close
    near = (cl >= 3) & use                                                                        # within 6 units
    edges = near & ~np.concatenate(([False], near[:-1]))
    f["sim_close_rate"] = float(edges.sum() / secs_ok)                                            # close passes per second
    runs, cur = [0], 0
    for v in (cl >= 4) & use:                                                                     # within 4 units, longest unbroken stretch
        cur = cur + 1 if v else 0
        runs.append(cur)
    f["sim_close_run_s"] = max(runs) / FPS
    modes_n = sch["mode"][::4][:n]
    for g, name in ((0, "cube"), (1, "fly"), (2, "wave")):
        m = use & np.array([GROUP.get(int(x), 0) == g for x in modes_n])
        f[f"sim_clr4_{name}"] = float(((cl >= 4) & m).sum() / m.sum()) if m.sum() > FPS else 0.0
    prof = gdsim.jitter_profile(w, sch, r0, sigmas=sigmas, trials=trials, seg=seg)
    nseg = len(next(iter(prof.values())))
    okseg = np.array([ok[i * seg:(i + 1) * seg].all() for i in range(nseg)])
    secs = max(okseg.sum() * seg / FPS, 1.0)
    mode_seg = sch["mode"][::4][:n][::seg][:nseg]
    for sg, pf in prof.items():
        p = np.where(okseg, pf, 0.0)
        tag = f"sim_mc{str(sg).replace('.', '')}_"
        f[tag + "sum"] = float(p.sum())
        f[tag + "rate"] = float(p.sum() / secs)                       # expected deaths per second of route
        f[tag + "max"] = float(p.max())
        f[tag + "n10"] = float((p > 0.1).sum())
        f[tag + "n30"] = float((p > 0.3).sum())
        f[tag + "haz"] = float(-np.log1p(-np.minimum(p, 0.99)).sum())  # -log(survival of one clean pass)
        f[tag + "frac"] = float((p > 0.02).sum() / max(okseg.sum(), 1))
        for g, name in ((0, "cube"), (1, "fly"), (2, "wave")):
            m = okseg & np.array([GROUP.get(int(x), 0) == g for x in mode_seg])
            f[f"{tag}{name}_rate"] = float(p[m].sum() / (m.sum() * seg / FPS)) if m.sum() * seg / FPS > 1.0 else 0.0
    mid = sigmas[len(sigmas) // 2]
    risk = np.where(okseg, prof[mid], 0.0)
    detail = dict(step=seg / FPS, sigma=mid, risk=risk.tolist(), mode=[int(x) for x in mode_seg], unsure=(~okseg).tolist(),
                  coverage=float(ok.mean()), survive=float(np.exp(-f[f"sim_mc{str(mid).replace('.', '')}_haz"])),
                  route=dict(inputs=r0["inputs"], forced=fails, counts=r0["counts"][:n].astype(np.uint16), clear=cl))
    if replay:
        detail["replay"] = replay_payload(w, sch, r0)
    return f, detail


def extract(text, **kw):
    """Features only (what training and the model use)."""
    return analyze(text, **kw)[0]
