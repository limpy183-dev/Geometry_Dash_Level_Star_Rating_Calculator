"""Check gdsim.py against the real game: replay the bot's route inside Geometry Dash (tools/gd-oracle mod) and compare.

    python tools/oracle.py main 1 2 3          # official levels
    python tools/oracle.py db 12345 67890      # corpus levels from data/levels.db
    python tools/oracle.py report              # compare the last run's results with the simulator

For every level the planner's route is written as a job (press / release on exact 240 Hz ticks); the mod plays it in
the game, records the player every tick and where the real attempt ends. The report prints, per level, how far the
route got in the game, the first tick where the simulated and the real player differ by more than a few units, and the
object that killed the real player. The game is started through Steam and closes itself when the jobs are done; it
does not save, so attempts and stats in your save are untouched.
"""
import base64, gzip, json, os, shutil, sqlite3, subprocess, sys, time, zlib

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import gdlevel, gdnet, gdsim

SAVE = os.path.join(os.environ.get("LOCALAPPDATA", ""), "GeometryDash", "geode", "mods", "starmeter.oracle")
MAIN = r"C:\Program Files (x86)\Steam\steamapps\common\Geometry Dash\Resources\levels"
STATE = os.path.join(SAVE, "routes.json")
LAG = 0                                                              # ticks to delay each injected input (measured with the oracle)
Y0 = 90.0                                                            # the game's ground line: y = 90 (a cube standing on it is at 105)


def level_text(src, lid):
    if src == "main":
        return gdnet.decode_level_string(open(os.path.join(MAIN, f"{lid}.txt")).read())
    db = sqlite3.connect(os.path.join(ROOT, "data", "levels.db"))
    blob = db.execute("select ldata from levels where id=?", (int(lid),)).fetchone()[0]
    return zlib.decompress(blob, 47).decode("utf-8", "replace")


def prepare(text):
    hdr, objs, aux = gdsim.parse_level(text)
    w = gdsim.build_world(hdr, objs, aux, gdlevel)
    return w, gdsim.build_schedule(w)


def route_events(inputs):
    """Planner step codes -> [(tick, press?)]: the low bit is the hold flag, the rest the tick inside the step it changes on."""
    ev, prev = [], 0
    for s, code in enumerate(inputs):
        h, k = int(code) & 1, int(code) >> 1
        if h != prev:
            ev.append((s * 4 + k, h == 1))
        prev = h
    return ev


def sim_ticks(w, sch, inputs):
    """Replay the route one physics tick at a time: rows (tick, x, y, vy, up, alive)."""
    p = gdsim.start_state(w, sch)
    out, prev, T = [], 0, sch["T"]
    for s, code in enumerate(inputs):
        u, k = int(code) & 1, int(code) >> 1
        for tk in range(s * 4, min(s * 4 + 4, T)):
            for e in range(sch["ev_ptr"][tk], sch["ev_ptr"][tk + 1]):
                gdsim.apply_event(p, sch["ev_type"][e], sch["ev_val"][e])
            ut = prev if tk - s * 4 < k else u
            press = 1 if (u == 1 and prev == 0 and tk - s * 4 == k) else 0
            code_ = gdsim.step_tick(p, w["GT"], w["col0"], w["ncols"], sch["x"][tk + 1], sch["mode"][tk], sch["mini"][tk], sch["spd"][tk],
                                    sch["floor"][tk], sch["ceil"][tk], ut, press, 0.0)
            out.append((tk + 1, sch["x"][tk + 1], p[gdsim.Y], p[gdsim.VY], p[gdsim.UP], code_ == 0))
            if code_:
                return out
        prev = u
    return out


def write_jobs(src, ids):
    jobs = os.path.join(SAVE, "jobs")
    shutil.rmtree(jobs, ignore_errors=True); shutil.rmtree(os.path.join(SAVE, "results"), ignore_errors=True)
    os.makedirs(jobs)
    try:
        os.remove(os.path.join(SAVE, "done.txt"))
    except OSError:
        pass
    routes = {}
    for lid in ids:
        text = level_text(src, lid)
        w, sch = prepare(text)
        r = gdsim.plan(w, sch, kmax=300)
        name = f"{src}_{lid}"
        lines = [f"level main {lid}" if src == "main" else f"level file {name}.lvl"]
        if src != "main":                                            # the game expects the compressed form
            open(os.path.join(jobs, name + ".lvl"), "w").write(base64.urlsafe_b64encode(gzip.compress(text.encode())).decode())
        lines += [f"{'press' if d else 'release'} {t + LAG}" for t, d in route_events(r["inputs"])]
        open(os.path.join(jobs, name + ".job"), "w").write("\n".join(lines) + "\n")
        routes[name] = dict(src=src, id=lid, inputs=r["inputs"].tolist(), forced=int(r["fails"].sum()))
        print(f"{name}: {len(r['inputs'])} steps, {int(r['fails'].sum())} forced", flush=True)
    json.dump(routes, open(STATE, "w"))


GAME = r"C:\Program Files (x86)\Steam\steamapps\common\Geometry Dash"
BUILT = os.path.join(ROOT, "tools", "gd-oracle", "build", "starmeter.oracle.geode")
QOL = os.path.join(os.environ.get("LOCALAPPDATA", ""), "GeometryDash", "geode", "mods", "thesillydoggo.qolmod", "saved.json")


def _gd_running():
    return b"geometrydash.exe" in subprocess.run(["tasklist"], capture_output=True).stdout.lower()


def run_game(timeout=1800):
    """Install the oracle mod, switch QOLMod's noclip off, play the jobs, then put everything back exactly as it was:
    the mod is removed again and the QOLMod settings file is restored byte for byte."""
    if _gd_running():
        raise SystemExit("Geometry Dash is running: close it first (the oracle needs its own session)")
    mod = os.path.join(GAME, "geode", "mods", "starmeter.oracle.geode")
    backup = os.path.join(SAVE, "qolmod_saved.backup")
    if os.path.exists(backup):                                       # an earlier run was cut off: put the user's file back first
        shutil.copy(backup, QOL)
    qol = open(QOL, "rb").read() if os.path.exists(QOL) else None
    if qol is not None:
        open(backup, "wb").write(qol)
    try:
        shutil.copy(BUILT, mod)
        if qol is not None:                                          # noclip stays off: it would hide every death
            d = json.loads(qol)
            d["noclip_enabled"] = False
            open(QOL, "w", encoding="utf-8").write(json.dumps(d))
        try:
            os.remove(os.path.join(SAVE, "done.txt"))                  # a leftover from the last run would end this one at once
        except OSError:
            pass
        open(os.path.join(SAVE, "start.flag"), "w").write("1")
        subprocess.Popen(["cmd", "/c", "start", "", "steam://rungameid/322170"])
        t0 = time.time()
        while not _gd_running() and time.time() - t0 < 120:          # Steam takes a moment to start the game
            time.sleep(1)
        while not os.path.exists(os.path.join(SAVE, "done.txt")) and time.time() - t0 < timeout:
            time.sleep(3)
        ok = os.path.exists(os.path.join(SAVE, "done.txt"))
        while _gd_running() and time.time() - t0 < timeout + 60:     # the mod quits the game itself
            time.sleep(2)
        if _gd_running():
            subprocess.run(["taskkill", "/IM", "GeometryDash.exe", "/F"], capture_output=True)
            time.sleep(3)
        return ok
    finally:
        for f in (mod, os.path.join(SAVE, "start.flag")):
            try:
                os.remove(f)
            except OSError:
                pass
        shutil.rmtree(os.path.join(SAVE, "jobs"), ignore_errors=True)
        if qol is not None:
            open(QOL, "wb").write(qol)
            os.remove(backup)


def report(tol=3.0):
    routes = json.load(open(STATE))
    for name, rt in routes.items():
        f = os.path.join(SAVE, "results", name + ".txt")
        if not os.path.exists(f):
            print(f"{name}: no result"); continue
        rows, end = [], None
        for line in open(f):
            a = line.split()
            if a[0] in ("DEAD", "COMPLETE", "PASS"):
                end = a
            else:
                rows.append([float(v) for v in a])
        real = np.array(rows)
        real[:, 2] -= Y0
        w, sch = prepare(rt["text"] if "text" in rt else level_text(rt["src"], rt["id"]))
        sim = np.array(sim_ticks(w, sch, np.array(rt["inputs"], np.uint8)))
        sim_end = f"sim dies at x={sim[-1, 1]:.0f}" if not sim[-1, 5] else "sim survives"
        # one row per physics tick on both sides: the game's row r is the state after simulated tick r. The game refreshes
        # the position every second tick, so the height is compared on fresh rows only; the velocity on every row.
        n = min(len(real) - 1, len(sim))
        R, Sm = real[1:n + 1], sim[:n]
        fresh = np.r_[True, np.diff(R[:, 1]) != 0]
        vbad = (np.abs(R[:, 3] - Sm[:, 3]) > 0.3) & (R[:, 4] != 4)  # the wave logs 5.193 for a 45-degree step: compare its height only
        bad = np.where(vbad | (fresh & (np.abs(R[:, 2] - Sm[:, 2]) > tol)))[0]
        first = bad[0] if len(bad) else None
        sim, real, off = Sm, R, 0
        x_end = w["x_end"]
        endtxt = (f"real {end[0].lower()} at x={float(end[2]):.0f} ({float(end[2]) / x_end:.0%}) by object {end[4]}" if end and end[0] == "DEAD"
                  else "real COMPLETE" if end else "real: no end")
        div = (f"first divergence at x={sim[first, 1]:.0f} ({sim[first, 1] / x_end:.0%}): sim y={sim[first, 2]:.1f} vy={sim[first, 3]:.2f}, "
               f"real y={real[first, 2]:.1f} vy={real[first, 3]:.2f} mode={int(real[first, 4])} (tick offset {off})" if first is not None else "no divergence")
        print(f"{name}: {endtxt}, {sim_end}; {div}")


def mode_check(names=None):
    """One-tick check per gamemode: start the simulator from the game's own state (height, velocity, gravity, size) at
    every tick, step once with the same input in an empty world, and compare the next velocity with the game's. Route
    differences cannot pile up this way, so what is left is each gamemode's own flight physics."""
    routes = json.load(open(STATE))
    hdr, objs, aux = gdsim.parse_level("kA2,0,kA4,0;1,1,2,-3000,3,-3000")
    W = gdsim.build_world(hdr, objs, aux, gdlevel)                   # nothing to collide with
    errs = {}
    for name, rt in routes.items():
        if names and name not in names:
            continue
        f = os.path.join(SAVE, "results", name + ".txt")
        if not os.path.exists(f):
            continue
        real = np.array([[float(v) for v in l.split()] for l in open(f) if l[0].isdigit()])
        w, sch = prepare(rt["text"] if "text" in rt else level_text(rt["src"], rt["id"]))
        hold = np.zeros(len(real) + 2, np.int64)
        for t, d in route_events(np.array(rt["inputs"], np.uint8)):
            hold[t + LAG:] = 1 if d else 0
        for r in range(1, len(real) - 1):
            a0, a1 = real[r], real[r + 1]
            m = int(a0[4])
            if m != int(a1[4]) or r >= sch["T"]:                     # skip portal ticks
                continue
            fl, ce = sch["floor"][r], sch["ceil"][r]
            if min(a0[2], a1[2]) - Y0 < fl + 31 or (ce < 1e8 and max(a0[2], a1[2]) - Y0 > ce - 31) or (m in (0, 2, 5, 6) and (a0[6] or a1[6])):
                continue                                             # near a floor / ceiling, or standing: contact, not flight
            p = np.zeros(gdsim.NS); p[gdsim.PAD] = -1.0; p[gdsim.TEL] = -1.0; p[gdsim.PAGE] = 99.0
            p[gdsim.Y] = a0[2] - Y0; p[gdsim.UP] = -1.0 if a0[5] else 1.0
            p[gdsim.VY] = a0[3]                                      # the game logs world velocity (up is positive)
            u, pr = int(hold[r]), int(hold[r] and not hold[r - 1])
            gdsim.step_tick(p, W["GT"], W["col0"], W["ncols"], 0.0, m, int(a0[7]), int(sch["spd"][r]), -1e9, 1e9, u, pr, 0.0)
            if m == 4:                                               # the wave logs 5.193 but moves 45 degrees: compare the step
                pred, game = (p[gdsim.Y] + Y0 - a0[2]), (a1[2] - a0[2])
                if a1[1] == a0[1] or r < 2 or a0[1] == real[r - 1][1]:
                    continue                                         # compare only one-tick steps between refreshed rows
            else:
                pred, game = p[gdsim.VY], a1[3]
            errs.setdefault(m, []).append((pred - game, name, r, u, pr, a0[3], game, pred))
    modes = ["cube", "ship", "ball", "ufo", "wave", "robot", "spider", "swing"]
    for m, e in sorted(errs.items()):
        d = np.array([x[0] for x in e])
        big = [x for x in e if abs(x[0]) > 0.05]
        print(f"{modes[m]:7s} ticks {len(d):6d}  median |err| {np.median(np.abs(d)):.4f}  off by >0.05: {len(big) / len(d):.1%}")
        for x in big[:6]:
            print(f"        {x[1]} tick {x[2]} hold {x[3]} press {x[4]}: game vy {x[5]:.3f} -> {x[6]:.3f}, sim -> {x[7]:.3f}")
    return errs


MODE_PORTAL = {1: 13, 2: 47, 3: 111, 4: 660, 5: 745, 6: 1331, 7: 1933}    # gamemode -> portal id
SPEED_PORTAL = {1: 200, 0: 201, 2: 202, 3: 203}                            # speed index -> portal id


def write_synth(ticks=2400, seed=7):
    """Empty corridors to test the flight physics of each gamemode: a speed portal, a size portal for mini, a gamemode
    portal, then nothing to hit, and a random pattern of holds and releases (no deaths possible, so no noclip needed)."""
    rng = np.random.default_rng(seed)
    jobs = os.path.join(SAVE, "jobs")
    shutil.rmtree(jobs, ignore_errors=True); shutil.rmtree(os.path.join(SAVE, "results"), ignore_errors=True)
    os.makedirs(jobs)
    head = level_text("main", 1).split(";")[0]
    routes = {}
    for mode, portal in MODE_PORTAL.items():
        for spd in (0, 2):
            for mini in (0, 1):
                # portals at ground height (the cube has to touch them); a block far away and high up sets the level length
                objs = [f"1,{SPEED_PORTAL[spd]},2,60,3,45", f"1,{portal},2,90,3,45"] + ([f"1,101,2,120,3,45"] if mini else [])
                objs.append("1,1,2,30000,3,3000")
                text = f"{head};" + ";".join(objs) + ";"
                steps = ticks // 4
                inputs, h, k = np.zeros(steps, np.uint8), 0, 0
                while k < steps:                                       # random holds / releases, 3 to 40 steps long
                    n = int(rng.integers(3, 40))
                    inputs[k:k + n] = h
                    h, k = 1 - h, k + n
                inputs[:30] = 0                                        # no input before the portals
                name = f"synth_{mode}_{spd}_{mini}"
                open(os.path.join(jobs, name + ".lvl"), "w").write(base64.urlsafe_b64encode(gzip.compress(text.encode())).decode())
                lines = [f"level file {name}.lvl", f"until {sch_x(text, ticks - 8):.1f}"]
                lines += [f"{'press' if d else 'release'} {t + LAG}" for t, d in route_events(inputs)]
                open(os.path.join(jobs, name + ".job"), "w").write("\n".join(lines) + "\n")
                routes[name] = dict(src="synth", id=name, text=text, inputs=inputs.tolist(), forced=0)
    json.dump(routes, open(STATE, "w"))
    print(len(routes), "synthetic levels")


def sch_x(text, tick):
    w, sch = prepare(text)
    return float(sch["x"][min(tick, sch["T"])])


def write_probes(ids, until=300.0):
    """One tiny level per object ID: the object in the player's path at ground height, upright (x=165) and turned 90
    degrees (x=195). The game then says whether it kills (solid / hazard), lifts the player (slope) or lets it pass."""
    jobs = os.path.join(SAVE, "jobs")
    shutil.rmtree(jobs, ignore_errors=True); shutil.rmtree(os.path.join(SAVE, "results"), ignore_errors=True)
    os.makedirs(jobs)
    try:
        os.remove(os.path.join(SAVE, "done.txt"))
    except OSError:
        pass
    head = level_text("main", 1).split(";")[0]
    for oid in ids:
        text = f"{head};1,{oid},2,165,3,15;1,{oid},2,195,3,15,6,90;"
        name = f"probe_{oid:05d}"
        open(os.path.join(jobs, name + ".lvl"), "w").write(base64.urlsafe_b64encode(gzip.compress(text.encode())).decode())
        open(os.path.join(jobs, name + ".job"), "w").write(f"level file {name}.lvl\nuntil {until}\n")
    json.dump({}, open(STATE, "w"))


def probe_report():
    """-> {id: "kill" | "lift" | "pass" | "?"} from the last probe run."""
    out = {}
    for f in sorted(os.listdir(os.path.join(SAVE, "results"))):
        oid = int(f[6:11])
        rows = [l.split() for l in open(os.path.join(SAVE, "results", f))]
        end = [r for r in rows if r[0] in ("DEAD", "PASS", "COMPLETE")]
        ys = [float(r[2]) for r in rows if r[0][0].isdigit()]
        if not end:
            out[oid] = "?"
        elif end[-1][0] == "DEAD":
            out[oid] = "kill"
        else:
            out[oid] = "lift" if ys and max(ys) > Y0 + 16.0 else "pass"
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "report":
        report()
    elif cmd == "modes":
        mode_check(sys.argv[2:] or None)
    elif cmd == "synth":                                             # every gamemode / speed / size in empty corridors
        write_synth()
        print("game finished" if run_game(timeout=3600) else "timed out")
        mode_check()
    elif cmd == "probe":                                             # probe every object the game types solid or slope
        ids = [d["id"] for d in map(json.loads, open(os.path.join(ROOT, "data", "objdump.jsonl"))) if d["type"] in (0, 25)]
        ids += [1, 8, 1006]                                          # controls: a block (kill), a spike (kill), a trigger (pass)
        write_probes(sorted(set(ids)))
        print("game finished" if run_game(timeout=5400) else "timed out")
        res = probe_report()
        json.dump(res, open(os.path.join(ROOT, "data", "objprobe.json"), "w"), indent=0)
        import collections
        print(collections.Counter(res.values()), "controls:", {i: res.get(i) for i in (1, 8, 1006)})
    else:
        write_jobs(cmd, sys.argv[2:])
        print("game finished" if run_game() else "timed out")
        report()
