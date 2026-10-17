"""Offline self-check: python test_starmeter.py  (no network, no GD install needed)."""
import base64, gzip, json, os, tempfile, threading, time, urllib.error, urllib.request
from http.server import ThreadingHTTPServer

import app, gdlevel, gdnet, model


def level(spikes=20, gap=150, speed=0):
    objs = ";".join(f"1,8,2,{600 + gap * i},3,15" for i in range(spikes))
    return f"kA2,0,kA4,{speed};{objs};1,13,2,{600 + gap * spikes},3,165;" + ";".join(f"1,1,2,{800 + gap * spikes + 30 * i},3,{15 if i % 2 else 315}" for i in range(40))


def pack(text):
    return base64.urlsafe_b64encode(gzip.compress(text.encode())).decode()


def test_decode_roundtrip():
    t = level()
    assert gdnet.decode_level_string(pack(t)) == t and gdnet.decode_level_string(t) == t
    for junk in ("just some words pasted by mistake", "AAAA" * 20):
        try:
            gdnet.decode_level_string(junk)
            assert False, "junk must be rejected"
        except gdnet.GDError:
            pass


def test_local_save_roundtrip():
    lv = pack(level())
    xml = f"<?xml version='1.0'?><plist><dict><k>LLM_01</k><d><k>k_0</k><d><k>k2</k><s>My Level</s><k>k4</k><s>{lv}</s></d></d></dict></plist>"
    enc = bytes(b ^ 11 for b in base64.urlsafe_b64encode(gzip.compress(xml.encode())))
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dat") as f:
        f.write(enc)
    try:
        got = gdnet.read_local_levels(f.name)
        assert got[0]["name"] == "My Level" and gdnet.decode_level_string(got[0]["data"]) == level()
    finally:
        os.unlink(f.name)


def test_timeline_and_speed():
    slow, fast = gdlevel.analyze(level())["features"], gdlevel.analyze(level(speed=4))["features"]
    assert fast["len_s"] < slow["len_s"] * 0.7                      # 4x speed shortens the level
    assert slow["n_haz"] == 20 and slow["share_ship"] > 0.05        # hazards and the ship portal were found


def test_hostile_coordinates():
    import time
    t = time.time()
    f = gdlevel.analyze("kA2,0;1,8,2,100,3,15;1,8,2,1000000000,3,15;1,8,2,nan,3,inf;1,8,2,500,3,99999999")["features"]
    assert time.time() - t < 2 and f["n_haz"] == 1                    # absurd coordinates are dropped, not allocated for


def test_denser_level_rates_harder():
    def cube(every, per):                                            # same 40 s cube level, spikes in groups of `per`
        objs = [f"1,8,2,{600 + k * every + 30 * j},3,15" for k in range((12000 - 600) // every) for j in range(per)]
        return "kA2,0,kA4,0;" + ";".join(objs) + ";1,1,2,12000,3,15"
    easy, hard = (model.predict(gdlevel.analyze(cube(e, p))["features"])[0] for e, p in ((900, 1), (150, 3)))
    assert hard > easy, (easy, hard)                                 # many spike clusters must not look easier than a few


def _sim():
    """The physics modules, or None when numba is not installed (the app then runs without the physics pass)."""
    try:
        import gdsim, simfeat
        return gdsim, simfeat
    except ImportError:
        return None


def test_sim_physics():
    mods = _sim()
    if not mods:
        print("     (numba missing: physics checks skipped)")
        return
    import numpy as np
    gdsim, _ = mods
    hdr, objs, aux = gdsim.parse_level("kA2,0,kA4,0;1,1,2,100,3,15;1,1,2,5000,3,15")        # an empty world
    w = gdsim.build_world(hdr, objs, aux, gdlevel)
    G = gdsim

    def run(mode, hold, ticks, spd=0, mini=0, y0=15.0, gnd=1.0, flo=0.0, cei=1e9):
        p = np.zeros(G.NS); p[G.Y] = y0; p[G.UP] = 1.0; p[G.GND] = gnd; p[G.PAD] = -1.0; p[G.PAGE] = 99.0
        ys, vys, x, prev = [], [], 200.0, 0
        for k in range(ticks):
            h = hold(k)
            assert G.step_tick(p, w["GT"], w["col0"], w["ncols"], x, mode, mini, spd, flo, cei, h, 1 if (h and not prev) else 0, 0.0) == 0
            prev, x = h, x + G.VX[spd] * G.DT
            ys.append(p[G.Y]); vys.append(p[G.VY])
        return np.array(ys), np.array(vys)

    ys, _ = run(G.CUBE, lambda k: k < 1, 400, gnd=2.0)                         # one cube jump at 1x speed, from standing
    air = next(i for i in range(5, len(ys)) if abs(ys[i] - 15) < 1e-6) * G.DT
    # measured in the game (tools/oracle.py): peak 63.8 above the ground, back down after about 25.7 frames
    assert 63.0 < ys.max() - 15 < 64.6 and 25.0 < air < 26.5, (ys.max() - 15, air)
    ys, vys = run(G.SHIP, lambda k: True, 600, y0=100, gnd=0, cei=300)
    assert abs(vys.max() - 8.0) < 0.05 and ys[-1] <= 285.01                      # ship: rise capped at 8, stops at the ceiling
    _, vys = run(G.SHIP, lambda k: False, 600, y0=250, gnd=0, cei=300)
    assert abs(vys.min() + 6.4) < 0.05                                            # fall capped at -6.4
    for mini, slope in ((0, 1.0), (1, 2.0)):
        ys, _ = run(G.WAVE, lambda k: True, 40, y0=150, gnd=0, cei=300, mini=mini)
        assert abs((ys[-1] - ys[0]) / (39 * G.DT * G.VX[0]) - slope) < 0.03          # wave: 45 degrees, mini twice as steep
    tap, _ = run(G.ROBOT, lambda k: k < 2, 400)
    held, _ = run(G.ROBOT, lambda k: k < 80, 400)
    assert held.max() > 2 * tap.max() > 0                                          # robot: hold to climb higher
    ys, _ = run(G.BALL, lambda k: k == 0, 300, cei=300)
    assert ys[-1] > 280                                                            # ball: a click sends it to the other surface


def test_sim_scale_and_flags():
    mods = _sim()
    if not mods:
        return
    gdsim, _ = mods
    def spike_w(extra):
        hdr, objs, aux = gdsim.parse_level(f"kA2,0,kA4,0;1,8,2,1000,3,15{extra};1,1,2,3000,3,15")
        g = gdsim.build_world(hdr, objs, aux, gdlevel)["GT"]
        return (g[10] - g[9]) if len(g[9]) else 0.0
    assert abs(spike_w("") - 6) < 1e-6 and abs(spike_w(",128,0.5,129,0.5") - 3) < 1e-6        # 2.2 scale keys shrink the hitbox
    assert spike_w(",121,1") == 0 and abs(spike_w(",135,1") - 6) < 1e-6                       # 'no touch' has no hitbox; 'hide' still kills (checked in the game)


def test_sim_teleport_and_rotate():
    mods = _sim()
    if not mods:
        return
    import numpy as np
    gdsim, _ = mods

    def fly(level, ticks=200):
        """Let the cube run without input; -> list of (x, y) after each tick, or None at a death."""
        hdr, objs, aux = gdsim.parse_level(level)
        w = gdsim.build_world(hdr, objs, aux, gdlevel)
        sch = gdsim.build_schedule(w)
        p, out = gdsim.start_state(w, sch), []
        for tk in range(min(ticks, sch["T"])):
            for e in range(sch["ev_ptr"][tk], sch["ev_ptr"][tk + 1]):
                gdsim.apply_event(p, sch["ev_type"][e], sch["ev_val"][e])
            if gdsim.step_tick(p, w["GT"], w["col0"], w["ncols"], sch["x"][tk + 1], sch["mode"][tk], sch["mini"][tk], sch["spd"][tk],
                               sch["floor"][tk], sch["ceil"][tk], 0, 0, 0.0):
                return None
            out.append((sch["x"][tk + 1], p[gdsim.Y]))
        return out
    far = ";1,1,2,30000,3,3000"
    # a teleport portal sends the player 150 up on the tick its box is touched (measured in the game: tick 143, x 185.6)
    ys = fly("kA2,0,kA4,0;1,747,2,200,3,15,54,150" + far)
    jump = next(i for i in range(1, len(ys)) if ys[i][1] - ys[i - 1][1] > 100)
    assert abs(ys[jump][1] - 165.0) < 1e-6 and abs(ys[jump][0] - 185.6) < 1.5, ys[jump]
    # a rotate trigger swings a block 180 degrees around its center group into the player's path (game: death); in place: no change
    assert fly("kA2,0,kA4,0;1,1,2,330,3,-105,57,8;1,1,2,330,3,-45,57,9,121,1;1,1346,2,60,3,300,51,8,71,9,68,180,10,0" + far, 400) is None
    assert fly("kA2,0,kA4,0;1,1,2,330,3,-45,128,4,57,8;1,1346,2,60,3,300,51,8,68,90,10,0" + far, 400) is not None


def test_sim_planner_and_noise():
    mods = _sim()
    if not mods:
        return
    gdsim, simfeat = mods
    f = simfeat.extract(level(spikes=20, gap=150))                      # spikes every 150 units: easy jumps
    assert f["sim_ok"] == 1 and f["sim_forced"] == 0 and 0 < f["sim_toggle_rate"] < 4, f
    wall = ";".join(f"1,1,2,900,3,{15 + 30 * i}" for i in range(12))      # a 12-block wall nobody can climb
    g = simfeat.extract("kA2,0,kA4,0;" + wall + ";1,8,2,1500,3,15;1,1,2,3000,3,15")
    assert g["sim_forced"] > 0 and g["sim_forced_frac"] > 0                 # reported as a spot the model could not pass
    def spikes(xs):
        return "kA2,0,kA4,0;" + ";".join(f"1,8,2,{x},3,15" for x in xs) + ";1,1,2,3500,3,15"
    one = simfeat.extract(spikes([600 + 300 * i for i in range(10)]))
    three = simfeat.extract(spikes([600 + 300 * i + 30 * j for i in range(10) for j in range(3)]))      # triple spikes demand precision
    assert three["sim_mc15_sum"] > 2 * one["sim_mc15_sum"] and three["sim_mc60_sum"] >= three["sim_mc15_sum"]


def test_http_api():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), app.H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        req = urllib.request.Request(base + "/api/paste", json.dumps({"data": pack(level()), "name": "Synthetic"}).encode(),
                                     {"Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(req))
        assert d["source"]["name"] == "Synthetic" and 0.5 <= d["rating"]["star"] <= 21 and d["timeline"]["inputs"]
        gmd = f"<?xml version='1.0'?><d><k>kCEK</k><i>4</i><k>k2</k><s>From File</s><k>k4</k><s>{pack(level())}</s></d>"
        req = urllib.request.Request(base + "/api/paste", json.dumps({"data": gmd, "file": True}).encode(), {"Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(req))
        assert d["source"]["name"] == "From File" and d["source"]["kind"] == "file"       # a .gmd export is unpacked
        assert "<title>" in urllib.request.urlopen(base + "/").read().decode()
        for url, headers, code in ((base + "/api/rate?id=abc", {}, 400), (base + "/", {"Host": "evil.example"}, 403)):
            try:
                urllib.request.urlopen(urllib.request.Request(url, headers=headers))
                assert False, f"{url} must be rejected"
            except urllib.error.HTTPError as e:
                assert e.code == code, (url, e.code)
    finally:
        srv.shutdown()


def test_http_sim_flow():
    if not _sim():
        return
    srv = ThreadingHTTPServer(("127.0.0.1", 0), app.H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        req = urllib.request.Request(base + "/api/paste", json.dumps({"data": pack(level()), "name": "Synthetic"}).encode(), {"Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(req))
        assert d["sim"]["status"] == "pending"                                       # the first answer does not wait for the physics pass
        d2 = json.load(urllib.request.urlopen(base + "/api/sim?token=" + d["sim"]["token"], timeout=300))
        sm = d2["sim"]
        assert sm["status"] == "ok" and len(sm["risk"]) == len(sm["mode"]) == len(sm["unsure"]) > 5 and 0 <= sm["coverage"] <= 1, sm.keys()
        r = json.load(urllib.request.urlopen(base + "/api/replay?token=" + sm["token"]))
        assert len(r["x"]) == len(r["y"]) == len(r["hold"]) + 1 and r["hazards"] and r["solids"]
        for url in ("/api/sim?token=nope", "/api/replay?token=nope"):
            try:
                urllib.request.urlopen(base + url)
                assert False, url
            except urllib.error.HTTPError as e:
                assert e.code == 404, (url, e.code)
    finally:
        srv.shutdown()


def test_live_follow():
    """The Starmeter Live mod writes current.json; the server reads it and rates the level it names."""
    srv = ThreadingHTTPServer(("127.0.0.1", 0), app.H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    old_file, old_online = app.LIVE_FILE, app.rate_online
    app.live_rate.last = (None, None)
    get = lambda path: json.load(urllib.request.urlopen(base + path, timeout=120))
    with tempfile.TemporaryDirectory() as tmp:
        live = os.path.join(tmp, "mod", "current.json")
        os.makedirs(os.path.dirname(live))
        game = os.path.join(tmp, "game")
        os.makedirs(os.path.join(game, "Resources", "levels"))
        open(os.path.join(game, "GeometryDash.exe"), "wb").close()
        open(os.path.join(game, "Resources", "levels", "1.txt"), "w").write(pack(level()))
        write = lambda d: open(live, "w").write(json.dumps(d))
        app.LIVE_FILE = live
        try:
            assert get("/api/live")["present"] is False                                            # no mod file yet
            open(live, "w").write("{not json")
            assert get("/api/live")["present"] is False                                            # half-written or damaged file
            write(dict(v=1, seq=1, ts=time.time() * 1000, kind="official", id=1, name="Stereo Madness", creator="RobTop", stars=1, demon=False, gameDir=game))
            st = get("/api/live")
            assert st["present"] and st["kind"] == "official" and st["seq"] == 1 and "game_dir" not in st and st["age"] < 60, st
            d = get("/api/live/rate")                                                              # official level: read from the game folder
            assert d["source"]["kind"] == "official" and d["live"]["seq"] == 1 and 0.5 <= d["rating"]["star"] <= 21 and d["community"]["star"] == 1.0
            write(dict(seq=2, ts=0, kind="official", id=1, name="x", gameDir=os.path.join(tmp, "elsewhere")))
            try:
                get("/api/live/rate")
                assert False, "a game folder without GeometryDash.exe must be refused"
            except urllib.error.HTTPError as e:
                assert e.code in (404, 502), e.code
            open(os.path.join(os.path.dirname(live), "level.txt"), "w").write(pack(level()))
            write(dict(seq=3, ts=time.time() * 1000, kind="local", id=0, name="My level", data=True))
            d = get("/api/live/rate")                                                              # your own level: data written by the mod
            assert d["source"]["kind"] == "local" and d["source"]["name"] == "My level" and d["live"]["seq"] == 3
            seen = []
            app.rate_online = lambda i: seen.append(i) or dict(source=dict(kind="online", id=i), rating={}, timeline={}, stats={}, warnings=[], sim=dict(status="off"))
            write(dict(seq=4, ts=time.time() * 1000, kind="online", id=10565740, name="Bloodbath"))
            assert get("/api/live/rate")["live"]["id"] == 10565740 and seen == [10565740]          # online levels go through the normal ID path
            get("/api/live/rate")
            assert seen == [10565740]                                                              # the page and the mod both ask: rated once per announcement
        finally:
            app.LIVE_FILE, app.rate_online = old_file, old_online
            srv.shutdown()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
    print("all checks passed")
