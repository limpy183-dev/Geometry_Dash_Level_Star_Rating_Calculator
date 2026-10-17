"""Starmeter - local Geometry Dash difficulty calculator.

    python app.py            # opens http://127.0.0.1:8765
    python app.py --port 9000 --no-browser
    python app.py --live-file C:\\path\\to\\current.json     # where the Starmeter Live mod writes (default: the Geode mod folder)

Binds to localhost only. Level IDs are fetched from the GD servers (and GDDL for the community rating);
pasted data, imported files and your local save never leave this computer.
"""
import collections, json, os, sys, threading, time, traceback, urllib.parse, uuid, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import gdlevel, gdnet, model

HERE = os.path.dirname(os.path.abspath(__file__))
WEB, CACHE = os.path.join(HERE, "web"), os.path.join(HERE, "data", "cache")
LOCK = threading.Lock()                       # one analysis at a time: bounded memory, and polite to the GD server
MAX_BODY = 60_000_000
REPLAYS = collections.OrderedDict()           # token -> the bot's route and the level's hitboxes, for the replay viewer
STASH = collections.OrderedDict()             # token -> (level text, source, extras): lets the physics pass run after the first answer
LIVE_FILE = os.environ.get("STARMETER_LIVE_FILE") or os.path.join(           # written by the "Starmeter Live" Geode mod
    os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "GeometryDash", "geode", "mods", "starmeter.live", "current.json")
LIVE_KINDS = ("online", "official", "local")


def community(meta):
    """What RobTop / the community say, shown next to the calculation (never fed into it)."""
    out = dict(official_stars=meta["stars"], face=meta["difficulty"])
    if meta["demon"]:
        try:
            g = gdnet.gddl_level(meta["id"])
        except gdnet.GDError:
            g = None
        if g:
            out.update(tier=round(g["tier"], 2), votes=g["votes"], star=round(model.tier_to_star(g["tier"]), 2))
    elif meta["stars"]:
        out["star"] = float(meta["stars"])
    return out


def sim_ready():
    """The physics pass needs numba (and numpy); the rating only reads it when a simulation-aware model is trained."""
    if "ready" not in sim_ready.__dict__:
        try:
            import simfeat                                    # noqa: F401  (imports numba)
            sim_ready.ready = True
        except Exception:
            sim_ready.ready = False
    return sim_ready.ready


def hardest(risk, step, mode, n=3, span=10):
    """The n worst non-overlapping stretches (span segments each) by expected failures."""
    out, taken = [], set()
    sums = [sum(risk[i:i + span]) for i in range(max(len(risk) - span + 1, 0))]
    for i in sorted(range(len(sums)), key=lambda k: -sums[k]):
        if len(out) >= n or sums[i] < 0.05:
            break
        if any(j in taken for j in range(i, i + span)):
            continue
        taken.update(range(i, i + span))
        mid = mode[min(i + span // 2, len(mode) - 1)]
        out.append(dict(t0=round(i * step, 1), t1=round((i + span) * step, 1), mode=mid, risk=round(sums[i], 2)))
    return sorted(out, key=lambda h: h["t0"])


def sim_block(f, detail, token=None):
    if not detail:
        return dict(status="none", note="There was no fixed route to play (an empty, decoration-only or platformer level), so the physics pass did not run.")
    return dict(status="ok", token=token, coverage=round(detail["coverage"], 3), forced=int(f["sim_forced"]), presses=round(f["sim_press_rate"], 2),
                toggles=round(f["sim_toggle_rate"], 2), busiest=int(f["sim_toggle_max1s"]), survive=round(detail["survive"], 4),
                sigma=detail["sigma"], step=round(detail["step"], 3), risk=[round(x, 3) for x in detail["risk"]], mode=detail["mode"],
                unsure=detail["unsure"], hardest=hardest(detail["risk"], detail["step"], detail["mode"]))


def run(text, source, simulate=False, extra=None, token=None):
    r = gdlevel.analyze(text)
    feats, detail = dict(r["features"]), None
    if simulate:
        import simfeat
        sf, detail = simfeat.analyze(text, replay=True)
        feats.update(sf)
    rating = model.rate(feats, source.get("id"))
    st, warn = r["stats"], []
    if source.get("difficulty") == "Auto":
        warn.append("This level is rated Auto: it plays itself, so its hazards say nothing about skill and the rating will read far too high.")
    if st["platformer"]:
        warn.append("Platformer level: there is no fixed route or timing, so this rating is unreliable.")
    if st["two_player"]:
        warn.append("Two-player level: it is rated as a one-player route, so real difficulty will differ.")
    if st["objects"] < 60:
        warn.append("Very small level: ratings for tiny levels are noisy.")
    if st["length_s"] < 8:
        warn.append("Under 8 seconds of gameplay was detected; the level may be incomplete.")
    if rating["in_training"]:
        warn.append("This level is part of the data the model learned from, so its estimate is more flattering than it would be for a new level.")
    if 0.3 < rating["p_demon"] < 0.7:
        warn.append("Borderline level: the model is split on whether this is a top-tier Insane or a demon, so the number sits between the two.")
    if rating["confidence"] == "low":
        warn.append("This level looks unlike the levels the model learned from, so treat the number with extra caution.")
    if simulate:
        sim = sim_block(feats, detail, token)
        if detail and detail.get("replay"):
            REPLAYS[token] = detail["replay"]
            while len(REPLAYS) > 3:
                REPLAYS.popitem(last=False)
        if sim["status"] == "ok" and sim["coverage"] < 0.7:
            warn.append(f"The simulator could only follow {round(sim['coverage'] * 100)}% of the route (gimmicks, moving parts or mechanics it doesn't model), so the physics numbers are partial.")
    elif sim_ready():
        token = uuid.uuid4().hex[:12]
        STASH[token] = (text, source, extra or {})
        while len(STASH) > 6:
            STASH.popitem(last=False)
        sim = dict(status="pending", token=token)
    else:
        sim = dict(status="off")
    return dict(source=source, rating=rating, timeline=r["timeline"], stats=st, warnings=warn, sim=sim, **(extra or {}))


def simulate(token):
    if token not in STASH:
        raise gdnet.GDError("That result expired. Rate the level again.")
    text, source, extra = STASH[token]
    return run(text, source, simulate=True, extra=extra, token=token)


def rate_online(level_id):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{level_id}.json")
    if os.path.exists(path):
        c = json.load(open(path))
    else:
        meta, s = gdnet.download_level(level_id)
        c = dict(meta=meta, data=s)
        json.dump(c, open(path, "w"))
    return run(gdnet.decode_level_string(c["data"]), dict(kind="online", **c["meta"]), extra=dict(community=community(c["meta"])))


def rate_pasted(body):
    data, name = (body.get("data") or "").strip(), (body.get("name") or "Pasted level")[:80]
    if "<k>k4</k>" in data:                                   # a whole exported .gmd file
        name, data = gdnet.read_gmd(data.encode())
    if len(data) < 20:
        raise gdnet.GDError("Paste a level string (the long text starting with H4sI...) or drop a .gmd file.")
    return run(gdnet.decode_level_string(data), dict(kind="file" if body.get("file") else "paste", name=name))


def read_live():
    """The level the Geode mod last reported, validated, or None when there is no file / it is unreadable."""
    try:
        with open(LIVE_FILE, encoding="utf-8", errors="replace") as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict) or d.get("kind") not in LIVE_KINDS:
        return None
    try:
        out = dict(seq=int(d.get("seq", 0)), kind=d["kind"], id=int(d.get("id", 0)), name=str(d.get("name") or "")[:80],
                   creator=str(d.get("creator") or "")[:40], stars=max(int(d.get("stars", 0)), 0), demon=bool(d.get("demon")),
                   game_dir=str(d.get("gameDir") or ""), has_data=bool(d.get("data")), ts=float(d.get("ts", 0)) / 1000.0)
    except (TypeError, ValueError):
        return None
    if not 0 <= out["id"] < 10 ** 10:
        return None
    return out


def live_state():
    """What /api/live answers: cheap and lock-free, so the page can poll it while a rating runs."""
    d = read_live()
    if not d:
        return dict(present=False, file=LIVE_FILE)
    d.update(present=True, file=LIVE_FILE, age=round(max(time.time() - d["ts"], 0), 1) if d["ts"] else None)
    d.pop("game_dir", None)
    return d


def official_text(level_id, game_dir):
    """Official levels are not on the servers; the game keeps them in Resources/levels/<id>.txt."""
    if not game_dir or not os.path.isfile(os.path.join(game_dir, "GeometryDash.exe")):
        raise gdnet.GDError("Couldn't find your Geometry Dash folder. Update the Starmeter Live mod or open the level from the Paste tab.")
    path = os.path.join(game_dir, "Resources", "levels", f"{int(level_id)}.txt")
    if not os.path.isfile(path):
        raise gdnet.GDError(f"This official level isn't in your game folder (looked for levels/{int(level_id)}.txt).")
    return gdnet.decode_level_string(open(path, encoding="utf-8", errors="replace").read().strip())


def live_rate():
    """Rate the level the mod reported most recently (the page and the mod both ask, so remember the last answer)."""
    d = read_live()
    if not d:
        raise gdnet.GDError("The Starmeter Live mod hasn't reported a level yet. Open a level in Geometry Dash.")
    key = (d["seq"], d["ts"], d["id"])
    if live_rate.last[0] == key:
        return live_rate.last[1]
    if d["kind"] == "online":
        out = rate_online(d["id"])
    elif d["kind"] == "official":
        text = official_text(d["id"], d["game_dir"])
        comm = dict(official_stars=d["stars"], face="")
        if d["stars"]:
            comm["star"] = float(d["stars"])
        out = run(text, dict(kind="official", name=d["name"] or f"Official level {d['id']}"), extra=dict(community=comm))
    else:
        path = os.path.join(os.path.dirname(LIVE_FILE), "level.txt")
        if not d["has_data"] or not os.path.isfile(path) or os.path.getsize(path) > MAX_BODY:
            raise gdnet.GDError("This level has no data yet. Build something in the editor, then open it again.")
        text = gdnet.decode_level_string(open(path, encoding="utf-8", errors="replace").read().strip())
        out = run(text, dict(kind="local", name=d["name"] or "Your level"))
    if out["sim"].get("status") == "pending":                  # the in-game badge waits for the physics-checked number
        out = simulate(out["sim"]["token"])                    # (the mod allows 120 s; a level takes about 5-20 s)
    out["live"] = dict(seq=d["seq"], kind=d["kind"], id=d["id"], name=d["name"], creator=d["creator"])
    live_rate.last = (key, out)
    return out


live_rate.last = (None, None)


def about():
    m = model.load()
    out = dict(metrics=m["metrics"], trained=m["trained"], demon_cuts=m["demon_cuts"], importance=m["importance"],
               n_features=len(m["features"]), err_bins=m["err_bins"])
    if model.has_sim():
        sm = model.load("sim")
        out["sim"] = dict(metrics=sm["metrics"], importance=sm["importance"], n_features=len(sm["features"]))
    out["physics"] = sim_ready()
    return out


class H(BaseHTTPRequestHandler):
    server_version = "Starmeter"

    def log_message(self, fmt, *a):
        pass

    def send(self, code, obj, ctype="application/json"):
        body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith("text") or "json" in ctype else ""))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def api(self, fn):
        try:
            with LOCK:
                self.send(200, fn())
        except gdnet.GDError as e:
            self.send(502 if "reach" in str(e) or "HTTP" in str(e) else 404, dict(error=str(e)))
        except (ValueError, KeyError) as e:
            self.send(422, dict(error=f"That level could not be read ({e}). Is it a complete level string?"))
        except FileNotFoundError as e:
            self.send(503, dict(error=str(e)))
        except Exception:
            traceback.print_exc()
            self.send(500, dict(error="Unexpected error while rating this level. Details are in the terminal."))

    def local_only(self):
        """Refuse requests addressed to any host name but localhost (blocks DNS-rebinding from other websites)."""
        if (self.headers.get("Host") or "").rsplit(":", 1)[0] in ("127.0.0.1", "localhost", "[::1]"):
            return True
        self.send(403, dict(error="Forbidden"))
        return False

    def do_GET(self):
        if not self.local_only():
            return
        u = urllib.parse.urlparse(self.path)
        q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
        if u.path == "/favicon.ico":                      # the page ships an inline icon; stop the browser's 404 probe
            return self.send(204, b"", "image/x-icon")
        if u.path == "/api/rate":
            lid = q.get("id", "").strip().lstrip("#")
            if not lid.isdigit() or not (0 < int(lid) < 10 ** 10):
                return self.send(400, dict(error="Enter a numeric level ID, for example 10565740."))
            return self.api(lambda: rate_online(int(lid)))
        if u.path == "/api/local":
            return self.api(lambda: [dict(i=i, name=l["name"], objects=l["objects"], kb=l["kb"]) for i, l in enumerate(gdnet.read_local_levels())])
        if u.path == "/api/local/rate":
            def go():
                lv = gdnet.read_local_levels()[int(q.get("i", -1))]
                return run(gdnet.decode_level_string(lv["data"]), dict(kind="local", name=lv["name"]))
            return self.api(go)
        if u.path == "/api/live":
            return self.send(200, live_state())
        if u.path == "/api/live/rate":
            return self.api(live_rate)
        if u.path == "/api/sim":
            tok = q.get("token", "")
            if not tok.isalnum() or len(tok) > 32:
                return self.send(400, dict(error="Bad request."))
            return self.api(lambda: simulate(tok))
        if u.path == "/api/replay":
            tok = q.get("token", "")
            if tok not in REPLAYS:
                return self.send(404, dict(error="That replay expired. Rate the level again."))
            return self.api(lambda: REPLAYS[tok])
        if u.path == "/api/about":
            return self.api(about)
        name = "index.html" if u.path in ("/", "") else u.path.lstrip("/")
        path = os.path.normpath(os.path.join(WEB, name))
        if not path.startswith(WEB) or not os.path.isfile(path):
            return self.send(404, dict(error="Not found"))
        ctype = {".html": "text/html", ".css": "text/css", ".js": "text/javascript", ".svg": "image/svg+xml"}.get(os.path.splitext(path)[1], "application/octet-stream")
        self.send(200, open(path, "rb").read(), ctype)

    def do_POST(self):
        if not self.local_only():
            return
        n = int(self.headers.get("Content-Length") or 0)
        if self.path != "/api/paste" or n > MAX_BODY:
            return self.send(413 if n > MAX_BODY else 404, dict(error="Request too large." if n > MAX_BODY else "Not found"))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return self.send(400, dict(error="Malformed request."))
        self.api(lambda: rate_pasted(body))


def main():
    global LIVE_FILE
    if "--live-file" in sys.argv:
        LIVE_FILE = sys.argv[sys.argv.index("--live-file") + 1]
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8765
    for p in range(port, port + 20):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", p), H)
            break
        except OSError:
            continue
    else:
        sys.exit("No free port found.")
    if sim_ready():                                           # compile the simulator in the background so the first physics pass is not slow
        def warm():
            try:
                import simfeat
                simfeat.extract("kA2,0,kA4,0;1,8,2,600,3,15;1,1,2,1500,3,15")
            except Exception:
                traceback.print_exc()
        threading.Thread(target=warm, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    print(f"Starmeter running at {url}  (Ctrl+C to stop)")
    if "--no-browser" not in sys.argv:
        threading.Timer(0.6, webbrowser.open, (url,)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("bye")


if __name__ == "__main__":
    main()
