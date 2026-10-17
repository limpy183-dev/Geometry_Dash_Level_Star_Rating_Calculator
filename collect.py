"""Build the training corpus (resumable, throttled, safe to Ctrl-C and re-run).

    python collect.py plan demons          # pick demons from the GDDL listing        -> data/plan_demons.json
    python collect.py plan plain           # pick rated non-demons via GD search       -> data/plan_plain.json
    python collect.py levels demons        # download a plan                           -> data/levels.db
    python collect.py gddl                 # attach community vote counts to demons (run alongside `levels`)

Labels: demons use GDDL's decimal tier; everything else uses the official star count.
The GD servers rate-limit hard (Cloudflare 1015), so every network step is slow, cached and resumable.
"""
import base64, collections, json, os, random, sqlite3, sys, time
import gdnet

HERE = os.path.dirname(os.path.abspath(__file__))
D = lambda n: os.path.join(HERE, "data", n)
DB, POOL, SEARCH = D("levels.db"), D("demon_pool.json"), D("search_cache.json")
FACES = {"auto": ("-3", 90), "easy": ("1", 150), "normal": ("2", 150), "hard": ("3", 190),
         "harder": ("4", 190), "insane": ("5", 190)}          # face -> (server diff code, target count)
PAGES, SEARCH_GAP = 15, 2.5                                    # pages of 10 per (face, list type); seconds between searches
MAX_OBJECTS = 350_000                                          # keeps parsing time sane
PER_TIER_BIN = 140                                             # cap demons per 1.5-tier bin so easy ones don't dominate


def connect():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    db = sqlite3.connect(DB, timeout=120)
    db.execute("pragma journal_mode=wal")
    db.execute("create table if not exists levels(id integer primary key, kind text, meta text, gddl text, ldata blob)")
    db.execute("create table if not exists skips(id integer primary key, why text)")
    return db


def put(db, sql, args=()):
    """One short write transaction, retried while another process (training, a 2nd worker) holds the lock."""
    for _ in range(60):
        try:
            db.execute(sql, args)
            db.commit()
            return
        except sqlite3.OperationalError as e:
            if "locked" not in str(e):
                raise
            db.rollback()
            time.sleep(3)
    raise sqlite3.OperationalError("database stayed locked")


def patient(fn, *a, **kw):
    """Call fn, sleeping through rate limits instead of failing."""
    for n in range(40):
        try:
            return fn(*a, **kw)
        except gdnet.GDError as e:
            if "429" not in str(e) and "reach" not in str(e):
                raise
            wait = 180 if "429" in str(e) else 30
            print(f"\n  rate-limited / offline ({e}); sleeping {wait}s", flush=True)
            time.sleep(wait)
    raise gdnet.GDError("giving up after repeated rate limits")


def demon_pool():
    if os.path.exists(POOL):
        return {d["id"]: d for d in json.load(open(POOL))}
    demons = {}
    for q in ({"sort": "popularity"}, {"minRating": 20, "sort": "popularity"}, {"minRating": 28, "sort": "popularity"}):
        for page in range(90):
            rows, _ = patient(gdnet.gddl_list, page, **q)
            for r in rows:
                if r["rating"] is not None and r["difficulty"] != "Official":
                    demons[r["id"]] = dict(id=r["id"], kind="demon", tier=r["rating"])
            print(f"\rGDDL {q} page {page}: {len(demons)} demons", end="", flush=True)
            if len(rows) < 25:
                break
    print()
    json.dump(list(demons.values()), open(POOL, "w"))
    return demons


def plan_demons():
    rng, bins = random.Random(7), collections.defaultdict(list)
    for d in demon_pool().values():
        bins[int(d["tier"] / 1.5)].append(d)
    chosen = [d for b in bins.values() for d in rng.sample(b, min(PER_TIER_BIN, len(b)))]
    rng.shuffle(chosen)
    json.dump(chosen, open(D("plan_demons.json"), "w"))
    print("planned", len(chosen), "demons")


def plan_demons2():
    """Round two: mid-tier demons (GDDL 8-20) that the popularity-sorted first round under-sampled."""
    rng, have, bins = random.Random(8), {d["id"] for d in json.load(open(D("plan_demons.json")))}, collections.defaultdict(list)
    for page in range(120):
        rows, _ = patch_get(page)
        for r in rows:
            if r["rating"] is not None and r["difficulty"] != "Official" and 8 <= r["rating"] < 20 and r["id"] not in have:
                bins[int(r["rating"] / 1.5)].append(dict(id=r["id"], kind="demon", tier=r["rating"]))
        print(f"\rGDDL minRating=8 page {page}: {sum(map(len, bins.values()))} new mid-tier demons", end="", flush=True)
        if len(rows) < 25:
            break
    chosen = [d for b in bins.values() for d in rng.sample(b, min(70, len(b)))]
    rng.shuffle(chosen)
    json.dump(chosen, open(D("plan_demons2.json"), "w"))
    print("\nplanned", len(chosen), "more demons")


def patch_get(page):
    return patient(gdnet.gddl_list, page, minRating=8, sort="popularity")


def plan_plain(more=False):
    cache = json.load(open(SEARCH)) if os.path.exists(SEARCH) else {}
    for face, (code, _) in FACES.items():
        for typ in (2, 4):                                     # most liked + recent => variety beyond polished classics
            for page in range(PAGES + (20 if more else 0)):
                key = f"{face}|{typ}|{page}"
                if key in cache:
                    if not cache[key]:
                        break
                    continue
                cache[key] = patient(gdnet.search_levels, type=typ, str="", diff=code, len="-", page=page, star=1, gap=SEARCH_GAP)
                json.dump(cache, open(SEARCH, "w"))
                print(f"\r{face:7s} type {typ} page {page:2d}: {len(cache[key])} levels", end="", flush=True)
                if not cache[key]:
                    break
    rng, chosen = random.Random(12 if more else 11), []
    taken = {p["id"] for p in json.load(open(D("plan_plain.json")))} if more else set()
    for face, (_, want) in FACES.items():
        want = {"auto": 0, "easy": 100, "normal": 100}.get(face, 120) if more else want
        got = {}
        for key, rows in cache.items():
            if key.startswith(face + "|"):
                for r in rows:
                    if not r["demon"] and not r["two_player"] and r["length"] < 5 and r["objects"] <= MAX_OBJECTS and r["stars"] > 0 and r["id"] not in taken:
                        got[r["id"]] = dict(id=r["id"], kind=face, stars=r["stars"])
        chosen += rng.sample(list(got.values()), min(want, len(got)))
    rng.shuffle(chosen)
    json.dump(chosen, open(D("plan_plain2.json" if more else "plan_plain.json"), "w"))
    print("\nplanned", len(chosen), collections.Counter(c["kind"] for c in chosen))


def plan_double(factor=1.06):
    """Double the corpus: for every demon tier bin and every plain face, add as many levels as already stored (x1.06 for skips).

    Demons come from the *whole* GDDL listing (cached in data/gddl_all.json), plain levels from the search cache. -> data/plan_more.json
    """
    allp = D("gddl_all.json")
    if not os.path.exists(allp):
        rows, page = [], 0
        while True:
            got, _ = patient(gdnet.gddl_list, page)
            rows += got
            print(f"\rGDDL listing page {page}: {len(rows)}", end="", flush=True)
            if len(got) < 25:
                break
            page += 1
        json.dump(rows, open(allp, "w"))
    db = connect()
    have, haveplain, seen = collections.Counter(), collections.Counter(), set()
    for i, kind, meta in db.execute("select id, kind, meta from levels"):
        seen.add(i)
        if kind == "demon":
            have[int(json.loads(meta)["tier"] / 1.5)] += 1
        else:
            haveplain[kind] += 1
    seen |= {r[0] for r in db.execute("select id from skips")}
    rng, bins = random.Random(9), collections.defaultdict(list)
    for r in json.load(open(allp)):
        if r["rating"] is not None and r["difficulty"] != "Official" and r["id"] not in seen:
            bins[int(r["rating"] / 1.5)].append(dict(id=r["id"], kind="demon", tier=r["rating"]))
    chosen = []
    for b, v in bins.items():
        rng.shuffle(v)
        chosen += v[:round(have[b] * factor)]
        bins[b] = v[round(have[b] * factor):]                  # leftovers
    # the hardest bins are exhausted (every known level is already stored): make up the shortfall evenly from bins that still have some
    short = sum(have.values()) - len(chosen)
    while short > 0 and any(bins.values()):
        for v in bins.values():
            if v and short > 0:
                chosen.append(v.pop())
                short -= 1
    cache = json.load(open(SEARCH))
    for face in FACES:
        if face == "auto":                                     # auto levels are excluded from training anyway
            continue
        got = {}
        for key, rows in cache.items():
            if key.startswith(face + "|"):
                for r in rows:
                    if not r["demon"] and not r["two_player"] and r["length"] < 5 and r["objects"] <= MAX_OBJECTS and r["stars"] > 0 and r["id"] not in seen:
                        got[r["id"]] = dict(id=r["id"], kind=face, stars=r["stars"])
        chosen += rng.sample(list(got.values()), min(len(got), round(haveplain[face] * factor)))
    rng.shuffle(chosen)
    json.dump(chosen, open(D("plan_more.json"), "w"))
    print("\nplanned", len(chosen), collections.Counter(c["kind"] for c in chosen))
    print("demon bins (have -> add / unused):", {b * 1.5: (have[b], sum(1 for c in chosen if c["kind"] == "demon" and int(c["tier"] / 1.5) == b), len(bins[b])) for b in sorted(set(have) | set(bins))})


def levels():
    name = sys.argv[2] if len(sys.argv) > 2 else "demons"
    db, todo = connect(), json.load(open(D(f"plan_{name}.json")))
    done = {r[0] for r in db.execute("select id from levels union select id from skips")}
    todo = [p for p in todo if p["id"] not in done]
    print(len(todo), "levels to fetch")
    fails, t0 = 0, time.time()
    for i, p in enumerate(todo, 1):
        try:
            meta, s = gdnet.download_level(p["id"], gap=1.0)
            fails = 0
        except gdnet.GDError as e:
            msg = str(e)
            if "not found" in msg:
                put(db, "insert or replace into skips values(?,?)", (p["id"], "missing"))
            else:
                fails += 1
                wait = 180 if "429" in msg else min(300, 15 * fails)
                print(f"\n! {p['id']} {msg}; sleeping {wait}s", flush=True)
                time.sleep(wait)                                # back off hard when the server pushes back
            continue
        why = ("two-player" if meta["two_player"] else "platformer" if meta["length"] == 5 else
               "too big" if meta["objects"] > MAX_OBJECTS else None)
        if why:
            put(db, "insert or replace into skips values(?,?)", (p["id"], why))
        else:
            try:
                blob = base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))   # store the gzip bytes (25% smaller than base64)
            except Exception:
                blob = s.encode()
            put(db, "insert or replace into levels(id,kind,meta,ldata) values(?,?,?,?)",
                (p["id"], p["kind"], json.dumps({**meta, **{k: v for k, v in p.items() if k in ("tier", "stars")}}), blob))
        if i % 20 == 0:
            rate = i / (time.time() - t0)
            print(f"\r{i}/{len(todo)}  {rate:.2f} lv/s  eta {(len(todo) - i) / rate / 60:.0f} min", end="", flush=True)
    print("\ndone")


def gddl():
    db = connect()
    while True:
        rows = db.execute("select id from levels where kind='demon' and gddl is null limit 100").fetchall()
        if not rows:
            print("no demons waiting; levels in db:", db.execute("select count(*) from levels").fetchone()[0])
            if "--follow" not in sys.argv:
                return
            time.sleep(60)
            continue
        for (i,) in rows:
            g = patient(gdnet.gddl_level, i, gap=0.8)
            put(db, "update levels set gddl=? where id=?", (json.dumps(g), i))          # None -> 'null' (rating hidden)
        print("gddl batch ok", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "plan"
    if cmd == "plan":
        {"double": plan_double, "demons": plan_demons, "plain": plan_plain, "demons2": plan_demons2, "plain2": lambda: plan_plain(True)}[sys.argv[2] if len(sys.argv) > 2 else "demons"]()
    else:
        {"levels": levels, "gddl": gddl}[cmd]()
