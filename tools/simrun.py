"""Run the physics simulation over the whole corpus -> table simf in data/levels.db  (resumable; ~2-3 h for 4,000 levels).

    python tools/simrun.py [workers]
"""
import io, json, os, sqlite3, sys, time, zlib

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
DB = os.path.join(ROOT, "data", "levels.db")


def work(lid):
    import simfeat
    db = sqlite3.connect("file:" + DB.replace("\\", "/") + "?mode=ro", uri=True, timeout=60)
    blob = db.execute("select ldata from levels where id=?", (lid,)).fetchone()[0]
    db.close()
    t = time.time()
    route = None
    try:
        f, detail = simfeat.analyze(zlib.decompress(blob, 47).decode("utf-8", "replace"))
        err = ""
        if detail:                                                    # keep the route itself: later features need no re-planning
            buf = io.BytesIO()
            np.savez_compressed(buf, **detail["route"], risk=np.array(detail["risk"], np.float16), mode=np.array(detail["mode"], np.int8))
            route = buf.getvalue()
    except Exception as e:                                           # a level that breaks the simulator must not stop the run
        f, err = dict(sim_ok=0.0), repr(e)[:200]
    return lid, f, err, time.time() - t, route


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else max(2, (os.cpu_count() or 4) - 2)
    db = sqlite3.connect(DB, timeout=60)
    db.execute("create table if not exists simf(id integer primary key, f text, err text, sec real)")
    db.execute("create table if not exists simr(id integer primary key, route blob)")
    db.commit()
    done = {r[0] for r in db.execute("select id from simf")}
    todo = [r[0] for r in db.execute("select id from levels") if r[0] not in done]
    print(len(todo), "levels to simulate with", workers, "workers", flush=True)
    work(todo[0]) if todo else None                                   # compile once in the parent so the workers load the cache
    t0, n = time.time(), 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for lid, f, err, sec, route in ex.map(work, todo, chunksize=1):
            db.execute("insert or replace into simf values(?,?,?,?)", (lid, json.dumps(f), err, sec))
            if route:
                db.execute("insert or replace into simr values(?,?)", (lid, route))
            n += 1
            if n % 25 == 0:
                db.commit()
                print(f"{n}/{len(todo)}  {time.time() - t0:.0f}s  last {sec:.1f}s", flush=True)
    db.commit()
    print("done", n, f"{time.time() - t0:.0f}s")
