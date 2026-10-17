"""Derive collision shapes per object ID -> objshapes.json   (used by gdsim.py)

    python tools/build_shapes.py            # needs data/levels.db (the corpus) for the placement statistics

Hazards: shape from the community object names (spike, saw, pit, monster...), sizes from game knowledge.
Solids:  30x30 unless the name says smaller, then confirmed by where real levels place the piece
         (half-height and quarter pieces sit on half-grid offsets; slabs sit about 7 units off the grid).
Slopes:  right triangles; orientation comes from each object's rotation and flip at load time.
"""
import collections, csv, io, json, os, random, re, sqlite3, sys, urllib.request, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
import gdlevel

URL = "https://raw.githubusercontent.com/FlowVix/gd-info-explorer/main/src/lib/objects.csv"
SAMPLE, MIN_N = 1200, 10                           # levels scanned for placement statistics; min placements to trust them


def names():
    txt = urllib.request.urlopen(urllib.request.Request(URL, headers={"User-Agent": "GD-StarRater"}), timeout=30).read().decode()
    return {int(r[0]): ",".join(r[1:]) for r in csv.reader(io.StringIO(txt)) if len(r) > 1 and r[0].isdigit()}


def size_word(n):
    return "large" if re.search(r"Large|Big|Very Large", n) else "small" if re.search(r"Small|Tiny|Mini", n) else "medium"


def hazard_shape(n):
    """-> [kind, a, b, ox, oy]; kind 0 rect (w=a, h=b) / 1 circle (r=a). Offsets from the object centre, unrotated (pointing up)."""
    if re.search(r"Scythe", n):
        return [1, 36 if size_word(n) == "large" else 22, 0, 0, 0]
    if re.search(r"Saw ?blade|Blade|Gear", n, re.I):
        r = {"large": 40, "medium": 27, "small": 14}[size_word(n)]
        return [1, r + (3 if "Spike Blade" in n else 0), 0, 0, 0]
    if re.search(r"Spiked (Round )?Monster", n):
        return [1, 30, 0, 0, 0]
    if re.search(r"Monster|Beast|Bat", n):
        return [1, 40 if size_word(n) == "large" else 22, 0, 0, 0]
    if re.search(r"Spiked (Square|Circle)|Triangle Hazard", n):
        return [1, 18, 0, 0, 0]
    if re.search(r"Pit Hazard|Thorn|Hazard", n):
        return [0, 26, 14, 0, 2]
    if re.search(r"Half|Small", n):
        return [0, 6, 6, 0, -4.5]
    if re.search(r"Tiny", n):
        return [0, 5, 5, 0, -5]
    return [0, 6, 12, 0, -1.5]                                    # standard spike: small core, raised off the base


def main():
    nm = names()
    cls = json.load(open(os.path.join(ROOT, "objclasses.json")))
    haz = {i: hazard_shape(nm.get(i, "")) for i in cls["hazard"]}
    # --- solids: name default, then placement statistics from the corpus (native frame: undo the 90-degree rotations)
    db = sqlite3.connect(os.path.join(ROOT, "data", "levels.db"), timeout=60)
    ids = [r[0] for r in db.execute("select id from levels")]
    random.Random(21).shuffle(ids)
    solid, flat = set(cls["solid"]), {i for i, sh in haz.items() if sh == [0, 26, 14, 0, 2]}
    al = collections.defaultdict(lambda: collections.Counter())
    gy = collections.defaultdict(collections.Counter)                       # ground-level y of flat hazards -> sprite height
    for lid in ids[:SAMPLE]:
        blob = db.execute("select ldata from levels where id=?", (lid,)).fetchone()[0]
        _, objs, aux = gdlevel.parse(zlib.decompress(blob, 47).decode("utf-8", "replace"), frozenset(("6", "4", "5", "128", "129")))
        for i, (oid, x, y, sc) in enumerate(objs):
            a = aux.get(i, {})
            if oid in flat and sc == 1.0 and not ({"6", "4", "5", "128", "129"} & a.keys()) and 0 <= y <= 40:
                gy[oid][round(y * 2) / 2] += 1
                continue
            if oid not in solid or sc != 1.0 or "128" in a or "129" in a:
                continue
            rot = float(a.get("6", 0) or 0)
            q = round(rot / 90.0)
            if abs(rot - 90.0 * q) > 0.5:
                continue
            nx, ny = (x % 30, y % 30) if q % 2 == 0 else (y % 30, x % 30)          # offsets in the object's own frame
            for ax, off in (("x", nx), ("y", ny)):
                al[oid][ax + "15"] += abs(off - 15) < 1.2
                al[oid][ax + "7"] += min(abs(off - 7.5), abs(off - 22.5)) < 1.2
                al[oid][ax + "3"] += min(abs(off - 3.75), abs(off - 11.25), abs(off - 18.75), abs(off - 26.25)) < 1.2
                al[oid][ax + "s"] += min(abs(off - 7), abs(off - 23)) < 0.8
            al[oid]["n"] += 1
    out_solid = {}
    for oid in solid:
        n = nm.get(oid, "")
        w = h = 30.0
        if "One-Sixteenth" in n:
            w = h = 7.5
        elif "Quarter" in n or re.search(r"Small (Square|Block)|Small Metal Block", n):
            w = h = 15.0
        elif re.search(r"Half", n):
            h = 15.0
        a = al.get(oid)
        slope = bool(re.search(r"Slope", n)) and not re.search(r"Connector|Corner|Hazard", n)
        if a and a["n"] >= MIN_N and not slope:               # slopes keep their name-based size: their bounding boxes sit off-grid                                 # statistics beat names when the piece has been seen enough
            def size(ax, dflt):
                f = {k: a[ax + k] / a["n"] for k in ("15", "7", "3", "s")}
                if f["15"] > .5: return 30.0
                if f["3"] > .5: return 7.5
                if f["s"] > .35 and f["7"] > .25 and re.search(r"Slab", n): return 14.0
                if f["7"] > .5: return 15.0
                return dflt
            w, h = size("x", w), size("y", h)
        if slope:
            out_solid[oid] = [1, 60.0 if "Wide" in n else w, h]
        elif (w, h) != (30.0, 30.0):
            out_solid[oid] = [0, w, h]
    # flat hazards sit on the ground: the object centre is half the sprite height above it. Hitbox = a fraction of that sprite box.
    flat_h = {}
    for oid, c in gy.items():
        y, k = c.most_common(1)[0]
        if sum(c.values()) >= 25 and k >= 0.3 * sum(c.values()):
            flat_h[oid] = min(max(2 * y, 4.0), 30.0)
    json.dump(dict(hazard=haz, solid=out_solid, flat=flat_h), open(os.path.join(ROOT, "objshapes.json"), "w"), separators=(",", ":"))
    kinds = collections.Counter(("circle" if v[0] else "rect") for v in haz.values())
    print("hazards", dict(kinds), "| non-default solids:", len(out_solid), "(slopes:", sum(1 for v in out_solid.values() if v[0] == 1), ")")


if __name__ == "__main__":
    main()
