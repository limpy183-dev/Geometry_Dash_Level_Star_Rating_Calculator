"""Collision shapes per object ID from the game itself -> objshapes.json   (used by gdsim.py)

    python tools/build_hitboxes.py          # reads data/objdump.jsonl

data/objdump.jsonl is written by the one-shot Geode mod in tools/hitbox-dump/: it creates every object ID inside
Geometry Dash, runs the game's own setup (GameObject::customSetup) and records the object type and hitbox
(getObjectRect, m_objectRadius). Sizes of hazards, solids, pads and orbs come from there.

Which pieces collide is decided by the game as well: data/objprobe.json (tools/oracle.py probe) put every object the game
types solid in front of the player in the real game. The 3DL edge pieces turned out to let the player through although
their type says solid; everything else stops it. Without the probe file the game's type is used. Slopes the community names
call hazards keep a hazard box.
"""
import json, os, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import gdlevel

SOLID, HAZARD, SLOPE, ANIM_HAZARD = 0, 2, 25, 47                   # GameObjectType values
PADS = {8, 9, 10, 34, 44}                                          # yellow, pink, gravity, red, spider pad
ORBS = {11, 12, 13, 29, 32, 35, 36, 37, 38, 43, 46}                # every ring type
PROBE = os.path.join(ROOT, "data", "objprobe.json")              # id -> kill / lift / pass, measured in the game
SLOPE_HAZARD = [0, 26, 14, 0, 2]                                   # box for slope-shaped hazards (the game reports the slope rect)


def main():
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", "objdump.jsonl"))]
    probe = {int(k): v for k, v in json.load(open(PROBE)).items()} if os.path.exists(PROBE) else {}
    haz, sol, pad, orb, por = {}, {}, {}, {}, {}
    for d in rows:
        i, t = d["id"], d["type"]
        named = gdlevel.CAT.get(i)
        if named == gdlevel.X_:                                     # portals: the box the player has to touch
            por[i] = [d["rw"], d["rh"]]
        if t in (HAZARD, ANIM_HAZARD):
            haz[i] = [1, d["rad"], 0, 0, 0] if d["rad"] > 0 else [0, d["rw"], d["rh"], d["box"][0], d["box"][1]]
        elif t == SLOPE and named == gdlevel.H_:
            haz[i] = SLOPE_HAZARD
        elif t in (SOLID, SLOPE) and probe.get(i, "kill") != "pass":
            sol[i] = [1 if t == SLOPE else 0, d["rw"], d["rh"]]
        elif t in PADS:
            pad[i] = [d["rw"], d["rh"]]
        elif t in ORBS:
            orb[i] = [d["rw"], d["rh"]]
    json.dump(dict(hazard=haz, solid=sol, pad=pad, orb=orb, portal=por, flat={}, source="game"), open(os.path.join(ROOT, "objshapes.json"), "w"), separators=(",", ":"))
    print(f"{len(sol)} solids ({sum(v[0] for v in sol.values())} slopes), {len(haz)} hazards ({sum(v[0] for v in haz.values())} round), {len(pad)} pads, {len(orb)} orbs, {len(por)} portals")


if __name__ == "__main__":
    main()
