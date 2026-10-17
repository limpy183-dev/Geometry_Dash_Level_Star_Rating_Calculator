"""Derive object-ID classes (hazard / saw / fake / invisible / solid / trigger) from community object names.

    python tools/build_objclasses.py        # downloads the name list, writes objclasses.json

Names: FlowVix/gd-info-explorer objects.csv (names credited there to Alphalaneous and the community).
Portal/orb/pad/trigger IDs are hard-coded in gdlevel.py (from the MIT-licensed gmdkit constants).
Only the derived ID lists are shipped, not the name list itself.
"""
import csv, io, json, os, re, urllib.request

URL = "https://raw.githubusercontent.com/FlowVix/gd-info-explorer/main/src/lib/objects.csv"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "objclasses.json")

NOT_HAZARD = re.compile(r"Decorative|Rod\b|Wheel|Pixel|Base\b|Filler|Outline|Particle|Collectable|Claw|Lava", re.I)
SAW = re.compile(r"Saw ?blade|Blade|Gear|Scythe|Wheel|Round Black Hazard|Monster|Beast|Bat", re.I)
SOLID = re.compile(r"\b(Squares?|Slabs?|Slopes?|Blocks?|Platforms?|Tiles?)\b", re.I)
NOT_SOLID = re.compile(r"Decorat|Outline|Pulsing|Particle|Pixel|Collectable|Non-Solid|Glow|Fake|Hazard|Spike|Blade|Saw|"
                       r"Portal|\bPad\b|\bOrb\b|Trigger|Damage|Collision|Toggle|State|Item|Checkpoint|Button|Force|Pipe|Animat|Filler|Inner|\bBase\b|^Colored ", re.I)   # "Inner" / "Base" fill pieces and "Colored" tint blocks have no hitbox (validated against beatable levels)


def classify(rows):
    cls = {k: [] for k in ("hazard", "saw", "fake", "invisible", "solid", "trigger")}
    for i, n in rows:
        if re.search(r"Fake", n) and re.search(r"Spike|Hazard", n):
            cls["fake"].append(i)
        elif (re.search(r"Hazard|Damage Block", n) or (re.search(r"\bSpike\b|Blade|Sawblade", n, re.I) and not NOT_HAZARD.search(n))) \
                and not re.search(r"Outline Blade|Spike Rod|Spike Wheel|Decorative", n):
            cls["hazard"].append(i)
            if SAW.search(n):
                cls["saw"].append(i)
            if re.search(r"Invisible", n):
                cls["invisible"].append(i)
        elif re.search(r"Trigger", n):
            cls["trigger"].append(i)
        elif SOLID.search(n) and not NOT_SOLID.search(n):
            cls["solid"].append(i)
    return cls


if __name__ == "__main__":
    txt = urllib.request.urlopen(urllib.request.Request(URL, headers={"User-Agent": "GD-StarRater"}), timeout=30).read().decode()
    rows = [(int(r[0]), ",".join(r[1:])) for r in csv.reader(io.StringIO(txt)) if len(r) > 1 and r[0].isdigit()]
    cls = classify(rows)
    json.dump(cls, open(OUT, "w"), separators=(",", ":"))
    print({k: len(v) for k, v in cls.items()}, "->", os.path.normpath(OUT))
