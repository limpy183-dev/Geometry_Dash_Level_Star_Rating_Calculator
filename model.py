"""Runtime model: feature dict -> star rating. Pure Python; the trees live in model.json (written by train.py).

Scale: official stars for normal levels (1-9.99); demons start at 10 and climb 0.25 per GDDL tier,
so Easy Demon ~ 10.0 and the hardest Extreme Demons ~ 19.5.

Prediction = P(demon) * demon specialist + (1 - P(demon)) * normal specialist, then a monotone calibration.

Two trained variants: "sim" also reads the physics-simulation features (model_sim.json) and is used whenever those are
available; "base" reads the structural features only (model.json) and is the fallback.
"""
import bisect, json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
PATHS = {"base": os.path.join(HERE, "model.json"), "sim": os.path.join(HERE, "model_sim.json")}
PATH = PATHS["base"]
_M = {}

# (feature, label) pairs for the "skill profile"; each is shown as a percentile vs. the training levels
PROFILE = [("inputs_p90", "Input pressure"), ("tight_p90", "Precision"), ("speed_mean", "Speed"),
           ("len_s", "Endurance"), ("obj_per_s", "Complexity"), ("portals_per_min", "Variety"), ("trig_per_s", "Gimmicks")]


def tier_to_star(t):
    return 10 + (t - 1) * 0.25


def star_to_tier(s):
    return (s - 10) / 0.25 + 1


def load(kind="base"):
    if kind not in _M:
        if not os.path.exists(PATHS[kind]):
            raise FileNotFoundError(os.path.basename(PATHS[kind]) + " is missing - run `python train.py` first.")
        _M[kind] = json.load(open(PATHS[kind]))
    return _M[kind]


def has_sim():
    return os.path.exists(PATHS["sim"])


def pick(features):
    """Which variant reads these features: the simulation-aware one when the level was simulated."""
    return "sim" if features.get("sim_ok") and has_sim() else "base"


def _score(mod, x):
    s = mod["init"]
    for L, R, F, T, V in mod["trees"]:
        i = 0
        while L[i] >= 0:
            i = L[i] if x[F[i]] <= T[i] else R[i]
        s += mod["lr"] * V[i]
    return s


def _interp(v, xs, ys):
    i = min(max(bisect.bisect_right(xs, v), 1), len(xs) - 1)
    return ys[i - 1] + (ys[i] - ys[i - 1]) * (v - xs[i - 1]) / (xs[i] - xs[i - 1])


def predict(features, kind=None):
    """-> (star rating, probability the level is a demon). Ensemble of boosted trees, then calibrated."""
    m = load(kind or pick(features))
    x = [float(features.get(n, 0.0)) for n in m["features"]]
    stars = ps = 0.0
    for mod in m["models"]:
        p = 1 / (1 + math.exp(-_score(mod["clf"], x)))
        stars += (1 - p) * _score(mod["rn"], x) + p * _score(mod["rd"], x)
        ps += p
    n = len(m["models"])
    return min(max(_interp(stars / n, m["calib"]["x"], m["calib"]["y"]), 0.5), 21.0), ps / n


def face(star):
    """Difficulty name for a star rating."""
    m = load()                                   # the band edges come from the base model's corpus
    if star < 9.5:
        for cut, name in ((1.5, "Auto"), (2.5, "Easy"), (3.5, "Normal"), (5.5, "Hard"), (7.5, "Harder")):
            if star < cut:
                return name
        return "Insane"
    for cut, name in m["demon_cuts"]:
        if star < cut:
            return name
    return "Extreme Demon"


def uncertainty(star, kind="base"):
    """Typical absolute error (50% / 80% of cases) around this rating, from out-of-fold predictions."""
    bins = load(kind)["err_bins"]
    for edge, p50, p80 in bins:
        if star < edge:
            return p50, p80
    return bins[-1][1], bins[-1][2]


def percentile(name, value, kind="base"):
    q = load(kind)["quant"].get(name)
    return round(100 * bisect.bisect_left(q, value) / len(q), 1) if q else None


def similar(features, k=5, exclude=None, kind="base"):
    """Nearest training levels in the model's most important features (a sanity check for the number)."""
    nn = load(kind)["nn"]
    z = [(features.get(c, 0.0) - mu) / sd for c, mu, sd in zip(nn["cols"], nn["mean"], nn["std"])]
    best = sorted(((sum((a - b) ** 2 for a, b in zip(z, r[4:])), r) for r in nn["rows"] if r[0] != exclude), key=lambda t: t[0])[:k]
    return [dict(id=r[0], name=r[1], star=r[2], kind=r[3], dist=round(math.sqrt(d), 2)) for d, r in best]


def rate(features, level_id=None):
    kind = pick(features)
    s, p = predict(features, kind)
    p50, p80 = uncertainty(s, kind)
    m = load(kind)
    sim, ref = similar(features, exclude=level_id, kind=kind), m["nn"]["ref"]
    d = sum(x["dist"] for x in sim) / len(sim)
    return dict(star=round(s, 2), tier=round(star_to_tier(s), 2) if s >= 9.75 else None, face=face(s), p_demon=round(p, 3),
                err50=round(p50, 2), err80=round(p80, 2), in_training=level_id is not None and any(r[0] == level_id for r in m["nn"]["rows"]),
                confidence="high" if d <= ref[0] else "medium" if d <= ref[1] else "low",
                engine="physics" if kind == "sim" else "structure",
                profile=[dict(key=k, label=l, pct=percentile(k, features.get(k, 0.0), kind)) for k, l in PROFILE],
                similar=sim)
