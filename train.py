"""Train the rating model from data/levels.db and write model.json  (needs numpy, scipy, scikit-learn).

    python train.py             # features are cached in the db; they refresh automatically when gdlevel.py changes
    python train.py --quick     # skip the depth comparison
    python train.py --skip-base # keep model.json, only (re)train the physics-aware model
    python train.py --skip-sim  # structural model only (use after adding levels that have no simulation features)

The physics-simulation features (table simf, filled by tools/simrun.py) train a second model, model_sim.json. It is only
kept when it beats the structural model on the same levels in the same grouped cross-validation.

Design: a classifier says how likely the level is a demon; one regressor is a specialist for normal levels
(1-9 stars) and one for demons (10+); the answer is their probability-weighted blend, then calibrated so
the extremes are not squeezed toward the middle. Everything is measured with out-of-fold predictions
grouped by creator, so no author appears on both sides of a split.
"""
import datetime, hashlib, json, multiprocessing as mp, os, sqlite3, statistics, sys, time, zlib
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import gdlevel, model as M

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "data", "levels.db")
DEMONS = ["Easy Demon", "Medium Demon", "Hard Demon", "Insane Demon", "Extreme Demon"]
# Pure decoration volume is a trap: in the corpus big, heavily decorated levels are mostly demons, so a model that
# reads it would call every lavishly decorated easy level a demon. Gameplay and gimmick (move/logic/toggle) signals stay.
DROP = ("obj_", "deco_", "n_obj", "log_obj", "solid_frac", "sim_n_solid", "trig_color", "trig_camera", "trig_audio", "trig_other", "n_trig", "trig_per_s")
REG = dict(n_estimators=400, learning_rate=0.04, max_depth=3, subsample=0.7, min_samples_leaf=6, max_features=0.5)
CLF = dict(n_estimators=250, learning_rate=0.05, max_depth=3, subsample=0.7, min_samples_leaf=6, max_features=0.5)


def _feat(job):
    lid, blob = job
    try:
        text = zlib.decompress(blob, 47).decode("utf-8", "replace")
    except zlib.error:
        text = blob.decode("utf-8", "replace")
    try:
        return lid, json.dumps({k: float(v) for k, v in gdlevel.analyze(text)["features"].items()})
    except Exception:                                       # one odd level must not sink the run
        return lid, None


def featurize(db):
    ver = hashlib.md5(open(gdlevel.__file__, "rb").read() + open(os.path.join(HERE, "objclasses.json"), "rb").read()).hexdigest()
    db.execute("create table if not exists feats(id integer primary key, f text)")
    db.execute("create table if not exists kv(k text primary key, v text)")
    db.commit()
    if (db.execute("select v from kv where k='ver'").fetchone() or [None])[0] != ver:
        db.execute("delete from feats")
        db.execute("insert or replace into kv values('ver', ?)", (ver,))
        db.commit()
    ids = [r[0] for r in db.execute("select id from levels where id not in (select id from feats)")]
    if not ids:
        return
    t0 = time.time()
    with mp.Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        for a in range(0, len(ids), 48):                   # small batches: bounded memory, short write locks (downloaders share the db)
            batch = [(i, db.execute("select ldata from levels where id=?", (i,)).fetchone()[0]) for i in ids[a:a + 48]]
            db.executemany("insert or replace into feats values(?,?)", pool.map(_feat, batch))
            db.commit()
            print(f"\rfeaturized {min(a + 48, len(ids))}/{len(ids)} ({time.time() - t0:.0f}s)", end="", flush=True)
    print()


def sim_names(db):
    cols = set()
    for (f,) in db.execute("select f from simf"):
        cols.update(json.loads(f))
    return sorted(cols)


def dataset(db, sim=False, only=None):
    """sim=True adds the simulation features (levels without them are skipped); `only` restricts to a set of level ids."""
    X, y, w, g, info, names, prof = [], [], [], [], [], None, {}
    sc = sim_names(db) if sim else []
    q = "select l.id, l.kind, l.meta, l.gddl, f.f, " + ("s.f" if sim else "null") + " from levels l join feats f using(id)" + (" join simf s using(id)" if sim else "")
    for lid, kind, meta, gddl, f, sf in db.execute(q):
        if f is None or (only is not None and lid not in only):
            continue
        meta, f = json.loads(meta), json.loads(f)
        if sim:
            sf = json.loads(sf)
            f = dict(f, **{k: float(sf.get(k, 0.0)) for k in sc})          # sim_ok = 0 and every other sim value 0 where nothing was simulated
        if meta["difficulty"] == "Auto":                    # auto levels play themselves, so their hazards say nothing about skill
            continue
        if kind == "demon":
            gd = json.loads(gddl) if gddl else None
            tier, votes = (gd["tier"], gd["votes"]) if gd else (meta.get("tier"), None)
            if tier is None or (votes is not None and votes < 5):
                continue
            label, wt = M.tier_to_star(tier), 0.5 + 0.5 * min(votes if votes is not None else 15, 30) / 30
        else:
            if meta["stars"] < 1 or meta["stars"] > 9:
                continue
            label, wt = float(meta["stars"]), 1.0
        names = names or [k for k in f if not k.startswith(DROP)]
        X.append([f[k] for k in names]); y.append(label); w.append(wt); g.append(meta.get("creator_id", 0))
        for k, _ in M.PROFILE:
            prof.setdefault(k, []).append(f[k])
        info.append((lid, meta["name"], meta["difficulty"], label))
    X = np.nan_to_num(np.array(X, float), nan=0.0, posinf=1e6, neginf=-1e6)
    dataset.prof = {k: np.array(v) for k, v in prof.items()}      # display-only columns for the skill profile
    return names, X, np.array(y), np.array(w), np.array(g), info


# ---------------------------------------------------------------- model
def fit_two_stage(X, y, w, seed, reg=REG, clf=CLF):
    d = y >= 10
    wc = w * np.where(d, 0.5 / max(d.mean(), 1e-6), 0.5 / max(1 - d.mean(), 1e-6))      # equal prior: the corpus is mostly demons, real levels are not
    return (GradientBoostingClassifier(random_state=seed, **clf).fit(X, d, sample_weight=wc),
            GradientBoostingRegressor(loss="squared_error", random_state=seed, **reg).fit(X[~d], y[~d], sample_weight=w[~d]),
            GradientBoostingRegressor(loss="squared_error", random_state=seed, **reg).fit(X[d], y[d], sample_weight=w[d]))


def predict_two_stage(ms, X):
    c, rn, rd = ms
    p = c.predict_proba(X)[:, 1]
    return (1 - p) * rn.predict(X) + p * rd.predict(X), p


def cv_oof(X, y, w, g, seed=0, **kw):
    oof, poof = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        oof[te], poof[te] = predict_two_stage(fit_two_stage(X[tr], y[tr], w[tr], seed, **kw), X[te])
    return oof, poof


def calib_fit(p, y, k=20):
    """Monotone piecewise-linear map from raw predictions to the typical truth: undoes the squeeze toward the middle.
    The normal range (< 10) and the demon range get their own k bins: with bins over everything, two thirds of which are
    demons, the whole 1-9 star range fell into a few wide bins and normal levels got worse (1.18 -> 1.27). Each bin maps to
    the median truth, which is what the absolute error rewards (mean bins are pulled by the demons rated far too low)."""
    xs, ys = [], []
    for m in (p < 10, p >= 10):
        if m.sum() < 2 * k:
            continue
        pp, yy = p[m], y[m]
        order = np.argsort(pp)
        xs += [pp[i].mean() for i in np.array_split(order, k)]
        ys += [float(np.median(yy[i])) for i in np.array_split(order, k)]
    xs, ys = np.array(xs), np.maximum.accumulate(np.array(ys))
    s0 = float(np.clip((ys[1] - ys[0]) / max(xs[1] - xs[0], 1e-6), .8, 2.5))
    s1 = float(np.clip((ys[-1] - ys[-2]) / max(xs[-1] - xs[-2], 1e-6), .8, 2.5))
    return [float(xs[0] - 4)] + xs.tolist() + [float(xs[-1] + 4)], [float(ys[0] - 4 * s0)] + ys.tolist() + [float(ys[-1] + 4 * s1)]


def calib_apply(p, xs, ys):
    return np.interp(p, xs, ys)


def report(oof, poof, y):
    err, dem = np.abs(oof - y), y >= 10
    mean = lambda m: round(float(err[m].mean()), 3) if m.any() else 0.0
    return dict(n=len(y), mae=mean(np.ones(len(y), bool)), rmse=round(float(np.sqrt((err ** 2).mean())), 3),
                r2=round(1 - float(((oof - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()), 3), spearman=round(float(spearmanr(oof, y)[0]), 3),
                spearman_demon=round(float(spearmanr(oof[dem], y[dem])[0]), 3), spearman_normal=round(float(spearmanr(oof[~dem], y[~dem])[0]), 3),
                auc_demon=round(float(roc_auc_score(dem, poof)), 3),
                mae_normal=mean(~dem), mae_demon=mean(dem), mae_demon_tiers=round(mean(dem) * 4, 2),
                within_half=round(float((err <= 0.5).mean()), 3), within_one=round(float((err <= 1).mean()), 3),
                n_normal=int((~dem).sum()), n_demon=int(dem.sum()))


def export_tree(t):
    t = t.tree_
    r = lambda a: [float(f"{v:.6g}") for v in a]
    return [t.children_left.tolist(), t.children_right.tolist(), t.feature.tolist(), r(t.threshold), r(t.value[:, 0, 0])]


def export_gbm(est, x0, clf=False):
    raw = est.decision_function(x0)[0] if clf else est.predict(x0)[0]
    init = float(raw - est.learning_rate * sum(e[0].predict(x0)[0] for e in est.estimators_))
    return dict(init=init, lr=est.learning_rate, trees=[export_tree(e[0]) for e in est.estimators_])


def build(names, X, y, w, g, info, quick, path, kind):
    """Depth search, grouped cross-validation, calibration, bagged final fit and export. Returns (metrics, out-of-fold predictions)."""
    reg, clf = REG, CLF
    if not quick:
        best = None
        for depth in (3, 4, 5):
            r = dict(REG, max_depth=depth, min_samples_leaf=6 + 3 * (depth - 3))
            oof, _ = cv_oof(X, y, w, g, reg=r)
            mae = float(np.abs(oof - y).mean())
            print(f"  [{kind}] depth {depth}: CV MAE {mae:.3f}")
            if best is None or mae < best[0]:
                best = (mae, r)
        reg = best[1]
    oof, poof = cv_oof(X, y, w, g, reg=reg, clf=clf)
    cal = np.zeros(len(y))                                          # honest calibration estimate: fit on other folds only
    for tr, te in GroupKFold(5).split(X, y, g):
        cal[te] = calib_apply(oof[te], *calib_fit(oof[tr], y[tr]))
    raw_m, cal_m = report(oof, poof, y), report(cal, poof, y)
    print(f"[{kind}] raw        :", raw_m)
    print(f"[{kind}] calibrated :", cal_m)
    use_cal = cal_m["mae"] < raw_m["mae"] - 0.005
    metrics, final_oof = (cal_m, cal) if use_cal else (raw_m, oof)
    cx, cy = calib_fit(oof, y) if use_cal else ([0.0, 25.0], [0.0, 25.0])

    models, fitted = [], []
    for seed in range(3):                                           # small bag: smooths single-model quirks
        ms = fit_two_stage(X, y, w, seed, reg=reg, clf=clf)
        fitted.append(ms)
        models.append(dict(clf=export_gbm(ms[0], X[:1], True), rn=export_gbm(ms[1], X[:1]), rd=export_gbm(ms[2], X[:1])))
    imp = np.mean([0.5 * m[1].feature_importances_ + 0.5 * m[2].feature_importances_ for m in fitted], axis=0)
    top = [names[i] for i in np.argsort(-imp)[:14]]
    print(f"[{kind}] top features:", ", ".join(f"{n} {imp[names.index(n)]:.3f}" for n in top[:10]))

    edges, err_bins, lo = [3, 5, 7, 9, 10.5, 12, 13.5, 15, 16.5, 18, 99], [], -99
    for e in edges:
        m = (final_oof >= lo) & (final_oof < e)
        if m.sum() >= 25:
            a = np.abs(final_oof - y)[m]
            err_bins.append([e, round(float(np.quantile(a, .5)), 3), round(float(np.quantile(a, .8)), 3)])
            lo = e
        elif err_bins:
            err_bins[-1][0] = e
    med = {d: statistics.median(c[3] for c in info if c[2] == d) for d in DEMONS if any(c[2] == d for c in info)}
    cuts = [[round((med[a] + med[b]) / 2, 2), a] for a, b in zip(DEMONS, DEMONS[1:]) if a in med and b in med]
    quant = {k: [float(f"{v:.5g}") for v in np.quantile(dataset.prof[k], np.linspace(0, 1, 101))] for k, _ in M.PROFILE}
    ti = [names.index(c) for c in top]
    mu, sd = X[:, ti].mean(0), X[:, ti].std(0) + 1e-9
    Z = (X[:, ti] - mu) / sd
    knn = []                                                        # leave-one-out mean distance to the 5 nearest training levels
    for a in range(0, len(Z), 400):
        d = cdist(Z[a:a + 400], Z)
        d[np.arange(len(d)), a + np.arange(len(d))] = np.inf
        knn.append(np.partition(d, 5, axis=1)[:, :5].mean(1))
    ref = [round(float(np.quantile(np.concatenate(knn), q)), 3) for q in (.5, .9)]
    nn = dict(cols=top, ref=ref, mean=[float(f"{v:.5g}") for v in mu], std=[float(f"{v:.5g}") for v in sd],
              rows=[[i[0], i[1][:40], round(i[3], 2), i[2], *[round(float(v), 2) for v in Z[k]]] for k, i in enumerate(info)])
    out = dict(features=names, models=models, calib=dict(x=cx, y=cy), err_bins=err_bins, demon_cuts=cuts, quant=quant, nn=nn,
               metrics=dict(metrics, calibrated=bool(use_cal)), importance={n: round(float(imp[names.index(n)]), 4) for n in top},
               trained=datetime.date.today().isoformat(), config=dict(reg=reg, clf=clf))
    json.dump(out, open(path, "w"), separators=(",", ":"))
    print(f"[{kind}] wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")

    M._M.pop(kind, None)                                            # export check: pure-Python == scikit-learn
    M.PATHS[kind] = path
    pred = np.array([M.predict(dict(zip(names, row)), kind)[0] for row in X[:60]])
    ref_p = np.clip(calib_apply(np.mean([predict_two_stage(ms, X[:60])[0] for ms in fitted], axis=0), cx, cy), 0.5, 21)  # calibrate the bag mean, as model.py does
    assert np.abs(pred - ref_p).max() < 0.05, np.abs(pred - ref_p).max()
    print(f"[{kind}] export check ok (max diff %.5f)" % np.abs(pred - ref_p).max())
    return metrics, final_oof


def main():
    quick = "--quick" in sys.argv
    db = sqlite3.connect(DB, timeout=120)
    featurize(db)
    names, X, y, w, g, info = dataset(db)
    print(f"{len(y)} levels, {len(names)} features; {int((y >= 10).sum())} demons / {int((y < 10).sum())} normal")
    if "--skip-base" in sys.argv and os.path.exists(M.PATHS["base"]):
        print("keeping the existing structural model (--skip-base)")
    else:
        build(names, X, y, w, g, info, quick, M.PATHS["base"], "base")
    has = db.execute("select name from sqlite_master where name='simf'").fetchone()
    done = db.execute("select count(*) from simf").fetchone()[0] if has else 0
    if "--skip-sim" in sys.argv:
        return
    if not done:
        print("no simulation features yet (run tools/simrun.py) - only the structural model was trained")
        return
    same = {i[0] for i in info} & {r[0] for r in db.execute("select id from simf")}
    print(f"\nsimulation features for {len(same)} of these levels")
    # like-for-like: the structural model on exactly the levels that have simulation features
    bn, bX, by, bw, bg, binfo = dataset(db, only=same)
    ref_path = os.path.join(HERE, "model_ref.json")
    ref_m, ref_oof = build(bn, bX, by, bw, bg, binfo, quick, ref_path, "ref")
    sn, sX, sy, sw, sg, sinfo = dataset(db, sim=True, only=same)
    print(f"{len(sy)} levels, {len(sn)} features ({len(sn) - len(bn)} from the simulation)")
    sim_m, sim_oof = build(sn, sX, sy, sw, sg, sinfo, quick, M.PATHS["sim"], "sim")
    assert [i[0] for i in binfo] == [i[0] for i in sinfo]                # same levels in the same order: the errors pair up
    d = np.abs(ref_oof - by) - np.abs(sim_oof - sy)                      # positive = the physics model is closer to the truth
    rng, groups = np.random.default_rng(0), np.unique(sg)
    members = {c: np.where(sg == c)[0] for c in groups}
    boots = []
    for _ in range(1000):                                                # resample whole creators, not single levels
        idx = np.concatenate([members[c] for c in rng.choice(groups, len(groups))])
        boots.append(d[idx].mean())
    lo, hi = np.quantile(boots, [0.025, 0.975])
    gain = float(d.mean())
    print(f"\nsame {len(sy)} levels: structural MAE {ref_m['mae']:.3f} -> with physics {sim_m['mae']:.3f} (gain {gain:+.3f}, 95% CI {lo:+.3f} to {hi:+.3f}); "
          f"spearman {ref_m['spearman']:.3f} -> {sim_m['spearman']:.3f}")
    try:
        os.remove(ref_path)
    except OSError:
        pass
    sim_m["gain_vs_structure"] = [round(gain, 3), round(float(lo), 3), round(float(hi), 3)]
    if not (gain > 0 and lo > 0):
        os.remove(M.PATHS["sim"])
        print("the physics features do not give a statistically clear gain - model_sim.json was not kept")
    else:                                                                # keep the paired-bootstrap result next to the model
        out = json.load(open(M.PATHS["sim"]))
        out["metrics"]["gain_vs_structure"] = sim_m["gain_vs_structure"]
        out["metrics"]["structure_mae_same_levels"] = ref_m["mae"]
        json.dump(out, open(M.PATHS["sim"], "w"), separators=(",", ":"))


if __name__ == "__main__":
    mp.freeze_support()
    main()
