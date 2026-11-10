"""Run every experiment behind the reports and dashboard, writing tables to outputs/.

    PYTHONPATH=src python -m river_health.experiments            # everything (~25 min on 2 cores)
    PYTHONPATH=src python -m river_health.experiments benchmark  # one stage
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from . import modeling as M

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
TAB, MOD = OUT / "tables", OUT / "models"
for d in (TAB, MOD):
    d.mkdir(parents=True, exist_ok=True)

MAIN_TARGETS = ["eco_good", "inv_good", "phos_good"]
ALL_TARGETS = MAIN_TARGETS + ["fish_good"]
MODELS = [("logistic", False), ("tree", False), ("random_forest", False), ("extra_trees", False),
          ("hist_gb", False), ("xgboost", False), ("xgboost", True), ("lightgbm", False),
          ("lightgbm", True), ("catboost", True)]


def load():
    return M.prepare(pd.read_parquet(ROOT / "data/processed/river_features.parquet"))


def _row(**kw):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in kw.items()}


# ------------------------------------------------------------------ 1. benchmark
def benchmark(t):
    ck = TAB / "bench_parts"
    ck.mkdir(exist_ok=True)
    for target in ALL_TARGETS:
        if (ck / f"{target}.csv").exists():
            continue
        rows = []
        for fs, feats in M.FEATURE_SETS.items():
            X, y, g, _ = M.xy(t, target, feats)
            schemes = ["spatial", "random"] if fs == "pressures" else ["spatial"]
            for scheme in schemes:
                folds = M.splits(X, y, g, scheme)
                for name, mono in MODELS:
                    if fs != "pressures" and name in ("extra_trees", "hist_gb", "xgboost"):
                        continue
                    t0 = time.time()
                    r, _ = M.evaluate(name, feats, X, y, folds, monotone=mono)
                    rows.append(_row(target=target, features=fs, cv=scheme, model=name,
                                     monotone=mono, n=len(y), seconds=time.time() - t0, **r))
                    print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(ck / f"{target}.csv", index=False)
    pd.concat([pd.read_csv(ck / f"{tg}.csv") for tg in ALL_TARGETS]).to_csv(TAB / "benchmark.csv", index=False)


# ------------------------------------------------------------------ 2. tuning
def _lgb_space(trial):
    return dict(n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                num_leaves=trial.suggest_int("num_leaves", 4, 48, log=True),
                min_child_samples=trial.suggest_int("min_child_samples", 10, 120, log=True),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30, log=True))


def tune(X, y, g, feats, n_trials=40, seed=M.SEED):
    folds = M.splits(X, y, g, "spatial", k=4, seed=seed)

    def obj(trial):
        p = M.oof_predict("lightgbm", feats, X, y, folds, params=_lgb_space(trial), monotone=True)
        return M.binary_metrics(y, p)["log_loss"]

    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    st.optimize(obj, n_trials=n_trials)
    return st.best_params, st


def nested_tuning(t, n_trials=25):
    """Outer 5-fold spatial CV; inner 4-fold spatial Optuna search. Honest tuned score.

    Each target is checkpointed to outputs/tables/tuning/, so an interrupted run resumes."""
    ck = TAB / "tuning"
    ck.mkdir(exist_ok=True)
    feats = M.FEATURE_SETS["pressures"]
    for target in MAIN_TARGETS:
        if (ck / f"{target}.json").exists():
            continue
        X, y, g, d = M.xy(t, target, feats)
        P, rows = np.zeros(len(y)), []
        for k, (tr, te) in enumerate(M.splits(X, y, g, "spatial")):
            params, _ = tune(X.iloc[tr], y[tr], g[tr], feats, n_trials=n_trials, seed=k)
            m = M.make_model("lightgbm", feats, params=params, monotone=True).fit(X.iloc[tr], y[tr])
            P[te] = m.predict_proba(X.iloc[te])[:, 1]
            rows.append(_row(target=target, fold=k, test_auc=roc_auc_score(y[te], P[te]),
                             **{f"p_{a}": b for a, b in params.items()}))
            print(rows[-1], flush=True)
        r = M.binary_metrics(y, P)
        rows.append(_row(target=target, fold="all", test_auc=r["roc_auc"], log_loss=r["log_loss"]))
        # final hyper-parameters for deployment: one search on all data
        params, _ = tune(X, y, g, feats, n_trials=n_trials * 2)
        pd.DataFrame({"water_body_id": d["water_body_id"], "y": y, "p_nested": P}).to_parquet(
            TAB / f"oof_nested_{target}.parquet", index=False)
        pd.DataFrame(rows).to_csv(ck / f"{target}.csv", index=False)
        (ck / f"{target}.json").write_text(json.dumps(params, indent=2))
    pd.concat([pd.read_csv(ck / f"{tg}.csv") for tg in MAIN_TARGETS]).to_csv(TAB / "nested_tuning.csv", index=False)
    best = {tg: json.loads((ck / f"{tg}.json").read_text()) for tg in MAIN_TARGETS}
    (TAB / "best_params.json").write_text(json.dumps(best, indent=2))


# ------------------------------------------------------------------ 3. champions
def champions(t):
    """Fit tuned monotone LightGBM per target; save OOF predictions and the model."""
    best = json.loads((TAB / "best_params.json").read_text())
    feats = M.FEATURE_SETS["pressures"]
    rows = []
    for target in MAIN_TARGETS:
        X, y, g, d = M.xy(t, target, feats)
        folds = M.splits(X, y, g, "spatial")
        oof = {}
        for name, mono, params in [("logistic", False, None), ("random_forest", False, None),
                                   ("lightgbm", True, best[target])]:
            r, P = M.evaluate(name, feats, X, y, folds, params=params, monotone=mono)
            oof[name] = P
            rows.append(_row(target=target, model=name + ("_tuned_mono" if mono else ""), **r))
        pd.DataFrame({"water_body_id": d["water_body_id"], "y": y, **{f"p_{k}": v for k, v in oof.items()}}
                     ).to_parquet(TAB / f"oof_{target}.parquet", index=False)
        model = M.make_model("lightgbm", feats, params=best[target], monotone=True).fit(X, y)
        joblib.dump({"model": model, "features": feats, "target": target, "params": best[target]},
                    MOD / f"lgbm_{target}.joblib")
    pd.DataFrame(rows).to_csv(TAB / "champions.csv", index=False)


# ------------------------------------------------------------------ 4. ordinal
def ordinal(t):
    """Invertebrate class 0-4: multiclass vs Frank & Hall cumulative binary decomposition."""
    import lightgbm as lgb
    feats = M.FEATURE_SETS["pressures"]
    X, y, g, d = M.xy(t, "inv_class", feats)
    folds = M.splits(X, y, g, "spatial")
    res = {}
    res["multiclass"] = M.oof_predict("lightgbm", feats, X, y, folds, task="ordinal")
    # Frank & Hall: P(y > k) for k = 0..3 from four monotone binary models
    P = np.zeros((len(y), 5))
    for tr, te in folds:
        gt = np.zeros((len(te), 4))
        for k in range(4):
            m = M.make_model("lightgbm", feats, monotone=True).fit(X.iloc[tr], (y[tr] > k).astype(int))
            gt[:, k] = m.predict_proba(X.iloc[te])[:, 1]
        gt = np.minimum.accumulate(gt, axis=1)  # enforce P(y>0) >= P(y>1) >= ...
        P[te, 0] = 1 - gt[:, 0]
        for k in range(1, 4):
            P[te, k] = gt[:, k - 1] - gt[:, k]
        P[te, 4] = gt[:, 3]
    res["frank_hall_monotone"] = P
    prior = np.bincount(y, minlength=5) / len(y)
    res["prior_only"] = np.tile(prior, (len(y), 1))
    rows = [_row(model=k, **M.ordinal_metrics(y, v)) for k, v in res.items()]
    pd.DataFrame(rows).to_csv(TAB / "ordinal_inv_class.csv", index=False)
    cm = pd.crosstab(pd.Series(y, name="observed"),
                     pd.Series(res["frank_hall_monotone"].argmax(1), name="predicted"))
    cm.to_csv(TAB / "ordinal_confusion.csv")
    print(pd.DataFrame(rows))


# ------------------------------------------------------------------ 5. single tree
def tree_study(t):
    """Cost-complexity pruning path for one decision tree, scored with spatial CV."""
    feats = M.FEATURE_SETS["pressures"]
    rows = []
    for target in MAIN_TARGETS:
        X, y, g, _ = M.xy(t, target, feats)
        folds = M.splits(X, y, g, "spatial")
        pre = M._onehot_pre(feats).fit(X)
        path = DecisionTreeClassifier(random_state=M.SEED, min_samples_leaf=5).cost_complexity_pruning_path(
            pre.transform(X), y)
        alphas = np.unique(np.quantile(path.ccp_alphas, np.linspace(0, 0.995, 25)))
        for a in alphas:
            r, _ = M.evaluate("tree", feats, X, y, folds, params={"ccp_alpha": a, "min_samples_leaf": 5})
            m = M.make_model("tree", feats, params={"ccp_alpha": a, "min_samples_leaf": 5}).fit(X, y)
            rows.append(_row(target=target, ccp_alpha=float(a), leaves=int(m[-1].get_n_leaves()),
                             depth=int(m[-1].get_depth()), **r))
        for depth in range(1, 11):
            r, _ = M.evaluate("tree", feats, X, y, folds, params={"max_depth": depth, "ccp_alpha": 0,
                                                                  "min_samples_leaf": 5})
            rows.append(_row(target=target, max_depth=depth, **r))
    pd.DataFrame(rows).to_csv(TAB / "tree_pruning.csv", index=False)


# ------------------------------------------------------------------ 6. forest
def forest_study(t):
    """OOB error vs spatial CV, number of trees, and impurity vs permutation importance."""
    from sklearn.inspection import permutation_importance
    feats = M.FEATURE_SETS["pressures"]
    ck = TAB / "forest_parts"
    ck.mkdir(exist_ok=True)
    for target in MAIN_TARGETS:
        if (ck / f"imp_{target}.csv").exists():
            continue
        rows, imp = [], []
        X, y, g, _ = M.xy(t, target, feats)
        sp, rnd = M.splits(X, y, g, "spatial"), M.splits(X, y, g, "random")
        for n in [25, 50, 100, 200, 400, 800]:
            m = M.make_model("random_forest", feats, params={"n_estimators": n}).fit(X, y)
            oob = roc_auc_score(y, m[-1].oob_decision_function_[:, 1])
            r_sp, _ = M.evaluate("random_forest", feats, X, y, sp, params={"n_estimators": n})
            r_rd, _ = M.evaluate("random_forest", feats, X, y, rnd, params={"n_estimators": n})
            rows.append(_row(target=target, n_trees=n, oob_auc=oob, random_cv_auc=r_rd["roc_auc"],
                             spatial_cv_auc=r_sp["roc_auc"]))
            print(rows[-1], flush=True)
        for mf in ["sqrt", 0.5, 1.0]:
            r, _ = M.evaluate("random_forest", feats, X, y, sp, params={"max_features": mf})
            rows.append(_row(target=target, n_trees=500, max_features=str(mf), spatial_cv_auc=r["roc_auc"]))
        # importances: impurity (biased to high-cardinality) vs permutation on held-out folds
        m = M.make_model("random_forest", feats).fit(X, y)
        names = m[0].get_feature_names_out()
        mdi = pd.Series(m[-1].feature_importances_, index=names)
        perm = []
        for tr, te in sp:
            mm = M.make_model("random_forest", feats).fit(X.iloc[tr], y[tr])
            pi = permutation_importance(mm, X.iloc[te], y[te], scoring="roc_auc", n_repeats=5,
                                        random_state=M.SEED, n_jobs=1)
            perm.append(pd.Series(pi.importances_mean, index=feats))
        perm = pd.concat(perm, axis=1).mean(axis=1)
        for f in feats:
            mdi_f = mdi[[c for c in names if c == f or c.startswith(f + "_")]].sum()
            imp.append(_row(target=target, feature=f, impurity=float(mdi_f), permutation_auc_drop=float(perm[f])))
        pd.DataFrame(rows).to_csv(ck / f"study_{target}.csv", index=False)
        pd.DataFrame(imp).to_csv(ck / f"imp_{target}.csv", index=False)
    pd.concat([pd.read_csv(ck / f"study_{tg}.csv") for tg in MAIN_TARGETS]).to_csv(TAB / "forest_study.csv", index=False)
    pd.concat([pd.read_csv(ck / f"imp_{tg}.csv") for tg in MAIN_TARGETS]).to_csv(TAB / "forest_importance.csv", index=False)


# ------------------------------------------------------------------ 7. SHAP + RNAG
GROUPS = {"Storm overflows": [f for f in M.SEWAGE if "spill" in f or "overflow" in f],
          "Sewage works": [f for f in M.SEWAGE if "stw" in f and "spill" not in f],
          "Private sewage": [f for f in M.SEWAGE if "private" in f],
          "Arable": ["up_lc_arable", "local_lc_arable"],
          "Grassland": ["up_lc_improved_grass", "local_lc_improved_grass"],
          "Urban land": ["up_lc_urban", "local_lc_urban"],
          "Trade & fish farms": M.OTHER_POINT,
          "Natural setting": M.NATURAL + M.CATEGORICAL}


def explain(t):
    import shap
    rows = []
    for target in MAIN_TARGETS:
        b = joblib.load(MOD / f"lgbm_{target}.joblib")
        X, y, g, d = M.xy(t, target, b["features"])
        sv = shap.TreeExplainer(b["model"]).shap_values(X)
        sv = sv[1] if isinstance(sv, list) else sv
        S = pd.DataFrame(sv, columns=b["features"])
        S.insert(0, "water_body_id", d["water_body_id"].values)
        S.to_parquet(TAB / f"shap_{target}.parquet", index=False)
        grp = pd.DataFrame({k: S[v].sum(axis=1) for k, v in GROUPS.items()})
        grp.insert(0, "water_body_id", d["water_body_id"].values)
        grp.to_parquet(TAB / f"shap_groups_{target}.parquet", index=False)
        # does the model blame the same pressure the EA's investigators did?
        merged = grp.merge(t[["water_body_id"] + [c for c in t.columns if c.startswith("rnag_")]],
                           on="water_body_id")
        failing = merged[y == 0]
        for grp_name, flag in [("Storm overflows", "rnag_overflow"), ("Sewage works", "rnag_sewage"),
                               ("Arable", "rnag_agriculture"), ("Grassland", "rnag_agriculture"),
                               ("Urban land", "rnag_urban")]:
            yy = failing[flag]
            if yy.nunique() == 2:
                # more negative SHAP = more blame; AUC of -SHAP for the EA flag
                rows.append(_row(target=target, group=grp_name, rnag_flag=flag, n_failing=len(failing),
                                 flag_rate=float(yy.mean()), auc=roc_auc_score(yy, -failing[grp_name])))
    pd.DataFrame(rows).to_csv(TAB / "rnag_agreement.csv", index=False)
    print(pd.DataFrame(rows))


# ------------------------------------------------------------------ 8. scenarios
SPILL_COLS = ["up_spill_hours_per_km2", "up_spill_count_per_km2", "local_spill_hours",
              "local_spill_count", "local_spill_hours_stw"]


def apply_spill_cut(X: pd.DataFrame, cut: float, high_only=False) -> pd.DataFrame:
    X = X.copy()
    for c in SPILL_COLS:
        X[c] = X[c] * (1 - cut)
    if cut >= 1:
        for c in ["local_n_overflows_60plus", "up_n_overflows_60plus"]:
            X[c] = 0
    elif cut >= 0.5:
        # halving spills takes most frequent-spilling overflows below 60 a year
        for c in ["local_n_overflows_60plus", "up_n_overflows_60plus"]:
            X[c] = np.floor(X[c] * (1 - cut))
    return X


def scenarios(t):
    rows, per_wb = [], []
    for target in MAIN_TARGETS:
        b = joblib.load(MOD / f"lgbm_{target}.joblib")
        feats = b["features"]
        d = t.copy()
        X = d[feats]
        base = b["model"].predict_proba(X)[:, 1]
        for cut in [0.25, 0.5, 0.75, 1.0]:
            p = b["model"].predict_proba(apply_spill_cut(X, cut))[:, 1]
            rows.append(_row(target=target, scenario=f"spills -{int(cut*100)}%", cut=cut,
                             expected_good_base=float(base.sum()), expected_good=float(p.sum()),
                             extra_good_wbs=float((p - base).sum()),
                             wbs_with_overflows=int((d["up_n_overflows"] > 0).sum())))
            if cut == 1.0:
                per_wb.append(pd.DataFrame({"water_body_id": d["water_body_id"], "target": target,
                                            "p_base": base, "p_no_spills": p, "gain": p - base}))
        # comparator: urban diffuse pressure (not a policy lever, a yardstick)
        Xu = X.copy()
        for c in ["up_lc_urban", "local_lc_urban"]:
            Xu[c] = Xu[c] * 0.5
        pu = b["model"].predict_proba(Xu)[:, 1]
        rows.append(_row(target=target, scenario="urban land -50% (yardstick)", cut=np.nan,
                         expected_good_base=float(base.sum()), expected_good=float(pu.sum()),
                         extra_good_wbs=float((pu - base).sum())))
        Xs = X.copy()
        for c in ["up_stw_dwf_m3d_per_km2", "local_stw_dwf_m3d"]:
            Xs[c] = Xs[c] * 0.5
        ps = b["model"].predict_proba(Xs)[:, 1]
        rows.append(_row(target=target, scenario="sewage works effluent -50% (yardstick)", cut=np.nan,
                         expected_good_base=float(base.sum()), expected_good=float(ps.sum()),
                         extra_good_wbs=float((ps - base).sum())))
    pd.DataFrame(rows).to_csv(TAB / "scenarios.csv", index=False)
    pd.concat(per_wb).to_parquet(TAB / "scenario_per_wb.parquet", index=False)
    print(pd.DataFrame(rows))


STAGES = {"benchmark": benchmark, "tune": nested_tuning, "champions": champions, "ordinal": ordinal,
          "tree": tree_study, "forest": forest_study, "explain": explain, "scenarios": scenarios}

if __name__ == "__main__":
    t = load()
    for s in (sys.argv[1:] or list(STAGES)):
        t0 = time.time()
        STAGES[s](t)
        print(f"== {s} done in {time.time() - t0:.0f}s", flush=True)
