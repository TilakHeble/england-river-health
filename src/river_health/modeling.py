"""Feature sets, spatial cross-validation and model zoo for the river-health models."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, cohen_kappa_score,
                             log_loss, mean_absolute_error, roc_auc_score)
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier

SEED = 42
CATEGORICAL = ["hmwb", "altitude", "size_class", "geology"]

NATURAL = ["local_area_km2", "up_area_km2", "links_to_sea", "n_upstream_wbs", "drains_to_sea",
           "up_lc_seminatural", "up_lc_wetland_bog", "up_lc_woodland", "up_lc_water"]
SEWAGE = ["up_spill_hours_per_km2", "up_spill_count_per_km2", "local_spill_hours",
          "local_spill_count", "local_n_overflows", "local_n_overflows_60plus",
          "up_n_overflows_60plus", "local_spill_hours_stw",
          "up_stw_dwf_m3d_per_km2", "local_stw_dwf_m3d", "local_n_stw",
          "up_n_private_sewage_per_km2", "local_n_private_sewage"]
LAND = ["up_lc_arable", "up_lc_improved_grass", "up_lc_urban", "local_lc_arable",
        "local_lc_improved_grass", "local_lc_urban"]
OTHER_POINT = ["up_n_trade_effluent", "up_n_fish_farms"]
CHEMISTRY = ["chem_orthophosphate_mgP", "chem_ammonia_mgN", "chem_bod_mg", "chem_nitrate_mgN",
             "chem_do_pct_sat"]
LOCATION = ["easting", "northing"]

PRESSURE_SET = CATEGORICAL + NATURAL + SEWAGE + LAND + OTHER_POINT
FEATURE_SETS = {
    "pressures": PRESSURE_SET,
    "pressures+chemistry": PRESSURE_SET + CHEMISTRY,
    "pressures+location": PRESSURE_SET + LOCATION,
}
# features where more pressure should never raise P(good status)
DECREASING = set(SEWAGE + ["up_lc_urban", "local_lc_urban", "up_n_trade_effluent",
                           "chem_ammonia_mgN", "chem_bod_mg", "chem_orthophosphate_mgP"])

TARGETS = {
    "eco_good": ("y_eco_good", "binary", "Ecological status good or better"),
    "inv_good": ("y_inv_good", "binary", "Invertebrates good or better"),
    "inv_class": ("y_inv_code", "ordinal", "Invertebrate class (0 bad to 4 high)"),
    "phos_good": ("phos_good", "binary", "Phosphate good or better"),
    "fish_good": ("fish_good", "binary", "Fish good or better"),
}


def prepare(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    t["phos_good"] = (t["y_phos_code"] >= 3).astype(float).where(t["y_phos_code"].notna())
    t["fish_good"] = (t["y_fish_code"] >= 3).astype(float).where(t["y_fish_code"].notna())
    for c in CATEGORICAL:
        t[c] = t[c].astype("category")
    return t


def xy(t: pd.DataFrame, target: str, features: list[str]):
    col = TARGETS[target][0]
    d = t[t[col].notna()].reset_index(drop=True)
    return d[features], d[col].astype(int).to_numpy(), d["management_catchment"].to_numpy(), d


# ----------------------------------------------------------------- CV
def splits(X, y, groups, scheme="spatial", k=5, seed=SEED):
    if scheme == "spatial":
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        return list(cv.split(X, y, groups))
    if scheme == "spatial_plain":
        return list(GroupKFold(n_splits=k).split(X, y, groups))
    return list(KFold(n_splits=k, shuffle=True, random_state=seed).split(X, y))


# ----------------------------------------------------------------- models
def _num_cols(features):
    return [f for f in features if f not in CATEGORICAL]


def _onehot_pre(features, scale=False, log=False):
    num = _num_cols(features)
    steps = [SimpleImputer(strategy="median", add_indicator=True)]
    if log:
        steps.insert(0, FunctionTransformer(lambda a: np.sign(a) * np.log1p(np.abs(a)),
                                            feature_names_out="one-to-one"))
    if scale:
        steps.append(StandardScaler())
    return ColumnTransformer([
        ("num", make_pipeline(*steps), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         [f for f in features if f in CATEGORICAL]),
    ], verbose_feature_names_out=False)


def monotone_vector(features, kind="xgb"):
    return [(-1 if f in DECREASING else 0) for f in features]


def make_model(name: str, features: list[str], task="binary", params=None, monotone=False):
    params = params or {}
    n_cls = 5 if task == "ordinal" else 2
    if name == "logistic":
        return make_pipeline(_onehot_pre(features, scale=True, log=True),
                             LogisticRegression(C=params.get("C", 0.3), max_iter=3000))
    if name == "tree":
        return make_pipeline(_onehot_pre(features),
                             DecisionTreeClassifier(random_state=SEED,
                                                    max_depth=params.get("max_depth", None),
                                                    min_samples_leaf=params.get("min_samples_leaf", 20),
                                                    ccp_alpha=params.get("ccp_alpha", 0.002),
                                                    class_weight=params.get("class_weight")))
    if name == "random_forest":
        return make_pipeline(_onehot_pre(features),
                             RandomForestClassifier(n_estimators=params.get("n_estimators", 500),
                                                    min_samples_leaf=params.get("min_samples_leaf", 3),
                                                    max_features=params.get("max_features", "sqrt"),
                                                    oob_score=True, n_jobs=-1, random_state=SEED))
    if name == "extra_trees":
        return make_pipeline(_onehot_pre(features),
                             ExtraTreesClassifier(n_estimators=500, min_samples_leaf=3,
                                                  max_features="sqrt", n_jobs=-1, random_state=SEED))
    if name == "hist_gb":
        cats = [f in CATEGORICAL for f in features]
        mono = monotone_vector(features) if monotone else None
        return HistGradientBoostingClassifier(
            learning_rate=params.get("learning_rate", 0.05), max_iter=params.get("max_iter", 300),
            max_leaf_nodes=params.get("max_leaf_nodes", 15), min_samples_leaf=params.get("min_samples_leaf", 30),
            l2_regularization=params.get("l2", 1.0), categorical_features=cats,
            monotonic_cst=mono if task == "binary" else None, random_state=SEED)
    if name == "xgboost":
        import xgboost as xgb
        mono = tuple(monotone_vector(features)) if (monotone and task == "binary") else None
        kw = dict(n_estimators=params.get("n_estimators", 400), learning_rate=params.get("learning_rate", 0.03),
                  max_depth=params.get("max_depth", 4), subsample=params.get("subsample", 0.8),
                  colsample_bytree=params.get("colsample_bytree", 0.7),
                  min_child_weight=params.get("min_child_weight", 5),
                  reg_lambda=params.get("reg_lambda", 1.0), enable_categorical=True,
                  tree_method="hist", n_jobs=2, random_state=SEED, monotone_constraints=mono)
        if task == "ordinal":
            kw.update(objective="multi:softprob", num_class=n_cls)
        return xgb.XGBClassifier(**kw)
    if name == "lightgbm":
        import lightgbm as lgb
        mono = monotone_vector(features) if (monotone and task == "binary") else None
        kw = dict(n_estimators=params.get("n_estimators", 400), learning_rate=params.get("learning_rate", 0.03),
                  num_leaves=params.get("num_leaves", 15), min_child_samples=params.get("min_child_samples", 30),
                  subsample=params.get("subsample", 0.8), subsample_freq=1,
                  colsample_bytree=params.get("colsample_bytree", 0.7),
                  reg_lambda=params.get("reg_lambda", 1.0), random_state=SEED, n_jobs=2, verbose=-1)
        if mono is not None:
            kw.update(monotone_constraints=mono, monotone_constraints_method="advanced")
        return lgb.LGBMClassifier(**kw)
    if name == "catboost":
        from catboost import CatBoostClassifier
        mono = None
        if monotone and task == "binary":
            mono = {f: -1 for f in features if f in DECREASING}
        return CatBoostClassifier(iterations=params.get("iterations", 600),
                                  learning_rate=params.get("learning_rate", 0.05),
                                  depth=params.get("depth", 5), l2_leaf_reg=params.get("l2_leaf_reg", 3),
                                  cat_features=[f for f in features if f in CATEGORICAL],
                                  monotone_constraints=mono, random_seed=SEED, verbose=0, allow_writing_files=False,
                                  thread_count=2,
                                  loss_function="MultiClass" if task == "ordinal" else "Logloss")
    raise ValueError(name)


def _fit_X(name, X):
    if name == "catboost":
        X = X.copy()
        for c in X.columns:
            if str(X[c].dtype) == "category":
                X[c] = X[c].astype(str)
    return X


# ----------------------------------------------------------------- evaluation
def oof_predict(name, features, X, y, folds, task="binary", params=None, monotone=False):
    """Out-of-fold predicted probabilities (n,) for binary or (n, 5) for ordinal."""
    P = np.zeros((len(y), 5)) if task == "ordinal" else np.zeros(len(y))
    for tr, te in folds:
        m = make_model(name, features, task, params, monotone)
        Xtr, Xte = _fit_X(name, X.iloc[tr]), _fit_X(name, X.iloc[te])
        m.fit(Xtr, y[tr])
        p = m.predict_proba(Xte)
        if task == "ordinal":
            full = np.zeros((len(te), 5))
            full[:, list(getattr(m, "classes_", range(p.shape[1])))] = p
            P[te] = full
        else:
            P[te] = p[:, 1]
    return P


def binary_metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "log_loss": log_loss(y, p), "brier": brier_score_loss(y, p), "base_rate": y.mean()}


def ordinal_metrics(y, P):
    pred = P.argmax(1)
    expected = (P * np.arange(5)).sum(1)
    return {"qwk": cohen_kappa_score(y, pred, weights="quadratic"),
            "qwk_expected": cohen_kappa_score(y, np.clip(np.rint(expected), 0, 4).astype(int),
                                              weights="quadratic"),
            "accuracy": (pred == y).mean(), "mae_expected": mean_absolute_error(y, expected),
            "log_loss": log_loss(y, np.clip(P, 1e-6, 1), labels=range(5))}


def evaluate(name, features, X, y, folds, task="binary", params=None, monotone=False):
    P = oof_predict(name, features, X, y, folds, task, params, monotone)
    return (binary_metrics(y, P) if task == "binary" else ordinal_metrics(y, P)), P
