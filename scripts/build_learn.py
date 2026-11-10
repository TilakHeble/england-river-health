"""Build and execute the teaching notebooks in learn/ (tree-based models, basic to advanced).

    python scripts/build_learn.py           # all
    python scripts/build_learn.py 04 08     # selected
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nbtools import code, md, write_and_run  # noqa: E402

LEARN = Path(__file__).resolve().parents[1] / "learn"

SETUP = code("""
import sys, warnings
from pathlib import Path
ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
pd.set_option("display.width", 160)
plt.rcParams.update({"figure.figsize": (9, 4), "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3})
from river_health import modeling as M
t = M.prepare(pd.read_parquet(ROOT / "data" / "processed" / "river_features.parquet"))
# a small, readable working set used throughout
F = ["up_stw_dwf_m3d_per_km2", "up_spill_hours_per_km2", "up_lc_urban", "up_lc_arable",
     "up_lc_improved_grass", "up_lc_seminatural", "up_area_km2", "local_n_private_sewage"]
d = t[t.phos_good.notna()].reset_index(drop=True)
X, y = d[F].to_numpy(float), d.phos_good.to_numpy(int)
groups = d.management_catchment.to_numpy()
print(X.shape, "share with good phosphate:", y.mean().round(3))
""")

# ----------------------------------------------------------------------------- 01
L01 = [
    md("""
# 01. How a decision tree chooses a split

A classification tree asks yes/no questions of the form *"is feature j ≤ threshold s?"*. At each node
it picks the (j, s) that makes the two child nodes as **pure** as possible.

For a node with share $p$ of class 1:
* **Gini impurity** $G = 1 - p^2 - (1-p)^2 = 2p(1-p)$, the chance two random rows disagree;
* **Entropy** $H = -p\\log_2 p - (1-p)\\log_2(1-p)$;
* **Misclassification** $E = \\min(p, 1-p)$.

A split's quality is the impurity decrease
$\\Delta = I(\\text{parent}) - \\frac{n_L}{n} I(L) - \\frac{n_R}{n} I(R)$.

Target throughout: is a river's **phosphate class good or better?** (3,706 English rivers.)
"""),
    SETUP,
    code("""
p = np.linspace(0, 1, 201)
plt.plot(p, 2 * p * (1 - p), label="Gini")
plt.plot(p, np.where((p > 0) & (p < 1), -(p * np.log2(p + 1e-12) + (1 - p) * np.log2(1 - p + 1e-12)), 0) / 2,
         label="entropy / 2")
plt.plot(p, np.minimum(p, 1 - p), label="misclassification")
plt.xlabel("share of class 1 in node"); plt.legend(); plt.show()
"""),
    md("""
Gini and entropy are strictly concave, so any split that changes class shares reduces them.
Misclassification is piecewise linear and often reports zero gain for a useful split, which is why trees grow
with Gini or entropy and are only *evaluated* on error.
"""),
    md("## Finding the best split from scratch"),
    code("""
def gini(y):
    if len(y) == 0: return 0.0
    p = y.mean(); return 2 * p * (1 - p)

def best_split(x, y):
    \"\"\"Scan every midpoint between sorted unique values; O(n log n) with cumulative counts.\"\"\"
    o = np.argsort(x); xs, ys = x[o], y[o]
    n = len(y); c1 = np.cumsum(ys); nl = np.arange(1, n + 1)
    pl = c1 / nl; pr = (c1[-1] - c1) / np.maximum(n - nl, 1)
    gl, gr = 2 * pl * (1 - pl), 2 * pr * (1 - pr)
    w = gini(y) - (nl / n) * gl - ((n - nl) / n) * gr
    valid = np.r_[xs[1:] != xs[:-1], False]          # only between distinct values
    w[~valid] = -np.inf
    k = int(np.argmax(w))
    return (xs[k] + xs[k + 1]) / 2, w[k]

res = pd.DataFrame([(f, *best_split(X[:, j], y)) for j, f in enumerate(F)],
                   columns=["feature", "threshold", "gini_decrease"]).sort_values("gini_decrease", ascending=False)
res
"""),
    code("""
from sklearn.tree import DecisionTreeClassifier
stump = DecisionTreeClassifier(max_depth=1).fit(X, y)
print("sklearn root:", F[stump.tree_.feature[0]], "<=", round(stump.tree_.threshold[0], 4))
print("ours        :", res.iloc[0].feature, "<=", round(res.iloc[0].threshold, 4))
"""),
    md("""
The best single question is **whether there is any built-up land upstream** (more than about 0.6% of the
catchment): 86% of the 800 rivers below that threshold are good for phosphate, against 38% of the rest.
Close behind are sewage works effluent per km² and arable share. Our scan and scikit-learn agree exactly.
"""),
    code("""
j = F.index(res.iloc[0].feature); s = res.iloc[0].threshold
left, right = y[X[:, j] <= s], y[X[:, j] > s]
print(f"left : n={len(left):5d}  P(good)={left.mean():.2f}")
print(f"right: n={len(right):5d}  P(good)={right.mean():.2f}")
xs = np.sort(np.unique(X[:, j]))[::10]
gains = [gini(y) - (X[:, j] <= v).mean() * gini(y[X[:, j] <= v]) - (X[:, j] > v).mean() * gini(y[X[:, j] > v]) for v in xs]
plt.plot(xs, gains); plt.xscale("symlog", linthresh=1e-3); plt.axvline(s, color="r", ls="--")
plt.xlabel(F[j] + " (symlog)"); plt.ylabel("Gini decrease"); plt.show()
"""),
    md("""
**Things to notice**
* Trees only care about the **order** of a feature, so log transforms, scaling and outliers make no
  difference to the splits (unlike the logistic regression baseline).
* The gain curve is flat near its peak, so small changes in the data can move the threshold. That is the
  first sign of trees' instability (notebook 03).
"""),
]

# ----------------------------------------------------------------------------- 02
L02 = [
    md("""
# 02. Growing a tree, overfitting, and cost-complexity pruning

A tree is grown greedily: split the root, then split each child, recursively, until a stopping rule
fires. Grown to the end it fits the training data perfectly and generalises badly.
"""),
    SETUP,
    md("## A complete tree in 40 lines"),
    code("""
class Node:
    def __init__(self, p, n, feature=None, threshold=None, left=None, right=None):
        self.p, self.n, self.feature, self.threshold, self.left, self.right = p, n, feature, threshold, left, right

def grow(X, y, depth=0, max_depth=3, min_leaf=20):
    node = Node(y.mean(), len(y))
    if depth == max_depth or len(y) < 2 * min_leaf or y.min() == y.max():
        return node
    best = (0.0, None, None)
    parent = 2 * y.mean() * (1 - y.mean())
    for j in range(X.shape[1]):
        o = np.argsort(X[:, j]); xs, ys = X[o, j], y[o]
        n = len(ys); c1 = np.cumsum(ys); nl = np.arange(1, n + 1)
        pl = c1 / nl; pr = (c1[-1] - c1) / np.maximum(n - nl, 1)
        gain = parent - nl / n * 2 * pl * (1 - pl) - (n - nl) / n * 2 * pr * (1 - pr)
        ok = np.r_[xs[1:] != xs[:-1], False] & (nl >= min_leaf) & (n - nl >= min_leaf)
        gain[~ok] = -1
        k = int(np.argmax(gain))
        if gain[k] > best[0]:
            best = (gain[k], j, (xs[k] + xs[k + 1]) / 2)
    if best[1] is None:
        return node
    _, j, s = best
    m = X[:, j] <= s
    node.feature, node.threshold = j, s
    node.left = grow(X[m], y[m], depth + 1, max_depth, min_leaf)
    node.right = grow(X[~m], y[~m], depth + 1, max_depth, min_leaf)
    return node

def predict(node, x):
    while node.feature is not None:
        node = node.left if x[node.feature] <= node.threshold else node.right
    return node.p

def show(node, indent=""):
    if node.feature is None:
        print(f"{indent}leaf: P(good)={node.p:.2f} (n={node.n})"); return
    print(f"{indent}{F[node.feature]} <= {node.threshold:.3g}?  (n={node.n})")
    show(node.left, indent + "  yes: "); show(node.right, indent + "  no:  ")

root = grow(X, y, max_depth=3, min_leaf=50)
show(root)
"""),
    code("""
from sklearn.tree import DecisionTreeClassifier
sk = DecisionTreeClassifier(max_depth=3, min_samples_leaf=50).fit(X, y)
ours = np.array([predict(root, x) for x in X])
print("max |ours - sklearn| =", np.abs(ours - sk.predict_proba(X)[:, 1]).max())
"""),
    md("## Overfitting: training vs unseen-catchment AUC"),
    code("""
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups))
rows = []
for depth in [1, 2, 3, 4, 6, 8, 12, 16, None]:
    tr_auc, te = [], np.zeros(len(y))
    for a, b in folds:
        m = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X[a], y[a])
        tr_auc.append(roc_auc_score(y[a], m.predict_proba(X[a])[:, 1])); te[b] = m.predict_proba(X[b])[:, 1]
    rows.append((str(depth), np.mean(tr_auc), roc_auc_score(y, te)))
r = pd.DataFrame(rows, columns=["max_depth", "train AUC", "held-out AUC"]); r.round(3)
"""),
    code("""
plt.plot(r["max_depth"], r["train AUC"], marker="o", label="train")
plt.plot(r["max_depth"], r["held-out AUC"], marker="o", label="held-out catchments")
plt.xlabel("max_depth"); plt.ylabel("AUC"); plt.legend(); plt.show()
"""),
    md("""
## Cost-complexity (weakest-link) pruning

Grow the full tree $T_0$, then for each $\\alpha \\ge 0$ find the subtree minimising
$$R_\\alpha(T) = R(T) + \\alpha\\,|T|,$$
where $R(T)$ is the total weighted impurity of the leaves and $|T|$ is the number of leaves. As $\\alpha$ rises,
the internal node whose collapse costs least per leaf removed,
$g(t) = \\dfrac{R(t) - R(T_t)}{|T_t| - 1}$, goes first. That yields a nested sequence of subtrees, and CV
chooses $\\alpha$.
"""),
    code("""
full = DecisionTreeClassifier(random_state=0, min_samples_leaf=5)
path = full.cost_complexity_pruning_path(X, y)
alphas = np.unique(np.quantile(path.ccp_alphas, np.linspace(0, 0.99, 30)))
rows = []
for a_ in alphas:
    te = np.zeros(len(y))
    for a, b in folds:
        m = DecisionTreeClassifier(random_state=0, min_samples_leaf=5, ccp_alpha=a_).fit(X[a], y[a])
        te[b] = m.predict_proba(X[b])[:, 1]
    leaves = DecisionTreeClassifier(random_state=0, min_samples_leaf=5, ccp_alpha=a_).fit(X, y).get_n_leaves()
    rows.append((a_, leaves, roc_auc_score(y, te)))
pr = pd.DataFrame(rows, columns=["alpha", "leaves", "held-out AUC"])
plt.plot(pr.leaves, pr["held-out AUC"], marker="o"); plt.xscale("log"); plt.xlabel("leaves"); plt.ylabel("held-out AUC")
plt.show(); pr.sort_values("held-out AUC", ascending=False).head(5).round(5)
"""),
    md("""
**Takeaways**
* Depth, minimum leaf size and $\\alpha$ all control the same thing: how finely the tree may carve the data.
* Held-out AUC peaks at depth 4 (0.77) and falls to 0.65 for the fully grown tree, which fits the training
  data perfectly. The best pruned tree has about 15 leaves (AUC 0.77).
* That is still well below an ensemble's 0.81 (notebook 03). Single trees have high **variance**.
"""),
]

# ----------------------------------------------------------------------------- 03
L03 = [
    md("""
# 03. Bagging and random forests

**Bagging** (bootstrap aggregating, Breiman 1996): fit B deep trees, each to a bootstrap resample, and
average them. If each tree has variance $\\sigma^2$ and pairwise correlation $\\rho$, the average has variance
$$\\rho\\sigma^2 + \\frac{1-\\rho}{B}\\sigma^2.$$
More trees remove the second term; only **decorrelating** the trees reduces the first. Random forests
(Breiman 2001) decorrelate by letting each split consider only a random subset of `max_features` features.
"""),
    SETUP,
    code("""
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups))

def bagged_oof(B, max_features=None, seed=0):
    rng = np.random.default_rng(seed); oof = np.zeros(len(y))
    for a, b in folds:
        P = np.zeros(len(b))
        for k in range(B):
            i = rng.choice(a, len(a), replace=True)
            m = DecisionTreeClassifier(max_features=max_features, min_samples_leaf=3,
                                       random_state=int(rng.integers(1e9))).fit(X[i], y[i])
            P += m.predict_proba(X[b])[:, 1]
        oof[b] = P / B
    return roc_auc_score(y, oof)

rows = [(B, bagged_oof(B), bagged_oof(B, max_features=3)) for B in [1, 5, 20, 60, 150]]
r = pd.DataFrame(rows, columns=["trees", "bagging (all 8 features)", "random forest (3 of 8)"]); r.round(3)
"""),
    code("""
plt.plot(r.trees, r.iloc[:, 1], marker="o", label=r.columns[1]); plt.plot(r.trees, r.iloc[:, 2], marker="o", label=r.columns[2])
plt.xscale("log"); plt.xlabel("trees"); plt.ylabel("held-out AUC (spatial CV)"); plt.legend(); plt.show()
"""),
    md("""
One deep tree scores 0.68. Averaging 20 lifts that to about 0.79-0.80, and restricting each split to 3 of 8
features adds about 0.01 more. Adding trees never overfits, it only stops helping.

## Out-of-bag estimates

Each bootstrap sample leaves out about $(1 - 1/n)^n \\approx e^{-1} \\approx 37\\%$ of rows. Scoring each
row only with the trees that did not see it gives a free validation estimate.
"""),
    code("""
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(400, min_samples_leaf=3, oob_score=True, n_jobs=-1, random_state=0).fit(X, y)
oob_auc = roc_auc_score(y, rf.oob_decision_function_[:, 1])
sp = np.zeros(len(y))
for a, b in folds:
    sp[b] = RandomForestClassifier(400, min_samples_leaf=3, n_jobs=-1, random_state=0).fit(X[a], y[a]).predict_proba(X[b])[:, 1]
print(f"OOB AUC {oob_auc:.3f}  vs  spatial-CV AUC {roc_auc_score(y, sp):.3f}")
"""),
    md("""
Here OOB says 0.824 and spatial CV 0.810. OOB is valid when rows are independent. Rivers are not: an out-of-bag water body usually has its
upstream and downstream neighbours *in* the bag, so OOB is optimistic, just like random-fold CV.
Use it for quick tuning, not for reporting.

## Variance in practice: how much do predictions move?
"""),
    code("""
rng = np.random.default_rng(1); preds_tree, preds_rf = [], []
for k in range(10):
    i = rng.choice(len(y), len(y), replace=True)
    preds_tree.append(DecisionTreeClassifier(min_samples_leaf=3, random_state=k).fit(X[i], y[i]).predict_proba(X)[:, 1])
    preds_rf.append(RandomForestClassifier(200, min_samples_leaf=3, n_jobs=-1, random_state=k).fit(X[i], y[i]).predict_proba(X)[:, 1])
print("mean SD of a river's prediction across resamples: tree", np.std(preds_tree, 0).mean().round(3),
      "| forest", np.std(preds_rf, 0).mean().round(3))
"""),
]

# ----------------------------------------------------------------------------- 04
L04 = [
    md("""
# 04. Gradient boosting from scratch

Bagging averages *independent* deep trees to cut variance. Boosting adds *small* trees **in sequence**,
each one correcting what the ensemble so far gets wrong, to cut bias.

For binary log loss with score $F(x)$ (log-odds) and $p = \\sigma(F)$:
$$\\ell(y, F) = -[y\\log p + (1-y)\\log(1-p)], \\qquad -\\frac{\\partial \\ell}{\\partial F} = y - p.$$

**Gradient boosting** (Friedman 2001):
1. Start from $F_0 = \\log\\frac{\\bar y}{1-\\bar y}$.
2. For $m = 1..M$: compute residuals $r_i = y_i - p_i$; fit a regression tree to $r$; set each leaf's
   value by one Newton step, $\\gamma = \\frac{\\sum r_i}{\\sum p_i(1-p_i)}$; update $F \\leftarrow F + \\eta\\,\\gamma$.

The learning rate $\\eta$ (shrinkage) makes each step small, so many trees share the work.
"""),
    SETUP,
    code("""
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, log_loss
sig = lambda z: 1 / (1 + np.exp(-z))

class Boost:
    def __init__(self, M=300, eta=0.05, depth=3, min_leaf=30):
        self.M, self.eta, self.depth, self.min_leaf = M, eta, depth, min_leaf
    def fit(self, X, y):
        self.F0 = np.log(y.mean() / (1 - y.mean())); F = np.full(len(y), self.F0); self.trees = []
        for m in range(self.M):
            p = sig(F); r = y - p
            tr = DecisionTreeRegressor(max_depth=self.depth, min_samples_leaf=self.min_leaf).fit(X, r)
            leaf = tr.apply(X); vals = {}
            for l in np.unique(leaf):                      # Newton step per leaf
                k = leaf == l
                vals[l] = r[k].sum() / max((p[k] * (1 - p[k])).sum(), 1e-9)
            self.trees.append((tr, vals))
            F += self.eta * np.vectorize(vals.get)(leaf)
        return self
    def staged(self, X):
        F = np.full(len(X), self.F0)
        for tr, vals in self.trees:
            F = F + self.eta * np.vectorize(vals.get)(tr.apply(X)); yield sig(F)

folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups))
a, b = folds[0]
bst = Boost(M=400, eta=0.05).fit(X[a], y[a])
tr_ll = [log_loss(y[a], p) for p in bst.staged(X[a])]
te_ll = [log_loss(y[b], p) for p in bst.staged(X[b])]
plt.plot(tr_ll, label="train"); plt.plot(te_ll, label="held-out catchments")
plt.xlabel("trees"); plt.ylabel("log loss"); plt.legend(); plt.show()
print("best number of trees on held-out fold:", int(np.argmin(te_ll)) + 1)
"""),
    md("""
Unlike a forest, boosting **can overfit by adding trees** (here the held-out minimum is near 200 trees): training loss keeps falling, held-out loss
bottoms out and turns up. The number of trees is a tuning parameter, usually set by **early stopping**.

## Learning rate vs number of trees
"""),
    code("""
for eta in [0.3, 0.1, 0.03]:
    bb = Boost(M=500, eta=eta).fit(X[a], y[a])
    plt.plot([log_loss(y[b], p) for p in bb.staged(X[b])], label=f"eta={eta}")
plt.xlabel("trees"); plt.ylabel("held-out log loss"); plt.ylim(None, 0.62); plt.legend(); plt.show()
"""),
    md("""
Smaller steps need more trees but reach a lower, flatter minimum: shrinkage is regularisation.

## Does it match a library?
"""),
    code("""
from sklearn.ensemble import GradientBoostingClassifier
oof_ours, oof_sk = np.zeros(len(y)), np.zeros(len(y))
for a_, b_ in folds:
    oof_ours[b_] = list(Boost(M=200, eta=0.05).fit(X[a_], y[a_]).staged(X[b_]))[-1]
    oof_sk[b_] = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3,
                                            min_samples_leaf=30, random_state=0).fit(X[a_], y[a_]).predict_proba(X[b_])[:, 1]
print(f"spatial-CV AUC  ours {roc_auc_score(y, oof_ours):.3f}   sklearn {roc_auc_score(y, oof_sk):.3f}")
print("correlation of predictions:", np.corrcoef(oof_ours, oof_sk)[0, 1].round(4))
"""),
]

# ----------------------------------------------------------------------------- 05
L05 = [
    md("""
# 05. XGBoost, LightGBM and CatBoost: what each adds

All three implement regularised second-order boosting. Each tree minimises a Taylor approximation
of the loss with gradients $g_i$ and Hessians $h_i$ (for log loss, $g = p - y$, $h = p(1-p)$), plus a
penalty $\\gamma|T| + \\tfrac12\\lambda\\sum_j w_j^2$ on leaves. The best leaf weight and the gain of a split are
$$w_j^* = -\\frac{G_j}{H_j + \\lambda}, \\qquad \\text{gain} = \\tfrac12\\Big[\\frac{G_L^2}{H_L+\\lambda} + \\frac{G_R^2}{H_R+\\lambda} - \\frac{(G_L+G_R)^2}{H_L+H_R+\\lambda}\\Big] - \\gamma.$$

| | XGBoost | LightGBM | CatBoost |
|---|---|---|---|
| tree growth | depth-wise (level by level) | **leaf-wise** (best leaf first) | **symmetric** (same split across a level) |
| split finding | histogram / exact | histogram + GOSS sampling | histogram, oblivious trees |
| categoricals | native (optimal partition) | native (gradient-sorted) | **ordered target statistics** |
| missing values | learned default direction | learned default direction | treated as min/max |
"""),
    SETUP,
    md("## The XGBoost split gain by hand"),
    code("""
p0 = y.mean(); g = p0 - y; h = np.full(len(y), p0 * (1 - p0)); lam = 1.0
j = 0; o = np.argsort(X[:, j]); G, H = np.cumsum(g[o]), np.cumsum(h[o])
GL, HL, GR, HR = G[:-1], H[:-1], G[-1] - G[:-1], H[-1] - H[:-1]
gain = 0.5 * (GL**2 / (HL + lam) + GR**2 / (HR + lam) - G[-1]**2 / (H[-1] + lam))
ok = X[o][1:, j] != X[o][:-1, j]
k = np.argmax(np.where(ok, gain, -np.inf))
print(f"best first split on {F[j]}: <= {(X[o][k, j] + X[o][k + 1, j]) / 2:.3f}, gain {gain[k]:.1f}")
wl, wr = -GL[k] / (HL[k] + lam), -GR[k] / (HR[k] + lam)
print(f"leaf weights (log-odds shifts): left {wl:+.3f}, right {wr:+.3f}")
"""),
    code("""
import xgboost as xgb
m = xgb.XGBClassifier(n_estimators=1, max_depth=1, learning_rate=1.0, reg_lambda=1.0, base_score=float(p0),
                      tree_method="exact", min_child_weight=0).fit(X[:, [0]], y)
print(m.get_booster().get_dump(with_stats=True)[0])
"""),
    md("""
The library finds the same threshold and the same leaf weights (`leaf=` values) as the formula. Its
reported `gain` is exactly twice ours: XGBoost's dump omits the ½. `missing=1` is the learned default
direction for missing values.

## Leaf-wise vs depth-wise growth

With the same number of leaves, LightGBM spends them where the gain is largest, so its trees can be deep and
lopsided. That makes it fast and accurate, but more prone to overfit small data unless `num_leaves` and
`min_child_samples` are limited. Below, both use a tiny minimum leaf size so the difference shows.
"""),
    code("""
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups))
def cv(model):
    o = np.zeros(len(y))
    for a, b in folds: o[b] = model.fit(X[a], y[a]).predict_proba(X[b])[:, 1]
    return roc_auc_score(y, o)
rows = []
for leaves in [4, 8, 16, 32, 64, 128]:
    rows.append(("LightGBM leaf-wise", leaves, cv(lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=leaves,
                                                                    min_child_samples=5, verbose=-1, random_state=0))))
    depth = int(np.log2(leaves))
    rows.append(("XGBoost depth-wise", leaves, cv(xgb.XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=depth,
                                                                   min_child_weight=1, random_state=0))))
r = pd.DataFrame(rows, columns=["growth", "max leaves", "AUC"])
r.pivot(index="max leaves", columns="growth", values="AUC").round(3)
"""),
    md("""
On 3,700 rows the two growth strategies score within 0.01 of each other. LightGBM peaks with about 8 leaves
and declines slowly as leaves are added, which is the mild overfitting expected from leaf-wise growth.
Shrinkage and 300 small steps keep both in check.
"""),
    md("""
## Categorical features: ordered target statistics (CatBoost)

Replacing a category with the mean target of its rows ("target encoding") leaks the row's own label into its
feature. CatBoost encodes each row using only rows *before it* in a random permutation, the same trick it
uses to compute gradients ("ordered boosting"). Below, management catchment (90 levels) is target-encoded
naively and in the ordered way.
"""),
    code("""
cat = d.management_catchment.to_numpy()
def naive_te(cat, y):
    s = pd.Series(y).groupby(cat).transform("mean").to_numpy(); return s
def ordered_te(cat, y, prior=y.mean(), a=1.0, seed=0):
    rng = np.random.default_rng(seed); o = rng.permutation(len(y)); out = np.zeros(len(y))
    sums, cnts = {}, {}
    for i in o:
        c = cat[i]; out[i] = (sums.get(c, 0) + a * prior) / (cnts.get(c, 0) + a)
        sums[c] = sums.get(c, 0) + y[i]; cnts[c] = cnts.get(c, 0) + 1
    return out
for name, enc in [("naive", naive_te(cat, y)), ("ordered", ordered_te(cat, y))]:
    print(f"{name:8s} in-sample AUC of the encoding alone: {roc_auc_score(y, enc):.3f}")
"""),
    md("""
The naive encoding scores 0.80 in-sample against 0.76 for the ordered one, because each row partly sees its own label. A tree built on it
would learn that inflated signal and disappoint on new data. (This project deliberately keeps management
catchment *out* of the features, because it defines the spatial CV groups.)

## Missing values

Trees in XGBoost and LightGBM learn a default direction for missing values at each split. Measured chemistry
is missing for 27% of rivers, and those rivers are *more* often good (0.60 vs 0.44): monitoring targets
problem rivers. The model can use missingness itself as information, which helps prediction but is a
reminder that monitoring is not random.
"""),
    code("""
chem = d.chem_orthophosphate_mgP.to_numpy()
Xc = np.c_[X, chem]
print(f"rivers without phosphate monitoring: {np.isnan(chem).mean():.0%}; "
      f"P(good) with data {y[~np.isnan(chem)].mean():.2f}, without {y[np.isnan(chem)].mean():.2f}")
for lab, XX in [("pressures", X), ("+ measured phosphate (with NaN)", Xc)]:
    o = np.zeros(len(y))
    for a, b in folds: o[b] = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, verbose=-1).fit(XX[a], y[a]).predict_proba(XX[b])[:, 1]
    print(f"{lab:35s} AUC {roc_auc_score(y, o):.3f}")
"""),
]

# ----------------------------------------------------------------------------- 06
L06 = [
    md("""
# 06. Tuning and regularisation

Boosting has many dials, but they control a few things:

| what it controls | XGBoost | LightGBM |
|---|---|---|
| tree size | `max_depth` | `num_leaves`, `max_depth` |
| leaf size / Hessian per leaf | `min_child_weight` | `min_child_samples` |
| step size | `learning_rate` | `learning_rate` |
| number of trees | `n_estimators` + early stopping | same |
| randomness | `subsample`, `colsample_bytree` | `subsample` (`bagging_fraction`), `colsample_bytree` |
| leaf-weight shrinkage | `reg_lambda` (L2), `reg_alpha` (L1), `gamma` | `reg_lambda`, `reg_alpha`, `min_split_gain` |
"""),
    SETUP,
    md("## Early stopping inside cross-validation"),
    code("""
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups))
best_iters = []
for a, b in folds:
    # split the training catchments again: never stop on the test fold
    inner = list(StratifiedGroupKFold(4, shuffle=True, random_state=1).split(X[a], y[a], groups[a]))[0]
    tr, va = a[inner[0]], a[inner[1]]
    m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.02, num_leaves=15, min_child_samples=30,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1)
    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
    best_iters.append(m.best_iteration_)
print("trees chosen by early stopping in each outer fold:", best_iters)
"""),
    md("""
Stopping on the outer test fold would make it part of training. The validation set for early stopping
must come from the training catchments.

## A one-at-a-time view of the main dials
"""),
    code("""
def cv(**kw):
    o = np.zeros(len(y))
    for a, b in folds:
        o[b] = lgb.LGBMClassifier(verbose=-1, random_state=0, **kw).fit(X[a], y[a]).predict_proba(X[b])[:, 1]
    return roc_auc_score(y, o)
base = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7)
fig, ax = plt.subplots(1, 4, figsize=(15, 3.4), sharey=True)
for a_, (k, vals) in zip(ax, [("num_leaves", [3, 7, 15, 31, 63, 127]), ("min_child_samples", [2, 10, 30, 80, 200]),
                               ("colsample_bytree", [0.2, 0.4, 0.7, 1.0]), ("reg_lambda", [0, 1, 10, 100])]):
    sc = [cv(**{**base, k: v}) for v in vals]
    a_.plot([str(v) for v in vals], sc, marker="o"); a_.set_title(k)
ax[0].set_ylabel("spatial-CV AUC"); plt.tight_layout(); plt.show()
"""),
    md("""
## Bayesian optimisation with Optuna

Optuna's TPE sampler models which regions of the search space gave good scores and samples there more
often. The project wraps it in **nested** CV (`river_health.experiments.nested_tuning`): each outer fold is
tuned only on its own training catchments, so the reported score includes the cost of tuning.
"""),
    code("""
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
def objective(trial):
    return cv(n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
              learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
              num_leaves=trial.suggest_int("num_leaves", 4, 48, log=True),
              min_child_samples=trial.suggest_int("min_child_samples", 10, 120, log=True),
              subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
              colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0))
st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
st.optimize(objective, n_trials=25)
print("best (non-nested, so optimistic) AUC:", round(st.best_value, 3)); st.best_params
"""),
    code("""
h = st.trials_dataframe()
plt.plot(h.number, h.value, "o", alpha=0.5); plt.plot(h.number, h.value.cummax(), "r-")
plt.xlabel("trial"); plt.ylabel("AUC"); plt.show()
"""),
    md("""
Tuning reaches 0.812, against about 0.808 for the hand-set defaults used elsewhere. Early stopping picks
anywhere from 174 to 515 trees depending on the fold, so the exact number of trees matters little at this
learning rate. The default-ish baseline is already close to the tuned optimum: on 3,700 rows the choice of features
and folds matters far more than the hyperparameters.
"""),
]

# ----------------------------------------------------------------------------- 07
L07 = [
    md("""
# 07. Monotonic constraints and calibration

## Monotonic constraints
A boosted model can learn that, in some narrow range, *more* sewage means a *higher* chance of good status:
noise fitted as signal, or a confounder. For a decision tool that is unacceptable. A monotone-decreasing
constraint on feature $j$ forces $f(x)$ to be non-increasing in $x_j$ with everything else held fixed.

How it is enforced: when a split on $x_j$ is considered, it is only allowed if the left child's value is
≥ the right child's (for a decreasing constraint), and bounds are passed down so that no later split in
either subtree can cross the other's values.
"""),
    SETUP,
    code("""
import lightgbm as lgb
mono = [-1, -1, -1, 0, 0, 0, 0, -1]          # sewage flow, spills, urban, private sewage: never helpful
print(dict(zip(F, mono)))
free = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=10, verbose=-1).fit(X, y)
cons = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=10, verbose=-1,
                          monotone_constraints=mono, monotone_constraints_method="advanced").fit(X, y)
j = F.index("up_spill_hours_per_km2"); grid = np.quantile(X[:, j], np.linspace(0, 0.98, 60))
fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
rng = np.random.default_rng(0)
for a_, m, title in [(ax[0], free, "unconstrained"), (ax[1], cons, "monotone")]:
    for i in rng.choice(len(y), 25, replace=False):           # ICE curves: one river at a time
        Z = np.repeat(X[[i]], len(grid), 0); Z[:, j] = grid
        a_.plot(grid, m.predict_proba(Z)[:, 1], alpha=0.5, lw=1)
    a_.set_title(title); a_.set_xlabel("spill hours per km2 upstream")
ax[0].set_ylabel("P(phosphate good)"); plt.show()
"""),
    code("""
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups))
for lab, kw in [("unconstrained", {}), ("monotone", dict(monotone_constraints=mono, monotone_constraints_method="advanced"))]:
    o = np.zeros(len(y))
    for a, b in folds:
        o[b] = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30, verbose=-1,
                                  **kw).fit(X[a], y[a]).predict_proba(X[b])[:, 1]
    print(f"{lab:14s} spatial-CV AUC {roc_auc_score(y, o):.3f}")
"""),
    md("""
The unconstrained curves wiggle up and down; the constrained ones only fall. Accuracy is unchanged
(0.810 constrained vs 0.808 free): the wiggles were noise.

## Calibration

A probability is calibrated if, among rivers given 30%, about 30% are good. Trees' raw scores often are not:
random forests pull towards 0.5 (averaging), and boosting with log loss is usually close but can be
over-confident when overfitted.
"""),
    code("""
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
oofs = {}
for lab, mk in [("random forest", lambda: RandomForestClassifier(300, min_samples_leaf=1, n_jobs=-1, random_state=0)),
                ("LightGBM", lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, verbose=-1)),
                ("LightGBM overfit", lambda: lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.1, num_leaves=63,
                                                                min_child_samples=2, verbose=-1))]:
    o = np.zeros(len(y))
    for a, b in folds: o[b] = mk().fit(X[a], y[a]).predict_proba(X[b])[:, 1]
    oofs[lab] = o
    fy, fx = calibration_curve(y, o, n_bins=10, strategy="quantile")
    plt.plot(fx, fy, marker="o", label=f"{lab} (Brier {brier_score_loss(y, o):.3f})")
plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("predicted"); plt.ylabel("observed"); plt.legend(); plt.show()
"""),
    md("""
## Fixing calibration: isotonic regression on out-of-fold scores

The deliberately overfitted booster is badly over-confident. Isotonic regression learns a monotone map
from score to observed frequency, fitted on other folds' out-of-fold scores.
"""),
    code("""
from sklearn.isotonic import IsotonicRegression
o = oofs["LightGBM overfit"]; fixed = np.zeros(len(y))
for a, b in folds:                                    # fit the calibrator on other folds' OOF scores
    fixed[b] = IsotonicRegression(out_of_bounds="clip").fit(o[a], y[a]).predict(o[b])
print(f"Brier before {brier_score_loss(y, o):.3f}  after {brier_score_loss(y, fixed):.3f}")
"""),
]

# ----------------------------------------------------------------------------- 08
L08 = [
    md("""
# 08. Interpreting tree ensembles: importance, partial dependence and SHAP

Three questions, three tools:
* **Which features matter overall?** Impurity importance (fast, biased) or permutation importance (honest, but splits credit between correlated features).
* **How does the prediction change with one feature?** Partial dependence (average) and ICE (one row at a time).
* **Why this prediction for this river?** SHAP values.
"""),
    SETUP,
    code("""
import lightgbm as lgb
m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30, verbose=-1).fit(X, y)
"""),
    md("""
## Shapley values

Treat the features as players in a game whose payout is the model's prediction. Feature $j$'s Shapley value
is its average marginal contribution over all orders in which features could be added:
$$\\phi_j = \\sum_{S \\subseteq N\\setminus\\{j\\}} \\frac{|S|!\\,(|N|-|S|-1)!}{|N|!}\\big[v(S\\cup\\{j\\}) - v(S)\\big].$$
They are the unique attribution that is *efficient* (contributions sum to prediction minus average),
*symmetric*, gives *zero to dummies* and is *additive*. In SHAP's interventional form, $v(S)$ is the mean
prediction when features in $S$ are fixed at this row's values and the others are drawn from a background sample.

With 8 features there are only $2^8 = 256$ subsets, so we can compute exact values by brute force and
compare them with the library.
"""),
    code("""
from itertools import combinations
from math import factorial
f = lambda Z: m.predict(Z, raw_score=True)          # log-odds
bg = X[np.random.default_rng(0).choice(len(X), 100, replace=False)]
def v(row, S):
    Z = bg.copy(); Z[:, list(S)] = row[list(S)]; return f(Z).mean()
def shapley(row):
    n = len(row); phi = np.zeros(n)
    for j in range(n):
        others = [k for k in range(n) if k != j]
        for r in range(n):
            for S in combinations(others, r):
                w = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
                phi[j] += w * (v(row, S + (j,)) - v(row, S))
    return phi
i = int(np.argmax(d.up_stw_dwf_m3d_per_km2.values))  # the river with most effluent per km2
phi = shapley(X[i])
import shap
sv = shap.TreeExplainer(m, data=bg, feature_perturbation="interventional").shap_values(X[[i]])[0]
pd.DataFrame({"feature": F, "value": X[i], "brute force": phi, "TreeSHAP": sv}).round(4)
"""),
    code("""
print("sum of contributions:", phi.sum().round(4), "= prediction - mean:", (f(X[[i]])[0] - f(bg).mean()).round(4))
"""),
    md("""
Brute force and TreeSHAP agree to four decimals. TreeSHAP gets the same answer in polynomial time by walking the trees
instead of enumerating subsets.

This river (the most effluent-heavy in England) is pushed down mainly by its 40% urban catchment. Note that
this *unconstrained* model gives its 94 spill hours per km² a **positive** contribution, which is physically
implausible. That is the kind of artefact the monotone constraints in notebook 07 remove.

## Correlated features share credit
"""),
    code("""
X2 = np.c_[X, X[:, 0] * np.exp(np.random.default_rng(1).normal(0, 0.05, len(X)))]   # a near-copy of sewage flow
m2 = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30, verbose=-1).fit(X2, y)
s1 = np.abs(shap.TreeExplainer(m).shap_values(X)).mean(0)
s2 = np.abs(shap.TreeExplainer(m2).shap_values(X2)).mean(0)
print(f"mean |SHAP| for sewage flow: alone {s1[0]:.3f}; with a near-copy {s2[0]:.3f} + copy {s2[-1]:.3f} = {s2[0] + s2[-1]:.3f}")
"""),
    md("""
Add a near-duplicate and the credit is split between the two (0.38 + 0.19), while their sum (0.57) is about what the
original had alone (0.56). This is why the project reports **SHAP summed over pressure groups** (all spill features
together, all sewage-works features together) rather than ranking single correlated features.

## Partial dependence and ICE
"""),
    code("""
from sklearn.inspection import PartialDependenceDisplay
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
PartialDependenceDisplay.from_estimator(m, X, [0, 1, 2], feature_names=F, kind="both", subsample=60,
                                        ice_lines_kw={"alpha": 0.2}, ax=ax, random_state=0)
plt.tight_layout(); plt.show()
"""),
    md("""
PDP averages over the data, including combinations that never occur (a mountain stream with the effluent
of a city). With strongly correlated features, prefer **accumulated local effects** (ALE) or SHAP dependence
plots, which only look at realistic neighbourhoods.
"""),
]

# ----------------------------------------------------------------------------- 09
L09 = [
    md("""
# 09. Spatial validation and leakage

Rivers are not independent rows. Water bodies on the same river share upstream pressures, geology,
climate and monitoring teams. A random train/test split puts neighbours on both sides, and a flexible model
can score well by recognising *places* rather than learning *relationships*.
"""),
    SETUP,
    md("## How similar are neighbours?"),
    code("""
from scipy.spatial import cKDTree
xy = d[["easting", "northing"]].to_numpy(); tree = cKDTree(xy)
dist, idx = tree.query(xy, k=6)
same = (y[idx[:, 1:]] == y[:, None]).mean()
print(f"share of 5 nearest neighbours with the same phosphate outcome: {same:.2f} (chance: {y.mean()**2 + (1-y.mean())**2:.2f})")
"""),
    md("## Random vs spatial folds"),
    code("""
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import KFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
Xl = np.c_[X, d[["easting", "northing"]].to_numpy()]      # coordinates make the leak visible
schemes = {"random 5-fold": list(KFold(5, shuffle=True, random_state=0).split(Xl)),
           "spatial 5-fold": list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xl, y, groups))}
models = {"logistic": lambda: make_pipeline(FunctionTransformer(np.log1p), StandardScaler(), LogisticRegression(max_iter=2000)),
          "LightGBM": lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=5, verbose=-1)}
rows = []
for feats, XX in [("pressures", X), ("pressures + coordinates", Xl)]:
    for sname, folds in schemes.items():
        for mname, mk in models.items():
            if mname == "logistic" and feats != "pressures": continue
            o = np.zeros(len(y))
            for a, b in folds: o[b] = mk().fit(XX[a], y[a]).predict_proba(XX[b])[:, 1]
            rows.append((feats, mname, sname, roc_auc_score(y, o)))
r = pd.DataFrame(rows, columns=["features", "model", "cv", "AUC"])
r.pivot_table(index=["features", "model"], columns="cv", values="AUC").round(3)
"""),
    md("""
Neighbouring rivers share their phosphate outcome 72% of the time, against 50% by chance.
With random folds, LightGBM given coordinates scores 0.878. With whole catchments held out it scores 0.839:
about half of the coordinates' apparent gain was the model recognising places. The other half is regional
structure the pressures miss (soils, rainfall, livestock). Note also that random folds flatter LightGBM
(+0.015) much more than the logistic model (+0.003). The question the project answers is *"what happens in a river the model
has not seen?"*, so all reported scores use spatial folds.

## Target leakage: features that are the target in disguise

The phosphate **class** is defined by comparing measured phosphate with a standard for the river type. Using
measured phosphate as a feature is close to using the label.
"""),
    code("""
chem = d.chem_orthophosphate_mgP.to_numpy()
folds = schemes["spatial 5-fold"]
for lab, XX in [("pressures only", X), ("+ measured phosphate", np.c_[X, chem])]:
    o = np.zeros(len(y))
    for a, b in folds: o[b] = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, verbose=-1).fit(XX[a], y[a]).predict_proba(XX[b])[:, 1]
    print(f"{lab:22s} AUC {roc_auc_score(y, o):.3f}")
"""),
    md("""
AUC jumps from 0.81 to 0.89. A big jump is not a better model of *why* rivers fail. It answers a different, nearly circular question.
The project therefore models from **pressures**, which are things a regulator can act on, and uses chemistry only
as a diagnostic.

## Other leaks to watch for in this kind of data
* **Time.** Pressures must precede the outcome. Spill features use 2021-24 for the 2025 classification;
  2025 spills are kept aside.
* **Grouping variables as features.** Management catchment defines the CV groups; target-encoding it would
  smuggle in neighbours' labels.
* **Preprocessing on all data.** Imputation, scaling and encoding are fitted inside each training fold
  (scikit-learn pipelines).
"""),
]

ALL = {"01": ("01_how_a_tree_splits.ipynb", L01), "02": ("02_growing_and_pruning.ipynb", L02),
       "03": ("03_bagging_and_random_forests.ipynb", L03), "04": ("04_gradient_boosting_from_scratch.ipynb", L04),
       "05": ("05_xgboost_lightgbm_catboost.ipynb", L05), "06": ("06_tuning_and_regularisation.ipynb", L06),
       "07": ("07_monotone_constraints_and_calibration.ipynb", L07), "08": ("08_interpreting_trees_shap.ipynb", L08),
       "09": ("09_spatial_validation_and_leakage.ipynb", L09)}

if __name__ == "__main__":
    pick = sys.argv[1:] or list(ALL)
    for k in pick:
        name, cells = ALL[k]
        write_and_run(cells, LEARN / name)
