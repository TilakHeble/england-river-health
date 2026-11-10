# Learn: tree-based models, from one split to SHAP

Nine notebooks covering decision trees, random forests and gradient boosting, built up from scratch on
this project's real data: 3,706 English rivers, where the question is whether the phosphate class is good.
Each idea is coded by hand first and then checked against the library implementation.

| # | Notebook | What you learn |
|---|---|---|
| 1 | [How a tree splits](01_how_a_tree_splits.ipynb) | Gini, entropy and misclassification; finding the best split by scanning sorted values; why trees ignore monotone transforms |
| 2 | [Growing and pruning](02_growing_and_pruning.ipynb) | A complete classification tree in 40 lines (matches scikit-learn exactly); overfitting with depth; cost-complexity (weakest-link) pruning |
| 3 | [Bagging and random forests](03_bagging_and_random_forests.ipynb) | Bootstrap averaging from scratch; the variance formula ρσ² + (1-ρ)σ²/B; `max_features` decorrelation; out-of-bag error and why it is optimistic for rivers |
| 4 | [Gradient boosting from scratch](04_gradient_boosting_from_scratch.ipynb) | Log-loss gradients, Newton leaf values, shrinkage, overfitting by adding trees; checked against scikit-learn |
| 5 | [XGBoost, LightGBM, CatBoost](05_xgboost_lightgbm_catboost.ipynb) | Second-order split gain and leaf weights by hand vs XGBoost's dump; leaf-wise vs depth-wise growth; CatBoost's ordered target statistics; learned missing-value directions |
| 6 | [Tuning and regularisation](06_tuning_and_regularisation.ipynb) | What each hyperparameter controls; early stopping without touching the test fold; Optuna TPE search |
| 7 | [Monotone constraints and calibration](07_monotone_constraints_and_calibration.ipynb) | Enforcing "more pollution never helps"; ICE curves before and after; reliability curves; isotonic recalibration |
| 8 | [Interpreting trees: SHAP](08_interpreting_trees_shap.ipynb) | Exact Shapley values by enumerating all 256 subsets vs TreeSHAP; credit-splitting between correlated features; PDP and ICE |
| 9 | [Spatial validation and leakage](09_spatial_validation_and_leakage.ipynb) | Spatial autocorrelation; random vs catchment-blocked folds; coordinates as a leak; target leakage through chemistry |

Run them from this folder after `python scripts/run_pipeline.py --data` has built `data/processed/`.
This project continues the portfolio's linear-model series: [linear regression](https://github.com/TilakHeble/uk-credit-stress-engine/tree/main/learn),
[logistic regression](https://github.com/TilakHeble/nhs-readmission-risk) and
[regularised regression](https://github.com/TilakHeble/uk-gdp-nowcast/tree/main/learn).

**Further reading**
* Hastie, Tibshirani and Friedman, *The Elements of Statistical Learning* (free online), ch. 9.2, 10, 15.
* Breiman, Friedman, Olshen and Stone (1984), *Classification and Regression Trees*.
* Breiman (2001), Random forests, *Machine Learning* 45.
* Friedman (2001), Greedy function approximation: a gradient boosting machine, *Annals of Statistics*.
* Chen and Guestrin (2016), XGBoost: a scalable tree boosting system, *KDD*.
* Ke et al. (2017), LightGBM: a highly efficient gradient boosting decision tree, *NeurIPS*.
* Prokhorenkova et al. (2018), CatBoost: unbiased boosting with categorical features, *NeurIPS*.
* Lundberg et al. (2020), From local explanations to global understanding with explainable AI for trees, *Nature Machine Intelligence*.
* Roberts et al. (2017), Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure, *Ecography*.
