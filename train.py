"""
train.py - train, compare and save the Student Performance Predictor
===================================================================

    python train.py

Steps
-----
1. Load and clean the data (src/data_prep.py).
2. Split: 80 % training, 20 % testing (random_state=42). The test set is only
   used once, at the very end, to report final scores.
3. Build one pipeline per model:
      fill missing values (median) -> add engineered features -> [scale] -> model
4. Tune Decision Tree and Random Forest with GridSearchCV (5-fold CV).
5. Compare all models with 5-fold cross-validation on the training set and
   pick the one with the lowest cross-validated RMSE.
6. Report test-set MAE, RMSE, R2 and "accuracy" (share of predictions within
   +/- 5 marks of the real mark), and save everything the app needs.
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src import config
from src.data_prep import load_clean
from src.features import FeatureEngineer

CV = KFold(n_splits=5, shuffle=True, random_state=config.SEED)


def make_pipeline(model, engineered: bool = True, scale: bool = False) -> Pipeline:
    steps = [("impute", SimpleImputer(strategy="median")),
             ("features", FeatureEngineer(enabled=engineered))]
    if scale:  # Linear Regression gives the same predictions without scaling,
        steps.append(("scale", StandardScaler()))  # but scaled coefficients are comparable
    steps.append(("model", model))
    return Pipeline(steps)


def accuracy_within(y_true, y_pred, tol=config.ACCURACY_TOLERANCE) -> float:
    """Share of predictions within +/- tol marks of the real mark."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= tol))


def test_metrics(y_true, y_pred) -> dict:
    return {
        "Accuracy (±5 marks)": accuracy_within(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def cv_scores(pipe, X, y) -> dict:
    cv = cross_validate(pipe, X, y, cv=CV, n_jobs=-1,
                        scoring={"rmse": "neg_root_mean_squared_error",
                                 "mae": "neg_mean_absolute_error", "r2": "r2"})
    return {"CV RMSE": float(-cv["test_rmse"].mean()), "CV RMSE std": float(cv["test_rmse"].std()),
            "CV MAE": float(-cv["test_mae"].mean()), "CV R2": float(cv["test_r2"].mean())}


def main():
    t0 = time.perf_counter()
    df, clean_report, source = load_clean()
    X, y = df[config.FEATURES], df[config.TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=config.SEED)
    print(f"Data: {source}\nRows after cleaning: {len(df)} | train {len(X_tr)} | test {len(X_te)}")

    # --- 1. Feature engineering check: Linear Regression with vs without -----
    lr_raw = cv_scores(make_pipeline(LinearRegression(), engineered=False, scale=True), X_tr, y_tr)
    lr_fe = cv_scores(make_pipeline(LinearRegression(), engineered=True, scale=True), X_tr, y_tr)
    fe_effect = {"without engineered features": lr_raw, "with engineered features": lr_fe}
    print(f"Feature engineering (Linear Regression CV RMSE): {lr_raw['CV RMSE']:.3f} -> {lr_fe['CV RMSE']:.3f}")

    # --- 2. Models ------------------------------------------------------------
    candidates = {
        "Linear Regression": (make_pipeline(LinearRegression(), scale=True), None),
        "Decision Tree": (make_pipeline(DecisionTreeRegressor(random_state=config.SEED)),
                          {"model__max_depth": [3, 5, 7, 10, None],
                           "model__min_samples_leaf": [1, 5, 10, 20]}),
        "Random Forest": (make_pipeline(RandomForestRegressor(random_state=config.SEED, n_jobs=-1)),
                          {"model__n_estimators": [200, 400],
                           "model__max_depth": [None, 8, 12],
                           "model__min_samples_leaf": [1, 3, 5]}),
    }

    rows, fitted, preds = [], {}, {}
    for name, (pipe, grid) in candidates.items():
        params = {}
        if grid:
            gs = GridSearchCV(pipe, grid, cv=CV, scoring="neg_root_mean_squared_error", n_jobs=-1)
            gs.fit(X_tr, y_tr)
            pipe = gs.best_estimator_
            params = {k.replace("model__", ""): v for k, v in gs.best_params_.items()}
        cvs = cv_scores(clone(pipe), X_tr, y_tr)
        pipe.fit(X_tr, y_tr)
        p = pipe.predict(X_te)
        tm = test_metrics(y_te, p)
        rows.append({"Model": name, "Best parameters": json.dumps(params) if params else "default", **cvs, **tm})
        fitted[name], preds[name] = pipe, p
        print(f"{name:18s} CV RMSE {cvs['CV RMSE']:.3f}±{cvs['CV RMSE std']:.3f} | test RMSE {tm['RMSE']:.3f} "
              f"MAE {tm['MAE']:.3f} R2 {tm['R2']:.3f} acc±5 {tm['Accuracy (±5 marks)']:.1%} {params}")

    comp = pd.DataFrame(rows).set_index("Model")

    # --- 3. Choose using cross-validation only --------------------------------
    best = comp["CV RMSE"].idxmin()
    reason = (f"{best} has the lowest cross-validated RMSE ({comp.loc[best, 'CV RMSE']:.2f} marks) "
              "on the training data. The test set was not used to choose.")
    print("Best model:", best)

    model = fitted[best]

    # Permutation importance on the test set, measured on the 4 original inputs
    pi = permutation_importance(model, X_te, y_te, n_repeats=20, random_state=config.SEED,
                                scoring="neg_root_mean_squared_error")
    importance = (pd.DataFrame({"feature": config.FEATURES, "importance": pi.importances_mean,
                                "std": pi.importances_std})
                    .sort_values("importance", ascending=False))

    # --- 4. Save ----------------------------------------------------------------
    joblib.dump(model, config.MODELS_DIR / "best_model.joblib")
    comp.to_csv(config.REPORTS_DIR / "model_comparison.csv")
    importance.to_csv(config.REPORTS_DIR / "feature_importance.csv", index=False)
    pd.DataFrame({"actual": y_te.to_numpy(), **{f"pred_{k}": v for k, v in preds.items()}}) \
      .to_csv(config.REPORTS_DIR / "test_predictions.csv", index=False)

    ranges = {c: {"min": float(df[c].min()), "max": float(df[c].max()),
                  "median": float(df[c].median())} for c in config.FEATURES}
    meta = {
        "data_source": source,
        "best_model": best,
        "selection_reason": reason,
        "test_metrics": {k: float(comp.loc[best, k]) for k in ["Accuracy (±5 marks)", "MAE", "RMSE", "R2"]},
        "cv_rmse": float(comp.loc[best, "CV RMSE"]),
        "accuracy_tolerance": config.ACCURACY_TOLERANCE,
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "feature_ranges": ranges,
        "feature_engineering_effect": fe_effect,
        "cleaning_report": clean_report,
    }
    (config.MODELS_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved model and reports in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
