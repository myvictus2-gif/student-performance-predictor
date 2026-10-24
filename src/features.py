"""
features.py - feature engineering
================================

Three new features are added to the four inputs. Each one comes from a
pattern seen during exploratory data analysis:

  log_study        = log(1 + study_hours)
                     The first hours of study help most; extra hours help less
                     ("diminishing returns"). A log curve captures that bend.

  effective_study  = study_hours * attendance / 100
                     Study time is worth more when the student also attends
                     class, so the two work together (an "interaction").

  sleep_gap        = |sleep_hours - 8|
                     Marks are highest around 7-9 hours of sleep and fall
                     with too little or too much. Distance from 8 hours turns
                     this U-shape into a straight-line relationship.

A straight-line model (Linear Regression) cannot discover curves or
interactions by itself, so these features help it most. Tree models can
find such patterns on their own, so they gain less.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config

ENGINEERED = ["log_study", "effective_study", "sleep_gap"]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds the engineered columns. Works inside a scikit-learn Pipeline."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=config.FEATURES).copy()
        if self.enabled:
            X["log_study"] = np.log1p(X["study_hours"].clip(lower=0))
            X["effective_study"] = X["study_hours"] * X["attendance"] / 100
            X["sleep_gap"] = (X["sleep_hours"] - 8).abs()
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(config.FEATURES + (ENGINEERED if self.enabled else []))
