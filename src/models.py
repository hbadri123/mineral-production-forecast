import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

class XGBoostModel:
    """XGBoost model"""
    
    def __init__(self, random_state=42):
        self.name = "xgboost"
        self.model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            random_state=random_state,
            n_jobs=-1
        )
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)

class NaiveBaseline:
    """Naive forecast: predict last value"""
    
    def __init__(self):
        self.name = "naive"
    
    def fit(self, X, y):
        return self
    
    def predict(self, X):
        if "y_level" in X.columns:
            return X["y_level"].values
        else:
            raise ValueError("Need y_level column")

class HistoricalMeanBaseline:
    """Historical mean baseline"""
    
    def __init__(self):
        self.name = "historical_mean"
        self.mean_ = None
    
    def fit(self, X, y):
        if "y_level" not in X.columns:
            raise ValueError("Need y_level column")
        self.mean_ = float(X["y_level"].mean())
        return self
    
    def predict(self, X):
        if self.mean_ is None:
            raise RuntimeError("Model not fitted")
        return np.full(len(X), self.mean_, dtype=float)

class RandomForestModel:
    """Random Forest model"""
    
    def __init__(self, random_state=42):
        self.name = "random_forest"
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1
        )
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)