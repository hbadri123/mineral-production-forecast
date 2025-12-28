import numpy as np
import pandas as pd

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