import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

class SARIMAModel(BaseModel):
    """SARIMA model"""
    
    def __init__(self):
        self.name = "sarima"
        self.model = None
    
    def fit(self, X, y):
        self.model = SARIMAX(
            y,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, 12),
            trend="c"
        ).fit(disp=False)
        return self
    
    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model not fitted")
        n = len(X)
        return self.model.forecast(steps=n)

class SARIMAXModel(BaseModel):
    """SARIMAX model with exogenous variables"""
    
    def __init__(self):
        self.name = "sarimax"
        self.model = None
    
    def fit(self, X, y):
        # Use subset of features for exog
        exog_cols = [c for c in X.columns if c.startswith(("y_lag_", "price_lag_", "month_"))]
        if not exog_cols:
            exog_cols = X.columns[:5]  # fallback
        
        exog = X[exog_cols]
        self.model = SARIMAX(
            y,
            exog=exog,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, 12),
            trend="c"
        ).fit(disp=False)
        return self
    
    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model not fitted")
        exog_cols = [c for c in X.columns if c.startswith(("y_lag_", "price_lag_", "month_"))]
        if not exog_cols:
            exog_cols = X.columns[:5]
        exog = X[exog_cols]
        n = len(exog)
        return self.model.forecast(steps=n, exog=exog)

class BaseModel:
    """Base class for all models"""
    name: str
    
    def fit(self, X, y):
        raise NotImplementedError
    
    def predict(self, X):
        raise NotImplementedError

class NaiveBaseline(BaseModel):
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

class HistoricalMeanBaseline(BaseModel):
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

class RandomForestModel(BaseModel):
    """Random Forest model"""
    
    def __init__(self, random_state=42):
        self.name = "random_forest"
        self.model = RandomForestRegressor(
            n_estimators=500,
            random_state=random_state,
            n_jobs=-1
        )
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)

class XGBoostModel(BaseModel):
    """XGBoost model"""
    
    def __init__(self, random_state=42):
        self.name = "xgboost"
        self.model = XGBRegressor(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=4,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0
        )
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)

class LightGBMModel(BaseModel):
    """LightGBM model"""
    
    def __init__(self, random_state=42):
        self.name = "lightgbm"
        self.model = LGBMRegressor(
            n_estimators=1200,
            learning_rate=0.03,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1
        )
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)

class CatBoostModel(BaseModel):
    """CatBoost model"""
    
    def __init__(self, random_state=42):
        self.name = "catboost"
        self.model = CatBoostRegressor(
            iterations=1500,
            learning_rate=0.03,
            random_seed=random_state,
            verbose=False
        )
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)

class ARIMAModel(BaseModel):
    """ARIMA model"""
    
    def __init__(self):
        self.name = "arima"
        self.model = None
    
    def fit(self, X, y):
        self.model = ARIMA(y, order=(1, 1, 1)).fit()
        return self
    
    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model not fitted")
        n = len(X)
        return self.model.forecast(steps=n)