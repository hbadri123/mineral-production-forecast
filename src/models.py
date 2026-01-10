from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor


def check_required_ml_dependencies() -> None:
    """
    Hard dependency policy: fail with a clear message if optional ML libraries are missing.
    Call this early from `main.py`.
    """
    missing = []
    try:
        import xgboost  # noqa: F401
    except Exception:
        missing.append("xgboost")
    try:
        import lightgbm  # noqa: F401
    except Exception:
        missing.append("lightgbm")
    try:
        import catboost  # noqa: F401
    except Exception:
        missing.append("catboost")

    if missing:
        pkgs = ", ".join(missing)
        raise ImportError(
            f"Missing required packages: {pkgs}. "
            "Install dependencies from environment.yml and retry: conda env create -f environment.yml"
        )


class BaseForecastModel:
    name: str

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseForecastModel":
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


@dataclass
class NaiveBaseline(BaseForecastModel):
    """
    Direct multi-horizon naive: predict y_{t+h} = y_t.
    Requires `X` to include `y_level` (the production level at time t).
    """

    name: str = "naive"
    y_level_col: str = "y_level"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveBaseline":
        if self.y_level_col not in X.columns:
            raise ValueError(f"NaiveBaseline requires column `{self.y_level_col}` in X.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.y_level_col].to_numpy(dtype=float)


@dataclass
class HistoricalMeanBaseline(BaseForecastModel):
    """
    Direct multi-horizon historical mean baseline: predict y_{t+h} = mean(y_{<=t}).
    Uses `y_level` from X_train to compute the mean (expanding-window style).
    """

    name: str = "historical_mean"
    y_level_col: str = "y_level"
    mean_: Optional[float] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HistoricalMeanBaseline":
        if self.y_level_col not in X.columns:
            raise ValueError(f"HistoricalMeanBaseline requires column `{self.y_level_col}` in X.")
        s = pd.to_numeric(X[self.y_level_col], errors="coerce")
        self.mean_ = float(s.dropna().mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Model not fitted.")
        return np.full(shape=(len(X),), fill_value=self.mean_, dtype=float)


@dataclass
class SklearnRegressorModel(BaseForecastModel):
    name: str
    regressor: object

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SklearnRegressorModel":
        self.regressor.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.regressor.predict(X), dtype=float)


@dataclass
class StatsmodelsARIMAModel(BaseForecastModel):
    """
    Direct-per-horizon ARIMA fit on the shifted target series (indexed by forecast origin).
    """

    name: str
    order: tuple[int, int, int] = (1, 1, 1)
    model_: object | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StatsmodelsARIMAModel":
        from statsmodels.tsa.arima.model import ARIMA

        y_clean = pd.to_numeric(y, errors="coerce").dropna()
        if len(y_clean) < 24:
            raise ValueError("Not enough training data for ARIMA.")
        self.model_ = ARIMA(y_clean, order=self.order).fit()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model not fitted.")
        # Predict one step for each provided row; evaluation calls with len(X)==1 typically.
        n = len(X)
        fc = self.model_.forecast(steps=n)
        return np.asarray(fc, dtype=float)


@dataclass
class StatsmodelsSARIMAModel(BaseForecastModel):
    """
    Direct-per-horizon SARIMA implemented via SARIMAX with seasonal_order.
    """

    name: str
    order: tuple[int, int, int] = (1, 1, 1)
    seasonal_order: tuple[int, int, int, int] = (1, 0, 1, 12)
    model_: object | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StatsmodelsSARIMAModel":
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        y_clean = pd.to_numeric(y, errors="coerce").dropna()
        if len(y_clean) < 36:
            raise ValueError("Not enough training data for SARIMA.")
        self.model_ = SARIMAX(
            y_clean,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model not fitted.")
        n = len(X)
        fc = self.model_.forecast(steps=n)
        return np.asarray(fc, dtype=float)


def _select_sarimax_exog(X: pd.DataFrame) -> pd.DataFrame:
    """
    Statsmodels SARIMAX can be numerically fragile with many exogenous features,
    so we use a conservative subset (lags + month dummies).
    """
    keep_prefixes = (
        "y_level",
        "y_lag_",
        "price_lag_",
        "cpi_lag_",
        "exrate_zar_usd_lag_",
        "industrial_production_index_lag_",
        "month_",
    )
    cols = [c for c in X.columns if c.startswith(keep_prefixes)]
    if not cols:
        # fallback to everything
        cols = list(X.columns)
    exog = X[cols].copy()
    # Ensure numeric
    for c in exog.columns:
        exog[c] = pd.to_numeric(exog[c], errors="coerce")
    return exog


@dataclass
class StatsmodelsSARIMAXModel(BaseForecastModel):
    """
    Direct-per-horizon SARIMAX fit on shifted target with exogenous regressors at forecast origin.
    """

    name: str
    order: tuple[int, int, int] = (1, 1, 1)
    seasonal_order: tuple[int, int, int, int] = (1, 0, 1, 12)
    model_: object | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StatsmodelsSARIMAXModel":
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        y_clean = pd.to_numeric(y, errors="coerce")
        exog = _select_sarimax_exog(X)

        df = pd.concat([y_clean.rename("y"), exog], axis=1).dropna()
        if len(df) < 36:
            raise ValueError("Not enough training data for SARIMAX.")

        self.model_ = SARIMAX(
            df["y"],
            exog=df.drop(columns=["y"]),
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model not fitted.")
        exog = _select_sarimax_exog(X).astype(float)
        n = len(exog)
        fc = self.model_.forecast(steps=n, exog=exog)
        return np.asarray(fc, dtype=float)


def get_model_specs(random_state: int = 42) -> Dict[str, Callable[[], BaseForecastModel]]:
    """
    Return model factories. Each factory builds a fresh model instance.
    """
    def rf_factory() -> BaseForecastModel:
        return SklearnRegressorModel(
            name="random_forest",
            regressor=RandomForestRegressor(
                n_estimators=500,
                random_state=random_state,
                n_jobs=-1,
                min_samples_leaf=1,
            ),
        )

    def xgb_factory() -> BaseForecastModel:
        from xgboost import XGBRegressor

        return SklearnRegressorModel(
            name="xgboost",
            regressor=XGBRegressor(
                n_estimators=800,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.0,
                reg_lambda=1.0,
                objective="reg:squarederror",
                random_state=random_state,
                n_jobs=-1,
                verbosity=0,  # Suppress warnings
            ),
        )

    def lgbm_factory() -> BaseForecastModel:
        from lightgbm import LGBMRegressor

        return SklearnRegressorModel(
            name="lightgbm",
            regressor=LGBMRegressor(
                n_estimators=1200,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                n_jobs=-1,
                verbosity=-1,  # Suppress warnings
                force_col_wise=True,  # Avoid deprecation warnings
            ),
        )

    def cat_factory() -> BaseForecastModel:
        from catboost import CatBoostRegressor

        return SklearnRegressorModel(
            name="catboost",
            regressor=CatBoostRegressor(
                iterations=1500,
                learning_rate=0.03,
                depth=6,
                loss_function="RMSE",
                random_seed=random_state,
                verbose=False,
                allow_writing_files=False,
            ),
        )

    return {
        "naive": lambda: NaiveBaseline(),
        "historical_mean": lambda: HistoricalMeanBaseline(),
        "arima": lambda: StatsmodelsARIMAModel(name="arima"),
        "sarima": lambda: StatsmodelsSARIMAModel(name="sarima"),
        "sarimax": lambda: StatsmodelsSARIMAXModel(name="sarimax"),
        "random_forest": rf_factory,
        "xgboost": xgb_factory,
        "lightgbm": lgbm_factory,
        "catboost": cat_factory,
    }

