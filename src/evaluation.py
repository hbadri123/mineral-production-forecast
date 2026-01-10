from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    Symmetric MAPE in percent.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred) + eps)
    return float(100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def directional_accuracy(y_level: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Directional accuracy: whether model predicts the correct sign of change vs origin level y_t.
    """
    y_level = np.asarray(y_level, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    true_dir = np.sign(y_true - y_level)
    pred_dir = np.sign(y_pred - y_level)
    return float(np.mean(true_dir == pred_dir))


def skill_score_rmse(rmse_model: float, rmse_naive: float, eps: float = 1e-12) -> float:
    """
    Skill score relative to naive baseline using RMSE:
      skill = 1 - RMSE_model / RMSE_naive
    """
    return float(1.0 - (rmse_model / (rmse_naive + eps)))


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    verbose: bool = True


@dataclass
class BacktestResult:
    y_true: pd.Series
    y_level: pd.Series
    y_pred: pd.DataFrame
    errors: pd.DataFrame  # per split/model error messages (optional)


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_type: str,
    random_state: int = 42,
    verbose: bool = True,
    n_trials: int = 20,
) -> Optional[Dict]:
    """
    Random search for hyperparameters - much faster than grid search.
    Returns best params dict or None if tuning fails.
    """
    import random
    random.seed(random_state)
    np.random.seed(random_state)
    
    # Define parameter ranges for random sampling
    param_ranges = {
        "random_forest": {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [3, 5, 7, None],
            "min_samples_leaf": [1, 3, 5, 10],
            "min_samples_split": [2, 5, 10],
        },
        "xgboost": {
            "n_estimators": [100, 200, 400, 800],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [2, 3, 4, 5],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_alpha": [0, 0.1, 0.5, 1.0],
            "reg_lambda": [0.5, 1.0, 2.0],
        },
        "lightgbm": {
            "n_estimators": [100, 300, 600, 1200],
            "learning_rate": [0.01, 0.03, 0.05],
            "num_leaves": [15, 31, 63],
            "max_depth": [3, 5, 7],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_samples": [10, 20, 30],
        },
        "catboost": {
            "iterations": [300, 600, 1000, 1500],
            "learning_rate": [0.01, 0.03, 0.05],
            "depth": [4, 6, 8],
            "l2_leaf_reg": [1, 3, 5],
        },
    }
    
    if model_type not in param_ranges:
        return None
    
    ranges = param_ranges[model_type]
    best_params = None
    best_val_rmse = float('inf')
    
    if verbose:
        print(f"    Random search {model_type}: {n_trials} trials...")
    
    for trial in range(n_trials):
        # Randomly sample parameters
        params = {}
        for key, values in ranges.items():
            params[key] = random.choice(values)
        
        try:
            # Create model with these params
            if model_type == "random_forest":
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(
                    **params,
                    random_state=random_state,
                    n_jobs=-1,
                )
            elif model_type == "xgboost":
                from xgboost import XGBRegressor
                model = XGBRegressor(
                    **params,
                    random_state=random_state,
                    n_jobs=-1,
                    verbosity=0,
                    objective="reg:squarederror",
                )
            elif model_type == "lightgbm":
                from lightgbm import LGBMRegressor
                model = LGBMRegressor(
                    **params,
                    random_state=random_state,
                    n_jobs=-1,
                    verbosity=-1,
                    force_col_wise=True,
                )
            elif model_type == "catboost":
                from catboost import CatBoostRegressor
                iterations = params.pop("iterations", 1000)
                model = CatBoostRegressor(
                    iterations=iterations,
                    **params,
                    random_seed=random_state,
                    verbose=False,
                    allow_writing_files=False,
                )
            else:
                continue
            
            # Train and evaluate
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            val_rmse = rmse(y_val.values, y_pred)
            
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_params = params.copy()
                if model_type == "catboost":
                    best_params["iterations"] = iterations
        except Exception:
            continue
    
    if best_params and verbose:
        print(f"    Best {model_type} params: {best_params}, val_rmse={best_val_rmse:.2f}")
    
    return best_params


def create_tuned_model_factory(
    model_type: str,
    best_params: Dict,
    random_state: int = 42,
) -> Callable:
    """
    Create a model factory function with tuned hyperparameters.
    """
    from src.models import SklearnRegressorModel
    
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        def factory():
            return SklearnRegressorModel(
                name="random_forest",
                regressor=RandomForestRegressor(
                    **best_params,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            )
        return factory
    
    elif model_type == "xgboost":
        from xgboost import XGBRegressor
        def factory():
            return SklearnRegressorModel(
                name="xgboost",
                regressor=XGBRegressor(
                    **best_params,
                    random_state=random_state,
                    n_jobs=-1,
                    verbosity=0,
                    objective="reg:squarederror",
                ),
            )
        return factory
    
    elif model_type == "lightgbm":
        from lightgbm import LGBMRegressor
        def factory():
            return SklearnRegressorModel(
                name="lightgbm",
                regressor=LGBMRegressor(
                    **best_params,
                    random_state=random_state,
                    n_jobs=-1,
                    verbosity=-1,
                    force_col_wise=True,
                ),
            )
        return factory
    
    elif model_type == "catboost":
        from catboost import CatBoostRegressor
        def factory():
            # Extract iterations separately if present (use copy to avoid mutating)
            params_copy = best_params.copy()
            iterations = params_copy.pop("iterations", 1000)
            return SklearnRegressorModel(
                name="catboost",
                regressor=CatBoostRegressor(
                    iterations=iterations,
                    **params_copy,
                    random_seed=random_state,
                    verbose=False,
                    allow_writing_files=False,
                ),
            )
        return factory
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def fit_with_early_stopping(
    model: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_name: str,
) -> None:
    """
    Fit model with early stopping for gradient boosting models.
    """
    if not hasattr(model, "regressor"):
        model.fit(X_train, y_train)
        return
    
    regressor = model.regressor
    
    if model_name == "xgboost":
        from xgboost import XGBRegressor
        if isinstance(regressor, XGBRegressor):
            regressor.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False,
            )
            return
    
    elif model_name == "lightgbm":
        from lightgbm import LGBMRegressor
        import lightgbm as lgb
        if isinstance(regressor, LGBMRegressor):
            regressor.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
            return
    
    elif model_name == "catboost":
        from catboost import CatBoostRegressor
        if isinstance(regressor, CatBoostRegressor):
            regressor.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=50,
                verbose=False,
            )
            return
    
    # Fallback to regular fit
    model.fit(X_train, y_train)


def single_split_evaluation(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    model_factories: Dict[str, Callable[[], object]],
    cfg: Optional[SplitConfig] = None,
    do_tune_hyperparameters: bool = False,
    random_state: int = 42,
) -> BacktestResult:
    """
    Single train/validation/test split evaluation.
    
    - Splits data into train, validation, and test sets
    - Trains models on training set
    - Evaluates on test set
    - Returns BacktestResult compatible with existing evaluation functions
    """
    cfg = cfg or SplitConfig()
    
    # Validate split ratios
    total_ratio = cfg.train_ratio + cfg.val_ratio + cfg.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
    
    if not X.index.equals(y.index):
        raise ValueError("X and y must have identical indices (forecast origin timestamps).")
    if "y_level" not in X.columns:
        raise ValueError("X must include `y_level` for baselines and directional accuracy.")
    
    n = len(X)
    if n < 10:
        raise ValueError(f"Not enough samples for splitting: n={n}")
    
    # Calculate split indices
    train_end = int(n * cfg.train_ratio)
    val_end = train_end + int(n * cfg.val_ratio)
    
    # Split data
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]
    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]
    
    if cfg.verbose:
        print(f"  Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        print(f"  Train range: {X_train.index[0].date()} to {X_train.index[-1].date()}")
        print(f"  Val range: {X_val.index[0].date()} to {X_val.index[-1].date()}")
        print(f"  Test range: {X_test.index[0].date()} to {X_test.index[-1].date()}")
    
    # Hyperparameter tuning for ML models if requested
    tuned_factories = model_factories.copy()
    if do_tune_hyperparameters:
        ml_models = ["random_forest", "xgboost", "lightgbm", "catboost"]
        for model_name in ml_models:
            if model_name in model_factories:
                if cfg.verbose:
                    print(f"  Tuning hyperparameters for {model_name}...")
                best_params = tune_hyperparameters(
                    X_train, y_train, X_val, y_val, model_name, 
                    random_state=random_state, verbose=cfg.verbose
                )
                if best_params:
                    # Create a new factory with tuned parameters
                    tuned_factories[model_name] = create_tuned_model_factory(
                        model_name, best_params, random_state=random_state
                    )
    
    model_names = list(tuned_factories.keys())
    preds: Dict[str, List[float]] = {m: [] for m in model_names}
    errs: Dict[str, List[str]] = {m: [] for m in model_names}
    y_trues: List[float] = []
    y_levels: List[float] = []
    test_indices: List[pd.Timestamp] = []
    
    # Store test set values
    for idx in X_test.index:
        test_indices.append(idx)
        y_trues.append(float(y_test.loc[idx]))
        y_levels.append(float(X_test.loc[idx, "y_level"]))
    
    # Train and predict for each model
    for model_name, factory in tuned_factories.items():
        if cfg.verbose:
            print(f"  Training {model_name}...")
        
        try:
            model = factory()
            # Fit on training set (with early stopping for gradient boosting if applicable)
            if hasattr(model, "fit"):
                if model_name in ["xgboost", "lightgbm", "catboost"] and do_tune_hyperparameters:
                    # Use early stopping for gradient boosting models
                    fit_with_early_stopping(model, X_train, y_train, X_val, y_val, model_name)
                else:
                    model.fit(X_train, y_train)
            else:
                raise TypeError(f"Model {model_name} has no fit() method.")
            
            # Predict on test set
            if hasattr(model, "predict"):
                y_hat = model.predict(X_test)
            else:
                raise TypeError(f"Model {model_name} has no predict() method.")
            
            # Store predictions
            y_hat_array = np.asarray(y_hat).ravel()
            for val in y_hat_array:
                preds[model_name].append(float(val))
                errs[model_name].append("")
        except Exception as e:
            # If model fails, store NaN for all test predictions
            for _ in range(len(X_test)):
                preds[model_name].append(np.nan)
                errs[model_name].append(str(e))
    
    idx = pd.DatetimeIndex(test_indices, name="date")
    y_true_s = pd.Series(y_trues, index=idx, name="y_true")
    y_level_s = pd.Series(y_levels, index=idx, name="y_level")
    y_pred_df = pd.DataFrame({k: pd.Series(v, index=idx) for k, v in preds.items()})
    err_df = pd.DataFrame({k: pd.Series(v, index=idx) for k, v in errs.items()})
    
    return BacktestResult(y_true=y_true_s, y_level=y_level_s, y_pred=y_pred_df, errors=err_df)


def compute_metrics(result: BacktestResult, *, naive_model_name: str = "naive") -> pd.DataFrame:
    """
    Compute evaluation metrics per model from a BacktestResult.
    """
    rows = []
    if naive_model_name not in result.y_pred.columns:
        raise ValueError(f"Expected naive model column `{naive_model_name}` in predictions.")

    # Per-model metrics, evaluated on non-NaN overlaps with y_true.
    for model in result.y_pred.columns:
        df = pd.concat(
            [
                result.y_true.rename("y_true"),
                result.y_level.rename("y_level"),
                result.y_pred[model].rename("y_pred"),
                result.y_pred[naive_model_name].rename("y_naive"),
            ],
            axis=1,
        ).dropna()

        if df.empty:
            rows.append(
                {
                    "model": model,
                    "n": 0,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "smape": np.nan,
                    "directional_accuracy": np.nan,
                    "skill_rmse_vs_naive": np.nan,
                }
            )
            continue

        y_true = df["y_true"].to_numpy()
        y_pred = df["y_pred"].to_numpy()
        y_level = df["y_level"].to_numpy()
        y_naive = df["y_naive"].to_numpy()

        rmse_m = rmse(y_true, y_pred)
        rmse_n = rmse(y_true, y_naive)

        rows.append(
            {
                "model": model,
                "n": int(len(df)),
                "rmse": round(rmse_m, 2),
                "mae": round(mae(y_true, y_pred), 2),
                "smape": round(smape(y_true, y_pred), 2),
                "directional_accuracy": round(directional_accuracy(y_level, y_true, y_pred), 2),
                "skill_rmse_vs_naive": round(skill_score_rmse(rmse_m, rmse_n), 2),
            }
        )

    out = pd.DataFrame(rows).sort_values(by=["rmse", "model"], ascending=[True, True]).reset_index(drop=True)
    return out


def generate_ml_vs_baseline_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary comparing ML models vs baseline models.
    
    Returns a DataFrame showing:
    - For each mineral/horizon: which ML models beat baseline
    - Skill scores for ML models
    - Summary statistics
    """
    # Define model categories
    baseline_models = {"naive", "historical_mean"}
    ml_models = {"random_forest", "xgboost", "lightgbm", "catboost"}
    classical_models = {"arima", "sarima", "sarimax"}
    
    # Filter to only valid rows (with non-null skill scores)
    valid_df = metrics_df[
        metrics_df["skill_rmse_vs_naive"].notna() & 
        (metrics_df["n"] > 0)
    ].copy()
    
    rows = []
    
    # For each mineral/horizon combination
    for (mineral, horizon), group in valid_df.groupby(["mineral", "horizon"]):
        # Get baseline RMSE (naive)
        naive_row = group[group["model"] == "naive"]
        if naive_row.empty:
            continue
        baseline_rmse = naive_row.iloc[0]["rmse"]
        
        # Get ML models for this mineral/horizon
        ml_group = group[group["model"].isin(ml_models)].copy()
        
        if ml_group.empty:
            continue
        
        # Find best ML model
        best_ml = ml_group.loc[ml_group["rmse"].idxmin()]
        
        # Count how many ML models beat baseline (positive skill score)
        ml_beats_baseline = (ml_group["skill_rmse_vs_naive"] > 0).sum()
        total_ml = len(ml_group)
        
        # Average skill score for ML models
        avg_ml_skill = ml_group["skill_rmse_vs_naive"].mean()
        
        # Best ML skill score
        best_ml_skill = best_ml["skill_rmse_vs_naive"]
        
        rows.append({
            "mineral": mineral,
            "horizon": horizon,
            "baseline_rmse": round(baseline_rmse, 2),
            "best_ml_model": best_ml["model"],
            "best_ml_rmse": round(best_ml["rmse"], 2),
            "best_ml_skill_score": round(best_ml_skill, 2),
            "ml_beats_baseline_count": ml_beats_baseline,
            "total_ml_models": total_ml,
            "ml_win_rate": round(ml_beats_baseline / total_ml if total_ml > 0 else 0.0, 2),
            "avg_ml_skill_score": round(avg_ml_skill, 2),
            "improvement_pct": round((1 - best_ml["rmse"] / baseline_rmse) * 100 if baseline_rmse > 0 else 0.0, 2),
        })
    
    summary_df = pd.DataFrame(rows)
    
    # Sort by mineral, then horizon
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["mineral", "horizon"]).reset_index(drop=True)
    
    return summary_df


def plot_forecast_vs_actual(
    result: BacktestResult,
    *,
    model: str,
    title: str,
    output_path: Path,
    show_naive: bool = True,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.concat([result.y_true, result.y_pred[model].rename("pred")], axis=1)
    if show_naive and "naive" in result.y_pred.columns:
        df = pd.concat([df, result.y_pred["naive"].rename("naive")], axis=1)
    df = df.dropna()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["y_true"], label="Actual", linewidth=2)
    ax.plot(df.index, df["pred"], label=f"Pred ({model})", linewidth=1.8)
    if show_naive and "naive" in df.columns:
        ax.plot(df.index, df["naive"], label="Naive", linewidth=1.2, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Production")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_residuals(
    result: BacktestResult,
    *,
    model: str,
    title: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.concat([result.y_true, result.y_pred[model].rename("pred")], axis=1).dropna()
    resid = df["pred"] - df["y_true"]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(resid.index, resid.values, label="Residual (pred-true)")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_forecast_vs_actual_all_models(
    result: BacktestResult,
    *,
    models: List[str],
    title: str,
    output_path: Path,
    show_naive: bool = True,
) -> None:
    """
    Plot forecast vs actual for all models in a single plot.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Combine all data
    df = pd.concat([result.y_true.rename("y_true")], axis=1)
    for model in models:
        if model in result.y_pred.columns:
            df = pd.concat([df, result.y_pred[model].rename(model)], axis=1)
    
    if show_naive and "naive" in result.y_pred.columns and "naive" not in models:
        df = pd.concat([df, result.y_pred["naive"].rename("naive")], axis=1)
    
    df = df.dropna()

    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot actual
    ax.plot(df.index, df["y_true"], label="Actual", linewidth=2.5, color="black", zorder=10)
    
    # Plot each model with different colors
    colors = plt.cm.tab10(range(len(models) + (1 if show_naive and "naive" in df.columns and "naive" not in models else 0)))
    color_idx = 0
    
    for model in models:
        if model in df.columns:
            ax.plot(df.index, df[model], label=f"{model}", linewidth=1.8, alpha=0.8, color=colors[color_idx])
            color_idx += 1
    
    # Plot naive if requested and not already in models list
    if show_naive and "naive" in df.columns and "naive" not in models:
        ax.plot(df.index, df["naive"], label="Naive", linewidth=1.2, alpha=0.7, linestyle="--", color=colors[color_idx])
    
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Production")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_residuals_all_models(
    result: BacktestResult,
    *,
    models: List[str],
    title: str,
    output_path: Path,
) -> None:
    """
    Plot residuals for all models in a single plot.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate residuals for each model
    residuals_dict = {}
    
    for model in models:
        if model in result.y_pred.columns:
            df = pd.concat([result.y_true, result.y_pred[model].rename("pred")], axis=1).dropna()
            if not df.empty:
                resid = df["pred"] - df["y_true"]
                residuals_dict[model] = resid
    
    if not residuals_dict:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot residuals for each model with different colors
    colors = plt.cm.tab10(range(len(residuals_dict)))
    for idx, (model, resid) in enumerate(residuals_dict.items()):
        ax.plot(resid.index, resid.values, label=f"{model}", linewidth=1.5, alpha=0.7, color=colors[idx])
    
    ax.axhline(0.0, color="black", linewidth=1, linestyle="-", zorder=0)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def make_feature_name_readable(feature_name: str) -> str:
    """
    Convert technical feature names to more readable, human-friendly names.
    """
    # Month dummies: month_1 -> "January", month_2 -> "February", etc.
    month_match = re.match(r"^month_(\d+)$", feature_name)
    if month_match:
        month_num = int(month_match.group(1))
        month_names = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        if 1 <= month_num <= 12:
            return month_names[month_num - 1]
    
    # Production level
    if feature_name == "y_level":
        return "Production Level (Current)"
    
    # Production lags: y_lag_N -> "Production (N months ago)"
    lag_match = re.match(r"^y_lag_(\d+)$", feature_name)
    if lag_match:
        lag = int(lag_match.group(1))
        month_word = "month" if lag == 1 else "months"
        return f"Production ({lag} {month_word} ago)"
    
    # Production rolling features: y_roll_mean_N -> "Production Rolling Mean (N months)"
    roll_mean_match = re.match(r"^y_roll_mean_(\d+)$", feature_name)
    if roll_mean_match:
        window = int(roll_mean_match.group(1))
        month_word = "month" if window == 1 else "months"
        return f"Production Rolling Mean ({window} {month_word})"
    
    roll_std_match = re.match(r"^y_roll_std_(\d+)$", feature_name)
    if roll_std_match:
        window = int(roll_std_match.group(1))
        month_word = "month" if window == 1 else "months"
        return f"Production Rolling Std ({window} {month_word})"
    
    # Production percentage changes: y_pct_change_N -> "Production % Change (N months)"
    pct_match = re.match(r"^y_pct_change_(\d+)$", feature_name)
    if pct_match:
        period = int(pct_match.group(1))
        month_word = "month" if period == 1 else "months"
        return f"Production % Change ({period} {month_word})"
    
    # Price lags: price_lag_N -> "Price (N months ago)"
    price_lag_match = re.match(r"^price_lag_(\d+)$", feature_name)
    if price_lag_match:
        lag = int(price_lag_match.group(1))
        month_word = "month" if lag == 1 else "months"
        return f"Price ({lag} {month_word} ago)"
    
    # Price rolling features
    price_roll_mean_match = re.match(r"^price_roll_mean_(\d+)$", feature_name)
    if price_roll_mean_match:
        window = int(price_roll_mean_match.group(1))
        month_word = "month" if window == 1 else "months"
        return f"Price Rolling Mean ({window} {month_word})"
    
    price_roll_std_match = re.match(r"^price_roll_std_(\d+)$", feature_name)
    if price_roll_std_match:
        window = int(price_roll_std_match.group(1))
        month_word = "month" if window == 1 else "months"
        return f"Price Rolling Std ({window} {month_word})"
    
    # Price percentage changes
    price_pct_match = re.match(r"^price_pct_change_(\d+)$", feature_name)
    if price_pct_match:
        period = int(price_pct_match.group(1))
        month_word = "month" if period == 1 else "months"
        return f"Price % Change ({period} {month_word})"
    
    # Macro features: cpi_lag_N -> "CPI (N months ago)"
    cpi_lag_match = re.match(r"^cpi_lag_(\d+)$", feature_name)
    if cpi_lag_match:
        lag = int(cpi_lag_match.group(1))
        month_word = "month" if lag == 1 else "months"
        return f"Consumer Price Index ({lag} {month_word} ago)"
    
    # Exchange rate: exrate_zar_usd_lag_N -> "ZAR/USD Exchange Rate (N months ago)"
    exrate_match = re.match(r"^exrate_zar_usd_lag_(\d+)$", feature_name)
    if exrate_match:
        lag = int(exrate_match.group(1))
        month_word = "month" if lag == 1 else "months"
        return f"ZAR/USD Exchange Rate ({lag} {month_word} ago)"
    
    # Industrial production index
    ipi_match = re.match(r"^industrial_production_index_lag_(\d+)$", feature_name)
    if ipi_match:
        lag = int(ipi_match.group(1))
        month_word = "month" if lag == 1 else "months"
        return f"Industrial Production Index ({lag} {month_word} ago)"
    
    # If no pattern matches, return the original name
    return feature_name


def save_feature_importance(
    model_obj: object,
    feature_names: List[str],
    output_path: Path,
    *,
    top_k: int = 30,
) -> None:
    """
    Persist feature importances for tree models (when available).
    Saves both CSV and visualization plot.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    importances = None
    if hasattr(model_obj, "regressor") and hasattr(model_obj.regressor, "feature_importances_"):
        importances = np.asarray(model_obj.regressor.feature_importances_, dtype=float)
    elif hasattr(model_obj, "feature_importances_"):
        importances = np.asarray(model_obj.feature_importances_, dtype=float)

    if importances is None:
        return

    fi = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi = fi.sort_values("importance", ascending=False).head(top_k)
    
    # Convert feature names to readable format for display
    fi["feature_readable"] = fi["feature"].apply(make_feature_name_readable)
    
    # Save CSV with both original and readable names
    csv_path = output_path.with_suffix(".csv")
    fi_csv = fi[["feature", "importance"]].copy()  # Keep original names in CSV
    fi_csv.to_csv(csv_path, index=False)
    
    # Create visualization with readable names
    plot_path = output_path.with_suffix(".png")
    fig, ax = plt.subplots(figsize=(12, max(6, len(fi) * 0.35)))
    y_pos = np.arange(len(fi))
    ax.barh(y_pos, fi["importance"].values, align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(fi["feature_readable"].values)
    ax.invert_yaxis()  # Top feature at top
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {len(fi)} Feature Importances")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

