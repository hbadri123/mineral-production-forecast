import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_forecasts(y_true, y_pred_dict, title="Forecast vs Actual", save_path=None):
    """Plot forecasts vs actual"""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.plot(y_true.index, y_true.values, label="Actual", linewidth=2)
    
    for name, pred in y_pred_dict.items():
        ax.plot(y_true.index, pred, label=name, alpha=0.7)
    
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Production")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()

def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))

def smape(y_true, y_pred):
    """Symmetric MAPE"""
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-8
    return 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom)

def evaluate_model(y_true, y_pred, y_level=None):
    """Evaluate model predictions"""
    metrics = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "smape": smape(y_true, y_pred)
    }
    
    # Directional accuracy if y_level provided
    if y_level is not None:
        true_dir = np.sign(y_true - y_level)
        pred_dir = np.sign(y_pred - y_level)
        metrics["directional_accuracy"] = np.mean(true_dir == pred_dir)
    
    return metrics

def train_test_evaluation(X, y, models, train_ratio=0.7):
    """Evaluate models with train/test split"""
    n = len(X)
    split_idx = int(n * train_ratio)
    
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    results = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results[name] = evaluate_model(y_test.values, y_pred, 
                                           y_level=X_test["y_level"].values)
        except Exception as e:
            results[name] = {"error": str(e)}
    
    return results


def skill_score(rmse_model, rmse_naive):
    """Skill score relative to naive baseline"""
    return 1.0 - (rmse_model / rmse_naive)

def evaluate_model(y_true, y_pred, y_level=None, y_naive=None):
    """Evaluate model predictions"""
    metrics = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "smape": smape(y_true, y_pred)
    }
    
    # Directional accuracy if y_level provided
    if y_level is not None:
        true_dir = np.sign(y_true - y_level)
        pred_dir = np.sign(y_pred - y_level)
        metrics["directional_accuracy"] = np.mean(true_dir == pred_dir)
    
    # Skill score if naive predictions provided
    if y_naive is not None:
        rmse_naive = rmse(y_true, y_naive)
        metrics["skill_rmse_vs_naive"] = skill_score(metrics["rmse"], rmse_naive)
    
    return metrics