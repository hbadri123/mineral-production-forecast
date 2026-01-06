import numpy as np
import pandas as pd

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