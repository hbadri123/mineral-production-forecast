import numpy as np

def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))

def evaluate_model(y_true, y_pred):
    """Evaluate model predictions"""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred)
    }