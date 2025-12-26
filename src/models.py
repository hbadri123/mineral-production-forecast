import numpy as np
import pandas as pd

class NaiveBaseline:
    """Naive forecast: predict last value"""
    
    def __init__(self):
        self.name = "naive"
    
    def fit(self, X, y):
        return self
    
    def predict(self, X):
        # Assume X has a column with current value
        if "y_level" in X.columns:
            return X["y_level"].values
        else:
            raise ValueError("Need y_level column")