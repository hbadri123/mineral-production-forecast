from pathlib import Path
import pandas as pd
import numpy as np

from src.data_loader import load_production_sales_xlsx, load_prices_xls, make_features, MINERALS
from src.models import NaiveBaseline, HistoricalMeanBaseline, RandomForestModel
from src.evaluation import evaluate_model

def main():
    data_dir = Path("data/raw")
    
    # Load data
    prod_data = load_production_sales_xlsx(data_dir / "production_sales.xlsx")
    prices = load_prices_xls(data_dir / "prices.xls")
    
    # Combine data (simple merge for now)
    panel = pd.concat([prod_data, prices], axis=1)
    panel = panel.sort_index()
    
    # Simple train/test split
    split_idx = int(len(panel) * 0.7)
    train_panel = panel.iloc[:split_idx]
    test_panel = panel.iloc[split_idx:]
    
    # Test on one mineral
    mineral = "Gold"
    horizon = 3
    
    X_train, y_train = make_features(train_panel, mineral, horizon)
    X_test, y_test = make_features(test_panel, mineral, horizon)
    
    # Train models
    models = {
        "naive": NaiveBaseline(),
        "historical_mean": HistoricalMeanBaseline(),
        "random_forest": RandomForestModel()
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = evaluate_model(y_test.values, y_pred)
        print(f"{name}: {results[name]}")

if __name__ == "__main__":
    main()