from pathlib import Path
import pandas as pd

from src.data_loader import load_all_data, build_panel, make_features, MINERALS
from src.models import (NaiveBaseline, HistoricalMeanBaseline, RandomForestModel,
                       XGBoostModel, LightGBMModel, CatBoostModel, ARIMAModel)
from src.evaluation import train_test_evaluation

def main():
    data_dir = Path("data/raw")
    
    # Load all data
    sources = load_all_data(data_dir)
    panel = build_panel(sources)
    
    horizons = [3, 6, 12]
    all_results = []
    
    for mineral in MINERALS:
        for horizon in horizons:
            print(f"\n{mineral} - Horizon {horizon}")
            
            X, y = make_features(panel, mineral, horizon)
            
            if len(X) < 20:
                print(f"  Not enough data, skipping")
                continue
            
            # Initialize models
            models = {
                "naive": NaiveBaseline(),
                "historical_mean": HistoricalMeanBaseline(),
                "random_forest": RandomForestModel(),
                "xgboost": XGBoostModel(),
                "lightgbm": LightGBMModel(),
                "catboost": CatBoostModel(),
                "arima": ARIMAModel()
            }
            
            # Evaluate
            results = train_test_evaluation(X, y, models)
            
            # Store results
            for model_name, metrics in results.items():
                if "error" not in metrics:
                    all_results.append({
                        "mineral": mineral,
                        "horizon": horizon,
                        "model": model_name,
                        **metrics
                    })
            
            # Print summary
            print(f"  Best RMSE: {min([r['rmse'] for r in results.values() if 'rmse' in r])}")
    
    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        Path("results").mkdir(exist_ok=True)
        results_df.to_csv("results/metrics.csv", index=False)
        print(f"\nSaved results to results/metrics.csv")

if __name__ == "__main__":
    main()