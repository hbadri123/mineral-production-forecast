from pathlib import Path
import pandas as pd

from src.data_loader import load_production_data, load_prices

def main():
    data_dir = Path("data/raw")
    
    # Load data
    prod_data = load_production_data(data_dir / "production_sales.xlsx")
    prices = load_prices(data_dir / "prices.xls")
    
    print("Data loaded!")
    print(f"Production shape: {prod_data.shape}")
    print(f"Prices shape: {prices.shape}")

if __name__ == "__main__":
    main()