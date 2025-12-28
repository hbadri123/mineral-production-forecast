import pandas as pd
import re
from pathlib import Path

MINERALS = ["Gold", "Coal", "Iron ore", "Copper"]

def parse_month_code(month_code):
    """Parse month code like 'MO011980' into date"""
    if month_code is None:
        return None
    s = str(month_code).strip()
    try:
        m = re.match(r"MO(\d{2})(\d{4})", s)
        if m:
            month = int(m.group(1))
            year = int(m.group(2))
            return pd.Period(f"{year}-{month:02d}", freq="M")
    except:
        return None
    return None

def load_production_sales_xlsx(path, section_col="H04", mineral_col="H05"):
    """Load production data"""
    df = pd.read_excel(path)
    
    # Find month columns
    month_cols = [col for col in df.columns if re.match(r"MO\d{6}", str(col))]
    if not month_cols:
        raise ValueError("No month columns found")
    
    id_cols = [c for c in df.columns if c not in month_cols]
    melted = df.melt(id_vars=id_cols, value_vars=month_cols, 
                     var_name="month_code", value_name="value")
    
    # Filter to production section
    melted[section_col] = melted[section_col].astype(str).str.strip()
    melted[mineral_col] = melted[mineral_col].astype(str).str.strip()
    
    melted = melted.loc[melted[section_col] == "Physical volume of mining production"]
    melted = melted.loc[melted[mineral_col].isin(MINERALS)]
    
    # Parse dates
    periods = melted["month_code"].apply(parse_month_code)
    melted = melted.loc[periods.notna()]
    melted["date"] = periods.apply(lambda p: p.to_timestamp() if p else None)
    melted = melted.dropna(subset=["date"])
    
    # Pivot
    pivot = melted.pivot_table(index="date", columns=mineral_col, 
                               values="value", aggfunc="mean")
    pivot = pivot.rename(columns={m: f"production_{m}" for m in pivot.columns})
    
    return pivot

def load_prices_xls(path):
    """Load price data"""
    df = pd.read_excel(path, engine="xlrd")
    df = df.iloc[3:].reset_index(drop=True)
    
    date_col = df.columns[0]
    df[date_col] = df[date_col].astype(str)
    periods = df[date_col].apply(parse_month_code)
    df = df.loc[periods.notna()]
    df["date"] = periods.apply(lambda p: p.to_timestamp() if p else None)
    df = df.dropna(subset=["date"])
    
    # Keep price columns
    price_tickers = ["PGOLD", "PCOAL", "PIORECR", "PCOPP"]
    keep = ["date"] + [c for c in price_tickers if c in df.columns]
    out = df[keep].set_index("date").sort_index()
    
    return out

def make_features(panel, mineral, horizon=3):
    """Create features for forecasting"""
    y_name = f"production_{mineral}"
    if y_name not in panel.columns:
        raise ValueError(f"Missing {y_name}")
    
    y = panel[y_name]
    
    features = {}
    features["y_level"] = y
    
    # Add lags
    for lag in [1, 3, 6, 12]:
        features[f"y_lag_{lag}"] = y.shift(lag)
    
    # Target
    y_target = y.shift(-horizon)
    
    # Combine
    X = pd.DataFrame(features, index=panel.index)
    
    # Drop rows with missing values
    valid = X.notna().all(axis=1) & y_target.notna()
    X = X.loc[valid]
    y_out = y_target.loc[valid]
    
    return X, y_out