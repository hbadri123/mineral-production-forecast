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
    
    # Map tickers to minerals
    ticker_to_mineral = {
        "PGOLD": "Gold",
        "PCOAL": "Coal",
        "PIORECR": "Iron ore",
        "PCOPP": "Copper"
    }
    
    keep = ["date"]
    for ticker, mineral in ticker_to_mineral.items():
        if ticker in df.columns:
            keep.append(ticker)
    
    out = df[keep].set_index("date").sort_index()
    
    # Rename columns
    rename = {ticker: f"price_{mineral}" 
              for ticker, mineral in ticker_to_mineral.items() 
              if ticker in out.columns}
    out = out.rename(columns=rename)
    
    # Convert to numeric
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    
    return out

def make_features(panel, mineral, horizon=3):
    """Create features for forecasting"""
    y_name = f"production_{mineral}"
    if y_name not in panel.columns:
        raise ValueError(f"Missing {y_name}")
    
    y = panel[y_name]
    
    features = {}
    features["y_level"] = y
    
    # Production lags
    for lag in [1, 3, 6, 12]:
        features[f"y_lag_{lag}"] = y.shift(lag)
    
    # Price features
    price_col = f"price_{mineral}"
    if price_col in panel.columns:
        p = panel[price_col]
        for lag in [1, 2, 3]:
            features[f"price_lag_{lag}"] = p.shift(lag)
    
    # Target
    y_target = y.shift(-horizon)
    
    # Combine
    X = pd.DataFrame(features, index=panel.index)
    
    # Drop rows with missing values
    valid = X.notna().all(axis=1) & y_target.notna()
    X = X.loc[valid]
    y_out = y_target.loc[valid]
    
    return X, y_out

def load_macro_data(filepath, series_name):
    """Load macro economic data from excel"""
    df = pd.read_excel(filepath, sheet_name="Monthly")
    
    # Find date column
    date_col = None
    for col in df.columns:
        if "date" in str(col).lower() or "observation" in str(col).lower():
            date_col = col
            break
    if date_col is None:
        date_col = df.columns[0]
    
    # Convert to datetime
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.set_index("date").sort_index()
    
    # Find value column (first numeric column)
    for col in df.columns:
        if col != date_col:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                if df[col].notna().sum() > 0:
                    out = df[[col]].rename(columns={col: series_name})
                    return out
            except:
                continue
    
    raise ValueError(f"Could not find numeric column in {filepath}")

def load_all_data(data_dir):
    """Load all data sources"""
    data_dir = Path(data_dir)
    
    sources = {}
    sources["production"] = load_production_sales_xlsx(data_dir / "production_sales.xlsx")
    sources["prices"] = load_prices_xls(data_dir / "prices.xls")
    sources["cpi"] = load_macro_data(data_dir / "cpi.xlsx", "cpi")
    sources["exrate"] = load_macro_data(data_dir / "EXSFUS.xlsx", "exrate_zar_usd")
    sources["ipi"] = load_macro_data(data_dir / "industrial_production_index.xlsx", 
                                      "industrial_production_index")
    
    return sources