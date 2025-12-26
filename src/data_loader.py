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

def load_production_data(filepath):
    """Load production data from excel file"""
    df = pd.read_excel(filepath)
    # TODO: need to filter and process properly
    return df

def load_prices(filepath):
    """Load price data"""
    df = pd.read_excel(filepath, engine="xlrd")
    # Skip first 3 rows
    df = df.iloc[3:].reset_index(drop=True)
    return df