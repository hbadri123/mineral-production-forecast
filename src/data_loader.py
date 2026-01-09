from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, Tuple, Optional

import numpy as np
import pandas as pd


MINERALS = ["Gold", "Coal", "Iron ore", "Copper"]

PRICE_TICKER_TO_MINERAL = {
    "PGOLD": "Gold",
    "PCOAL": "Coal",
    "PIORECR": "Iron ore",
    "PCOPP": "Copper",
}

MACRO_FILE_TO_NAME = {
    "cpi.xlsx": "cpi",
    "EXSFUS.xlsx": "exrate_zar_usd",
    "industrial_production_index.xlsx": "industrial_production_index",
}


@dataclass(frozen=True)
class DataConfig:
    start: str = "1990-01-01"
    end: str = "2002-12-01"
    freq: str = "MS"  # month start
    interpolate_limit: int = 3  # short gaps

    production_sales_file: str = "production_sales.xlsx"
    prices_file: str = "prices.xls"

    # Production vs sales selector (user-provided)
    production_section_value: str = "Physical volume of mining production"
    section_col: str = "H04"
    mineral_col: str = "H05"


def _month_index(cfg: DataConfig) -> pd.DatetimeIndex:
    return pd.date_range(cfg.start, cfg.end, freq=cfg.freq)


def parse_month_code(month_code: object) -> pd.Period | None:
    """
    Parse month code like 'MO011980' or '1990M1' into a monthly Period.
    """
    if month_code is None:
        return None
    s = str(month_code).strip()
    if not s:
        return None
    try:
        m = re.match(r"MO(\d{2})(\d{4})", s)
        if m:
            month = int(m.group(1))
            year = int(m.group(2))
            return pd.Period(f"{year}-{month:02d}", freq="M")
        m = re.match(r"(\d{4})M(\d{1,2})", s)
        if m:
            year = int(m.group(1))
            month = int(m.group(2))
            return pd.Period(f"{year}-{month:02d}", freq="M")
    except Exception:
        return None
    return None


def _to_month_start(dt: pd.Series) -> pd.Series:
    """Convert datetime series to month-start timestamps."""
    dt = pd.to_datetime(dt, errors="coerce")
    # Convert to period and then to timestamp (defaults to month start)
    return dt.dt.to_period("M").dt.to_timestamp()


def load_prices_xls(path: Path) -> pd.DataFrame:
    """
    Load `prices.xls` where the first column contains month codes and price tickers are columns.
    The first 3 rows are metadata.
    """
    # `prices.xls` is legacy Excel; pandas needs `xlrd` for xls.
    df = pd.read_excel(path, engine="xlrd")
    df = df.iloc[3:].reset_index(drop=True).copy()

    date_col = df.columns[0]
    df[date_col] = df[date_col].astype(str)
    periods = df[date_col].apply(parse_month_code)
    df = df.loc[periods.notna()].copy()
    # Convert Period to month-start timestamp
    df["date"] = periods.apply(lambda p: p.to_timestamp() if p is not None else None)
    df = df.dropna(subset=["date"])

    keep = ["date"] + [c for c in PRICE_TICKER_TO_MINERAL.keys() if c in df.columns]
    out = df[keep].copy()
    out = out.set_index("date").sort_index()

    # Rename to stable feature names
    rename = {ticker: f"price_{mineral}" for ticker, mineral in PRICE_TICKER_TO_MINERAL.items() if ticker in out.columns}
    out = out.rename(columns=rename)
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def load_production_sales_xlsx(path: Path, cfg: DataConfig) -> pd.DataFrame:
    """
    Load `production_sales.xlsx` which is wide with month-coded columns `MOmmYYYY`.
    Filter to production only using cfg.section_col == cfg.production_section_value.
    Return a monthly time series of production per mineral: columns `production_<Mineral>`.
    """
    df = pd.read_excel(path)
    month_cols = [col for col in df.columns if re.match(r"MO\d{6}", str(col))]
    if not month_cols:
        raise ValueError(f"No month columns found in {path.name}")

    id_cols = [c for c in df.columns if c not in month_cols]
    melted = df.melt(id_vars=id_cols, value_vars=month_cols, var_name="month_code", value_name="value")

    # Filter to production section + target minerals
    if cfg.section_col not in melted.columns:
        raise ValueError(f"Expected section column {cfg.section_col} not found in {path.name}")
    if cfg.mineral_col not in melted.columns:
        raise ValueError(f"Expected mineral column {cfg.mineral_col} not found in {path.name}")

    melted[cfg.section_col] = melted[cfg.section_col].astype(str).str.strip()
    melted[cfg.mineral_col] = melted[cfg.mineral_col].astype(str).str.strip()

    melted = melted.loc[melted[cfg.section_col] == cfg.production_section_value].copy()
    melted = melted.loc[melted[cfg.mineral_col].isin(MINERALS)].copy()

    periods = melted["month_code"].apply(parse_month_code)
    melted = melted.loc[periods.notna()].copy()
    # Convert Period to month-start timestamp
    melted["date"] = periods.apply(lambda p: p.to_timestamp() if p is not None else None)
    melted = melted.dropna(subset=["date"])

    melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
    pivot = (
        melted.pivot_table(index="date", columns=cfg.mineral_col, values="value", aggfunc="mean")
        .sort_index()
        .copy()
    )
    pivot = pivot.rename(columns={m: f"production_{m}" for m in pivot.columns})
    return pivot


def load_fred_monthly_xlsx(path: Path, series_name: str) -> pd.DataFrame:
    """
    Load FRED-style excel with a `Monthly` sheet and a date column.
    Returns a single-column monthly series indexed by month start.
    """
    df = pd.read_excel(path, sheet_name="Monthly")
    if df.empty:
        raise ValueError(f"Empty sheet in {path.name}")

    date_col = None
    for c in df.columns:
        cl = str(c).lower()
        if "date" in cl or "observation" in cl:
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]

    dt = _to_month_start(df[date_col])
    tmp = df.copy()
    tmp["date"] = dt
    tmp = tmp.dropna(subset=["date"]).set_index("date").sort_index()

    # Find first numeric column other than the date col
    candidates = [c for c in tmp.columns if c != date_col]
    value_col = None
    for c in candidates:
        s = pd.to_numeric(tmp[c], errors="coerce")
        if s.notna().sum() > 0:
            value_col = c
            tmp[c] = s
            break
    if value_col is None:
        raise ValueError(f"Could not find numeric value column in {path.name}")

    out = tmp[[value_col]].rename(columns={value_col: series_name})
    return out


def load_raw_sources(data_dir: Path, cfg: DataConfig | None = None) -> Dict[str, pd.DataFrame]:
    cfg = cfg or DataConfig()
    data_dir = Path(data_dir)

    prod_path = data_dir / cfg.production_sales_file
    prices_path = data_dir / cfg.prices_file

    sources: Dict[str, pd.DataFrame] = {}
    sources["production"] = load_production_sales_xlsx(prod_path, cfg)
    sources["prices"] = load_prices_xls(prices_path)

    for fn, series_name in MACRO_FILE_TO_NAME.items():
        p = data_dir / fn
        sources[series_name] = load_fred_monthly_xlsx(p, series_name=series_name)

    return sources


def build_monthly_panel(sources: Dict[str, pd.DataFrame], cfg: DataConfig | None = None) -> pd.DataFrame:
    """
    Create a single monthly panel indexed by month start containing:
    - production_<mineral>
    - price_<mineral>
    - macro series
    Filtered to 1990–2002 inclusive and aligned by calendar month.
    Uses intersection of available months across all sources for robustness.
    """
    cfg = cfg or DataConfig()
    target_range = _month_index(cfg)

    # Find intersection of available months across all sources (within target range)
    available_months = None
    parts = []
    for name, df in sources.items():
        tmp = df.copy()
        if not isinstance(tmp.index, pd.DatetimeIndex):
            raise ValueError(f"Source {name} must have a DatetimeIndex")
        # Filter to target range
        tmp = tmp.sort_index().loc[(tmp.index >= target_range.min()) & (tmp.index <= target_range.max())]
        # Find months with actual data (non-null in at least one column)
        has_data = tmp.notna().any(axis=1)
        source_months = tmp.index[has_data]
        if available_months is None:
            available_months = set(source_months)
        else:
            available_months = available_months.intersection(set(source_months))
        parts.append(tmp)

    if available_months is None or len(available_months) == 0:
        raise ValueError("No overlapping months found across all data sources in the target date range")

    # Use intersection of months, but still align to full target range for consistency
    # (missing months will be NaN and handled by interpolation)
    idx = target_range
    panel = pd.concat(parts, axis=1).reindex(idx)
    panel = panel.sort_index()

    # Missing values: short-gap interpolation + forward fill for macro
    panel = panel.astype(float)
    panel = panel.interpolate(method="time", limit=cfg.interpolate_limit)

    macro_cols = list(MACRO_FILE_TO_NAME.values())
    for c in macro_cols:
        if c in panel.columns:
            panel[c] = panel[c].ffill()

    # Warn if intersection is much smaller than target range
    intersection_size = len(available_months)
    target_size = len(target_range)
    if intersection_size < target_size * 0.8:
        import warnings
        warnings.warn(
            f"Only {intersection_size}/{target_size} months have data across all sources. "
            f"Using intersection: {min(available_months).date()} to {max(available_months).date()}",
            UserWarning,
        )

    return panel


def _rolling_features(s: pd.Series, windows: Iterable[int], prefix: str) -> pd.DataFrame:
    out = {}
    for w in windows:
        out[f"{prefix}_roll_mean_{w}"] = s.rolling(window=w, min_periods=max(2, w // 2)).mean()
        out[f"{prefix}_roll_std_{w}"] = s.rolling(window=w, min_periods=max(2, w // 2)).std()
    return pd.DataFrame(out, index=s.index)


def make_supervised(
    panel: pd.DataFrame,
    mineral: str,
    horizon: int,
    *,
    production_lags: Tuple[int, ...] = (1, 3, 6, 12),
    price_lags: Tuple[int, ...] = (1, 2, 3, 4, 5, 6),
    macro_lags: Tuple[int, ...] = (1, 2, 3),
    rolling_windows: Tuple[int, ...] = (3, 6, 12),
) -> Tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    """
    Build leakage-safe supervised data for direct forecasting:
    Features at time t predict y_{t+horizon}.
    Returns X, y, and the feature timestamp index (t).
    """
    if mineral not in MINERALS:
        raise ValueError(f"Unknown mineral: {mineral}. Expected one of {MINERALS}.")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    y_name = f"production_{mineral}"
    if y_name not in panel.columns:
        raise ValueError(f"Missing target column {y_name} in panel.")

    y0 = panel[y_name].copy()

    feats = {}
    # Level at forecast origin (y_t): used for naive baseline + directional metrics
    feats["y_level"] = y0
    # Production lags
    for lag in production_lags:
        feats[f"y_lag_{lag}"] = y0.shift(lag)

    # Rolling features on production (computed using past values only)
    feats_df = pd.DataFrame(feats, index=panel.index)
    feats_df = pd.concat([feats_df, _rolling_features(y0, rolling_windows, prefix="y")], axis=1)
    feats_df["y_pct_change_1"] = y0.pct_change(1)
    feats_df["y_pct_change_12"] = y0.pct_change(12)

    # Price features for this mineral (if available)
    price_col = f"price_{mineral}"
    if price_col in panel.columns:
        p = panel[price_col]
        for lag in price_lags:
            feats_df[f"price_lag_{lag}"] = p.shift(lag)
        feats_df = pd.concat([feats_df, _rolling_features(p, rolling_windows, prefix="price")], axis=1)
        feats_df["price_pct_change_1"] = p.pct_change(1)
        feats_df["price_pct_change_12"] = p.pct_change(12)

    # Macro features
    macro_cols = [c for c in MACRO_FILE_TO_NAME.values() if c in panel.columns]
    for c in macro_cols:
        s = panel[c]
        for lag in macro_lags:
            feats_df[f"{c}_lag_{lag}"] = s.shift(lag)

    # Month-of-year seasonality
    month = panel.index.month
    month_dummies = pd.get_dummies(month, prefix="month", drop_first=False)
    month_dummies.index = panel.index
    feats_df = pd.concat([feats_df, month_dummies], axis=1)

    # Target shift for direct horizon
    y = y0.shift(-horizon).rename(f"{y_name}_h{horizon}")

    # Drop rows with any missing features/target
    valid = feats_df.notna().all(axis=1) & y.notna()
    X = feats_df.loc[valid].copy()
    y_out = y.loc[valid].copy()
    idx = X.index
    return X, y_out, idx


def analyze_data_overlap(
    data_dir: Path,
    cfg: DataConfig | None = None,
    verbose: bool = True,
) -> Optional[Dict[str, Dict]]:
    """
    Load all data files and analyze overlapping months across datasets.
    
    Returns a dictionary with dataset information including:
    - 'data': DataFrame with loaded data
    - 'date_col': name of date column
    - 'months': set of available months (as Period objects)
    - 'filename': source filename
    
    Also returns the set of overlapping months across all datasets.
    """
    cfg = cfg or DataConfig()
    data_dir = Path(data_dir)
    
    datasets: Dict[str, Dict] = {}
    all_months: Optional[set] = None
    
    if verbose:
        print("Loading and analyzing data files...")
        print("=" * 60)
    
    # Load production data
    prod_path = data_dir / cfg.production_sales_file
    if prod_path.exists():
        if verbose:
            print(f"\nProcessing production_sales ({cfg.production_sales_file})...")
        try:
            df_prod = load_production_sales_xlsx(prod_path, cfg)
            # Extract months from production data
            months = set(df_prod.index.to_period('M'))
            if verbose:
                print(f"  Shape: {df_prod.shape}")
                print(f"  Date range: {df_prod.index.min()} to {df_prod.index.max()}")
                print(f"  Number of unique months: {len(months)}")
            
            datasets["production_sales"] = {
                'data': df_prod,
                'date_col': 'date',
                'months': months,
                'filename': cfg.production_sales_file
            }
            
            if all_months is None:
                all_months = months
            else:
                all_months = all_months.intersection(months)
        except Exception as e:
            if verbose:
                print(f"  Error loading production_sales: {str(e)}")
    
    # Load prices data
    prices_path = data_dir / cfg.prices_file
    if prices_path.exists():
        if verbose:
            print(f"\nProcessing prices ({cfg.prices_file})...")
        try:
            df_prices = load_prices_xls(prices_path)
            months = set(df_prices.index.to_period('M'))
            if verbose:
                print(f"  Shape: {df_prices.shape}")
                print(f"  Date range: {df_prices.index.min()} to {df_prices.index.max()}")
                print(f"  Number of unique months: {len(months)}")
            
            datasets["prices"] = {
                'data': df_prices,
                'date_col': 'date',
                'months': months,
                'filename': cfg.prices_file
            }
            
            if all_months is None:
                all_months = months
            else:
                all_months = all_months.intersection(months)
        except Exception as e:
            if verbose:
                print(f"  Error loading prices: {str(e)}")
    
    # Load macro data files
    for fn, series_name in MACRO_FILE_TO_NAME.items():
        macro_path = data_dir / fn
        if macro_path.exists():
            if verbose:
                print(f"\nProcessing {series_name} ({fn})...")
            try:
                df_macro = load_fred_monthly_xlsx(macro_path, series_name=series_name)
                months = set(df_macro.index.to_period('M'))
                if verbose:
                    print(f"  Shape: {df_macro.shape}")
                    print(f"  Date range: {df_macro.index.min()} to {df_macro.index.max()}")
                    print(f"  Number of unique months: {len(months)}")
                
                datasets[series_name] = {
                    'data': df_macro,
                    'date_col': 'date',
                    'months': months,
                    'filename': fn
                }
                
                if all_months is None:
                    all_months = months
                else:
                    all_months = all_months.intersection(months)
            except Exception as e:
                if verbose:
                    print(f"  Error loading {series_name}: {str(e)}")
    
    if verbose:
        print("\n" + "=" * 60)
    
    if all_months is None or len(all_months) == 0:
        if verbose:
            print("No overlapping months found!")
        return None
    
    if verbose:
        print(f"\nOverlapping months across all datasets: {len(all_months)}")
        sorted_months = sorted(all_months)
        print(f"Date range: {sorted_months[0]} to {sorted_months[-1]}")
        if len(sorted_months) <= 20:
            print(f"All months: {sorted_months}")
        else:
            print(f"First 10 months: {sorted_months[:10]}")
            print(f"Last 10 months: {sorted_months[-10:]}")
    
    # Add overlapping months to return dict
    result = {
        'datasets': datasets,
        'overlapping_months': all_months
    }
    
    return result


def filter_and_save_processed_data(
    analysis_result: Dict[str, Dict],
    output_dir: Path,
    cfg: DataConfig | None = None,
    verbose: bool = True,
) -> None:
    """
    Filter datasets to overlapping months and save to processed data directory.
    
    Args:
        analysis_result: Result from analyze_data_overlap()
        output_dir: Directory to save processed data files
        cfg: DataConfig instance
        verbose: Whether to print progress information
    """
    cfg = cfg or DataConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = analysis_result['datasets']
    overlapping_months = analysis_result['overlapping_months']
    
    if verbose:
        print("\n" + "=" * 60)
        print("Filtering and saving data...")
        print("=" * 60)
    
    for name, dataset_info in datasets.items():
        df = dataset_info['data'].copy()
        date_col = dataset_info['date_col']
        filename = dataset_info['filename']
        
        # Convert index to period for filtering
        df_periods = df.index.to_period('M')
        
        # Filter to overlapping months
        mask = df_periods.isin(overlapping_months)
        df_filtered = df[mask].copy()
        
        # For production data, it's already filtered to target minerals
        # For prices data, it's already filtered to mineral price columns
        # For macro data, no additional filtering needed
        
        # Reset index to have date as a column for saving
        # The index is a DatetimeIndex, so reset_index() will create a 'date' column
        df_filtered = df_filtered.reset_index()
        
        # Ensure the date column is named correctly (in case index had a different name)
        if date_col not in df_filtered.columns and len(df_filtered.columns) > 0:
            # Rename the first column (which should be the index) to the expected date column name
            first_col = df_filtered.columns[0]
            df_filtered.rename(columns={first_col: date_col}, inplace=True)
        
        if verbose:
            print(f"\n{name}:")
            if name == "production_sales":
                # Count minerals (columns that start with 'production_')
                prod_cols = [c for c in df_filtered.columns if c.startswith('production_')]
                print(f"  Production columns: {prod_cols}")
            elif name == "prices":
                price_cols = [c for c in df_filtered.columns if c.startswith('price_')]
                print(f"  Price columns: {price_cols}")
            print(f"  Original rows: {len(df)}")
            print(f"  Filtered rows: {len(df_filtered)}")
            if len(df_filtered) > 0 and date_col in df_filtered.columns:
                print(f"  Date range: {df_filtered[date_col].min()} to {df_filtered[date_col].max()}")
        
        # Save filtered data
        output_path = output_dir / filename
        
        # Convert .xls to .xlsx for consistency
        if output_path.suffix == '.xls':
            output_path = output_path.with_suffix('.xlsx')
        
        df_filtered.to_excel(output_path, index=False)
        
        if verbose:
            print(f"  Saved to: {output_path}")
    
    if verbose:
        print("\n" + "=" * 60)
        print("Processing complete!")
        print(f"Filtered data saved to: {output_dir}")

