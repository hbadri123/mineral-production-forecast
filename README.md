## Forecasting Monthly Mineral Production in South Africa (1990–2002)

This repository implements a **reproducible, regression-based comparison study** for forecasting monthly mineral production in South Africa over **3-, 6-, and 12-month horizons**, using rolling-origin evaluation (no shuffling, no leakage).

### Research Question

**Does machine learning outperform baseline forecasting methods?**

This project compares ML models (Random Forest, XGBoost, LightGBM, CatBoost) against baseline methods (naive, historical mean) and classical time-series models (ARIMA, SARIMA, SARIMAX) to determine **whether** ML approaches outperform baseline methods for mineral production forecasting. **The primary goal is to investigate whether modern ML approaches can outperform simple baseline methods across different minerals and forecast horizons, and to identify conditions where ML is most effective.** Model performance is evaluated using skill scores relative to the naive baseline, where positive skill scores indicate superior performance.

### Repository structure

```
.
├── README.md
├── PROPOSAL.md
├── AI_USAGE.md
├── environment.yml
├── main.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── models.py
│   └── evaluation.py
├── data/
│   └── raw/
├── results/
└── notebooks/
```

### Setup

Install dependencies using conda:

```bash
conda env create -f environment.yml
conda activate hajer-ds-project
```

### Data

Place the following files under `data/raw/`:
- `production_sales.xlsx`
- `prices.xls`
- `cpi.xlsx`
- `EXSFUS.xlsx`
- `industrial_production_index.xlsx`

### Run

```bash
python main.py
```

### Outputs

`python main.py` will write artifacts to `results/`, including:
- **Tables** (`results/tables/`)
  - `metrics_by_mineral_horizon.csv` - All model metrics for all minerals/horizons
  - `metrics_summary.csv` - Best model per mineral/horizon (by RMSE)
  - **`ml_vs_baseline_comparison.csv`** - **ML vs baseline comparison summary** ⭐
  - per-mineral/horizon metric tables and prediction CSVs
- **Summary** (`results/`)
  - **`ML_vs_Baseline_Summary.md`** - **Detailed markdown summary highlighting ML performance vs baseline** ⭐
- **Plots** (`results/plots/<mineral>/h<horizon>/`)
  - forecast vs actual plots
  - residual plots

### Notes on reproducibility

- Random seeds are fixed (`random_state=42` where applicable).
- All evaluation is time-ordered (rolling-origin backtesting).
- No future information is used in features.
