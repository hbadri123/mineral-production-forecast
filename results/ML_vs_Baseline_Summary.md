# ML vs Baseline Comparison Summary

## Overview

This summary highlights the performance of **Machine Learning models** compared to **baseline methods** (naive and historical mean forecasts).

**Key Metric**: Skill Score = 1 - RMSE_model / RMSE_naive
- **Positive skill score** = ML model outperforms naive baseline
- **Negative skill score** = ML model underperforms naive baseline

---

## Results by Mineral and Horizon

### Coal

| Horizon | Best ML Model | Skill Score | Improvement % | ML Wins | Total ML Models |
|---------|---------------|-------------|----------------|---------|-----------------|
| h=3 | catboost | **0.15** ✓ | 15.16% | 3/3 | 3 |
| h=6 | lightgbm | **0.33** ✓ | 33.10% | 3/3 | 3 |
| h=12 | catboost | **0.17** ✓ | 17.46% | 3/3 | 3 |

### Copper

| Horizon | Best ML Model | Skill Score | Improvement % | ML Wins | Total ML Models |
|---------|---------------|-------------|----------------|---------|-----------------|
| h=3 | random_forest | **0.41** ✓ | 40.93% | 3/3 | 3 |
| h=6 | random_forest | **0.06** ✓ | 5.68% | 2/3 | 3 |
| h=12 | random_forest | **0.29** ✓ | 28.99% | 3/3 | 3 |

### Gold

| Horizon | Best ML Model | Skill Score | Improvement % | ML Wins | Total ML Models |
|---------|---------------|-------------|----------------|---------|-----------------|
| h=3 | random_forest | -4.56 ✗ | -456.08% | 0/3 | 3 |
| h=6 | catboost | -2.74 ✗ | -273.67% | 0/3 | 3 |
| h=12 | catboost | -1.15 ✗ | -115.25% | 0/3 | 3 |

### Iron ore

| Horizon | Best ML Model | Skill Score | Improvement % | ML Wins | Total ML Models |
|---------|---------------|-------------|----------------|---------|-----------------|
| h=3 | lightgbm | **0.07** ✓ | 7.48% | 2/3 | 3 |
| h=6 | lightgbm | -0.19 ✗ | -19.33% | 0/3 | 3 |
| h=12 | random_forest | -0.14 ✗ | -13.95% | 0/3 | 3 |

---

## Overall Statistics

- **Total cases analyzed**: 12
- **ML beats baseline**: 7/12 cases (58.3%)
- **Average improvement when ML wins**: 21.26%
- **Average ML skill score**: -0.72

## Key Findings

✅ **ML models outperform baseline in 7 out of 12 cases** (58.3%)

When ML models beat the baseline, they achieve an average improvement of 21.26% in RMSE compared to the naive forecast.

⚠️ **Baseline outperforms ML in 5 out of 12 cases** (41.7%)

This highlights the importance of model selection and the challenge of forecasting in certain mineral/horizon combinations.

---

*Generated automatically from model evaluation results.*