from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from src.data_loader import DataConfig, MINERALS, build_monthly_panel, load_raw_sources, make_supervised
from src.evaluation import (
    compute_metrics,
    generate_ml_vs_baseline_summary,
    plot_forecast_vs_actual_all_models,
    plot_residuals_all_models,
    single_split_evaluation,
    SplitConfig,
    save_feature_importance,
)
from src.models import check_required_ml_dependencies, get_model_specs


def ensure_structure(project_root: Path) -> None:
    (project_root / "results" / "tables").mkdir(parents=True, exist_ok=True)
    (project_root / "results" / "plots").mkdir(parents=True, exist_ok=True)
    (project_root / "results" / "feature_importance").mkdir(parents=True, exist_ok=True)
    (project_root / "notebooks").mkdir(parents=True, exist_ok=True)
    (project_root / "data" / "raw").mkdir(parents=True, exist_ok=True)


def round_numeric_columns(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    """
    Round all numeric columns in a DataFrame to specified decimal places.
    """
    df_rounded = df.copy()
    numeric_cols = df_rounded.select_dtypes(include=[np.number]).columns
    df_rounded[numeric_cols] = df_rounded[numeric_cols].round(decimals)
    return df_rounded


def generate_ml_summary_markdown(ml_vs_baseline_df: pd.DataFrame, output_path: Path) -> None:
    """
    Generate a markdown summary highlighting ML vs baseline performance.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if ml_vs_baseline_df.empty:
        content = "# ML vs Baseline Comparison Summary\n\nNo data available.\n"
        output_path.write_text(content)
        return
    
    lines = [
        "# ML vs Baseline Comparison Summary",
        "",
        "## Overview",
        "",
        "This summary highlights the performance of **Machine Learning models** compared to **baseline methods** (naive and historical mean forecasts).",
        "",
        "**Key Metric**: Skill Score = 1 - RMSE_model / RMSE_naive",
        "- **Positive skill score** = ML model outperforms naive baseline",
        "- **Negative skill score** = ML model underperforms naive baseline",
        "",
        "---",
        "",
        "## Results by Mineral and Horizon",
        "",
    ]
    
    # Group by mineral
    for mineral in sorted(ml_vs_baseline_df["mineral"].unique()):
        mineral_df = ml_vs_baseline_df[ml_vs_baseline_df["mineral"] == mineral].sort_values("horizon")
        
        lines.append(f"### {mineral}")
        lines.append("")
        lines.append("| Horizon | Best ML Model | Skill Score | Improvement % | ML Wins | Total ML Models |")
        lines.append("|---------|---------------|-------------|----------------|---------|-----------------|")
        
        for _, row in mineral_df.iterrows():
            skill = row["best_ml_skill_score"]
            improvement = row["improvement_pct"]
            wins = int(row["ml_beats_baseline_count"])
            total = int(row["total_ml_models"])
            
            # Format skill score with color indicator
            skill_str = f"{skill:.2f}"
            if skill > 0:
                skill_str = f"**{skill_str}** ✓"
            else:
                skill_str = f"{skill_str} ✗"
            
            improvement_str = f"{improvement:.2f}%" if improvement > 0 else f"{improvement:.2f}%"
            
            lines.append(
                f"| h={int(row['horizon'])} | {row['best_ml_model']} | {skill_str} | {improvement_str} | {wins}/{total} | {total} |"
            )
        
        lines.append("")
    
    # Overall statistics
    lines.extend([
        "---",
        "",
        "## Overall Statistics",
        "",
    ])
    
    total_cases = len(ml_vs_baseline_df)
    ml_wins = (ml_vs_baseline_df["best_ml_skill_score"] > 0).sum()
    win_rate = 100 * ml_wins / total_cases if total_cases > 0 else 0
    
    ml_wins_df = ml_vs_baseline_df[ml_vs_baseline_df["best_ml_skill_score"] > 0]
    avg_improvement = ml_wins_df["improvement_pct"].mean() if not ml_wins_df.empty else 0
    avg_skill = ml_vs_baseline_df["avg_ml_skill_score"].mean()
    
    lines.extend([
        f"- **Total cases analyzed**: {total_cases}",
        f"- **ML beats baseline**: {ml_wins}/{total_cases} cases ({win_rate:.1f}%)",
        f"- **Average improvement when ML wins**: {avg_improvement:.2f}%",
        f"- **Average ML skill score**: {avg_skill:.2f}",
        "",
        "## Key Findings",
        "",
    ])
    
    if ml_wins > 0:
        lines.extend([
            f"✅ **ML models outperform baseline in {ml_wins} out of {total_cases} cases** ({win_rate:.1f}%)",
            "",
            "When ML models beat the baseline, they achieve an average improvement of "
            f"{avg_improvement:.2f}% in RMSE compared to the naive forecast.",
            "",
        ])
    
    if ml_wins < total_cases:
        cases_where_baseline_wins = total_cases - ml_wins
        lines.extend([
            f"⚠️ **Baseline outperforms ML in {cases_where_baseline_wins} out of {total_cases} cases** "
            f"({100*cases_where_baseline_wins/total_cases:.1f}%)",
            "",
            "This highlights the importance of model selection and the challenge of forecasting "
            "in certain mineral/horizon combinations.",
            "",
        ])
    
    lines.extend([
        "---",
        "",
        "*Generated automatically from model evaluation results.*",
    ])
    
    content = "\n".join(lines)
    output_path.write_text(content)


def main() -> int:
    warnings.filterwarnings("ignore")

    project_root = Path(__file__).resolve().parent
    ensure_structure(project_root)

    # Hard dependency policy (per project requirements)
    try:
        check_required_ml_dependencies()
    except ImportError as e:
        # Fail fast but gracefully (no stack trace) with an actionable message.
        print(str(e))
        return 1

    cfg = DataConfig()
    data_dir = project_root / "data" / "raw"

    required_files = [
        cfg.production_sales_file,
        cfg.prices_file,
        "cpi.xlsx",
        "EXSFUS.xlsx",
        "industrial_production_index.xlsx",
    ]
    missing = [f for f in required_files if not (data_dir / f).exists()]
    if missing:
        print(f"Missing raw data files under {data_dir}: {missing}")
        return 2

    print("Loading data...")
    sources = load_raw_sources(data_dir, cfg=cfg)
    panel = build_monthly_panel(sources, cfg=cfg)
    print(f"Panel built: shape={panel.shape}, range={panel.index.min().date()}..{panel.index.max().date()}")

    horizons = [3, 6, 12]
    model_factories = get_model_specs(random_state=42)

    all_metrics = []
    all_predictions = []
    all_errors = []

    for mineral in MINERALS:
        for h in horizons:
            print("\n" + "=" * 80)
            print(f"Mineral: {mineral} | Horizon: {h} months")
            print("=" * 80)

            X, y, idx = make_supervised(panel, mineral=mineral, horizon=h)
            print(f"Supervised dataset: n={len(X)}, p={X.shape[1]}")

            bt = single_split_evaluation(
                X,
                y,
                model_factories=model_factories,
                cfg=SplitConfig(verbose=True),
                do_tune_hyperparameters=True,
            )

            metrics = compute_metrics(bt, naive_model_name="naive")
            metrics.insert(0, "horizon", h)
            metrics.insert(0, "mineral", mineral)
            all_metrics.append(metrics)

            tables_dir = project_root / "results" / "tables"

            # Plots for top models (by RMSE), plus naive
            top_models = [m for m in metrics["model"].head(3).tolist() if m in bt.y_pred.columns]
            if "naive" not in top_models:
                top_models.append("naive")

            plots_dir = project_root / "results" / "plots" / mineral.replace(" ", "_") / f"h{h}"
            
            # Combined forecast vs actual plot for all models
            plot_forecast_vs_actual_all_models(
                bt,
                models=top_models,
                title=f"{mineral} | h={h} | Forecast vs Actual (All Models)",
                output_path=plots_dir / "forecast_vs_actual_all_models.png",
                show_naive=True,
            )
            
            # Combined residuals plot for all models
            plot_residuals_all_models(
                bt,
                models=top_models,
                title=f"{mineral} | h={h} | Residuals (All Models)",
                output_path=plots_dir / "residuals_all_models.png",
            )

            # Collect predictions with mineral and horizon identifiers
            pred_df = pd.concat([bt.y_true, bt.y_level, bt.y_pred], axis=1)
            pred_df = pred_df.reset_index()  # Convert index to column
            pred_df.insert(0, "horizon", h)
            pred_df.insert(0, "mineral", mineral)
            all_predictions.append(pred_df)

            # Collect errors (if any) with mineral and horizon identifiers
            err_nonempty = (bt.errors.replace("", np.nan).notna()).any().any()
            if err_nonempty:
                err_df = bt.errors.reset_index()  # Convert index to column
                err_df.insert(0, "horizon", h)
                err_df.insert(0, "mineral", mineral)
                all_errors.append(err_df)

            # Save feature importance for tree-based models
            # Train a final model on all available data to get feature importances
            tree_models = ["random_forest", "xgboost", "lightgbm", "catboost"]
            fi_dir = project_root / "results" / "feature_importance" / mineral.replace(" ", "_") / f"h{h}"
            for model_name in tree_models:
                if model_name in model_factories:
                    try:
                        # Train on all data to get representative feature importances
                        final_model = model_factories[model_name]()
                        final_model.fit(X, y)
                        fi_path = fi_dir / f"{model_name}_importance"
                        save_feature_importance(
                            final_model,
                            feature_names=list(X.columns),
                            output_path=fi_path,
                            top_k=30,
                        )
                    except Exception as e:
                        # Skip if model doesn't support feature importance or fails
                        pass

    all_metrics_df = pd.concat(all_metrics, ignore_index=True)
    out_all = project_root / "results" / "tables" / "metrics_by_mineral_horizon.csv"
    round_numeric_columns(all_metrics_df).to_csv(out_all, index=False)

    # Save combined predictions
    if all_predictions:
        all_predictions_df = pd.concat(all_predictions, ignore_index=True)
        out_predictions = project_root / "results" / "tables" / "predictions_all.csv"
        all_predictions_df.to_csv(out_predictions, index=False)

    # Save combined errors
    if all_errors:
        all_errors_df = pd.concat(all_errors, ignore_index=True)
        out_errors = project_root / "results" / "tables" / "errors_all.csv"
        all_errors_df.to_csv(out_errors, index=False)

    # Simple summary: best model per mineral/horizon by RMSE
    summary = (
        all_metrics_df.sort_values(["mineral", "horizon", "rmse"])
        .groupby(["mineral", "horizon"], as_index=False)
        .first()
        .loc[:, ["mineral", "horizon", "model", "rmse", "mae", "smape", "directional_accuracy", "skill_rmse_vs_naive", "n"]]
    )
    out_summary = project_root / "results" / "tables" / "metrics_summary.csv"
    round_numeric_columns(summary).to_csv(out_summary, index=False)

    # ML vs Baseline comparison summary
    ml_vs_baseline = generate_ml_vs_baseline_summary(all_metrics_df)
    out_ml_comparison = project_root / "results" / "tables" / "ml_vs_baseline_comparison.csv"
    round_numeric_columns(ml_vs_baseline).to_csv(out_ml_comparison, index=False)

    # Generate markdown summary highlighting ML performance
    summary_md_path = project_root / "results" / "ML_vs_Baseline_Summary.md"
    generate_ml_summary_markdown(ml_vs_baseline, summary_md_path)

    print("\n" + "=" * 80)
    print("Summary (best RMSE per mineral/horizon):")
    print(round_numeric_columns(summary).to_string(index=False))
    print("\n" + "=" * 80)
    print("\nML vs Baseline Comparison:")
    print("=" * 80)
    if not ml_vs_baseline.empty:
        display_cols = ["mineral", "horizon", "best_ml_model", "best_ml_skill_score", 
                       "ml_beats_baseline_count", "total_ml_models", "improvement_pct"]
        print(round_numeric_columns(ml_vs_baseline[display_cols]).to_string(index=False))
        
        # Overall statistics
        total_cases = len(ml_vs_baseline)
        ml_wins = (ml_vs_baseline["best_ml_skill_score"] > 0).sum()
        avg_improvement = ml_vs_baseline[ml_vs_baseline["improvement_pct"] > 0]["improvement_pct"].mean()
        
        print(f"\nOverall ML Performance:")
        print(f"  - ML beats baseline in {ml_wins}/{total_cases} cases ({100*ml_wins/total_cases:.1f}%)")
        print(f"  - Average improvement when ML wins: {avg_improvement:.2f}%")
        print(f"  - Average ML skill score: {ml_vs_baseline['avg_ml_skill_score'].mean():.2f}")
    else:
        print("No ML vs baseline comparison data available.")
    print("\nSaved:")
    print(f"  - {out_all}")
    print(f"  - {out_summary}")
    print(f"  - {out_ml_comparison}")
    if all_predictions:
        print(f"  - {out_predictions}")
    if all_errors:
        print(f"  - {out_errors}")
    print(f"  - {summary_md_path}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
