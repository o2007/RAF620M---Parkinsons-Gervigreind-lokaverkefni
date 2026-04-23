"""
Samanburðarkóði: ber saman öll líkön
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from init import (
    LONGITUDINAL_FEATURE_COLS,
    MODEL_COLORS,
    RF_PARAMS,
    SVR_PARAMS,
    XGB_PARAMS,
    add_longitudinal_features,
    build_linear_pipeline,
    build_longitudinal_pairs,
    build_rf_pipeline,
    build_svr_pipeline,
    build_xgb_pipeline,
    evaluate_delta_pipeline,
    get_output_dir,
    load_source_tables,
    prepare_clinical_table,
    save_plot,
    split_by_patient_time,
)

warnings.filterwarnings("ignore")
OUTPUT_DIR = get_output_dir("comparison")


def plot_bar_chart(comparison_df):
    metric_map = [("mae", "MAE"), ("rmse", "RMSE"), ("r2", "R2")]
    labels = comparison_df["model"].tolist()
    colors = [MODEL_COLORS.get(label, "#888888") for label in labels]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (metric_key, title) in zip(axes, metric_map):
        values = comparison_df[metric_key].tolist()
        bars = ax.bar(labels, values, color=colors, width=0.6, edgecolor="white")
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02 * abs(bar.get_height()) + 0.05,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, alpha=0.3, axis="y")
        if metric_key == "r2":
            ax.set_ylim(min(0, min(values)) - 0.05, 1.05)
        else:
            ax.set_ylim(0, max(values) * 1.3)

    fig.suptitle("Samanburður líkana á prófunargögnum", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_plot("comparison_bar_chart.png", OUTPUT_DIR)


def plot_error_cdf(results, y_test):
    fig, ax = plt.subplots(figsize=(10, 6))
    for entry in results:
        abs_errors = np.abs(entry["preds"] - y_test.to_numpy())
        sorted_errors = np.sort(abs_errors)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        ax.plot(
            sorted_errors,
            cdf * 100,
            label=f"{entry['model']} (MAE={entry['mae']:.2f})",
            color=MODEL_COLORS.get(entry["model"], "#888888"),
            linewidth=2,
        )

    ax.axvline(5, linestyle="--", color="orange", linewidth=1.2, label="5 stig")
    ax.axvline(10, linestyle="--", color="red", linewidth=1.2, label="10 stig")
    ax.set_xlabel("MAE (UPDRS-III stig)", fontsize=12)
    ax.set_ylabel("Samtalas % spáa", fontsize=12)
    ax.set_title("Uppsöfnuð dreifing skekkju - öll líkön", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot("comparison_error_cdf.png", OUTPUT_DIR)


def plot_scatter_grid(results, y_test):
    cols = 3
    rows = (len(results) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.array(axes).flatten()

    for ax, entry in zip(axes, results):
        preds = entry["preds"]
        ax.scatter(y_test, preds, alpha=0.3, s=14, color=MODEL_COLORS.get(entry["model"], "#888888"))
        limits = [min(float(y_test.min()), float(preds.min())) - 1, max(float(y_test.max()), float(preds.max())) + 1]
        ax.plot(limits, limits, "r--", linewidth=1.2)
        ax.set_title(f"{entry['model']}\nMAE={entry['mae']:.2f}  R2={entry['r2']:.3f}", fontsize=10)
        ax.set_xlabel("Raunverulegt UPDRS-III", fontsize=9)
        ax.set_ylabel("Spáð UPDRS-III", fontsize=9)
        ax.grid(True, alpha=0.3)

    for ax in axes[len(results) :]:
        fig.delaxes(ax)

    fig.suptitle("Spáð vs raunverulegt - öll líkön", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_plot("comparison_scatter.png", OUTPUT_DIR)


def plot_error_boxplot(results, y_test):
    data = [np.abs(entry["preds"] - y_test.to_numpy()) for entry in results]
    labels = [entry["model"] for entry in results]

    fig, ax = plt.subplots(figsize=(11, 6))
    bp = ax.boxplot(data, patch_artist=True, medianprops=dict(color="black", linewidth=2))
    for patch, label in zip(bp["boxes"], labels):
        patch.set_facecolor(MODEL_COLORS.get(label, "#888888"))
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=15, fontsize=11)
    ax.set_ylabel("Töluleg skekkja (UPDRS-III stig)", fontsize=12)
    ax.set_title("Dreifing skekkju eftir líkönum", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    save_plot("comparison_error_boxplot.png", OUTPUT_DIR)


def plot_bias_chart(results, y_test):
    labels = [entry["model"] for entry in results]
    biases = [float(np.mean(entry["preds"] - y_test.to_numpy())) for entry in results]
    colors = [MODEL_COLORS.get(label, "#888888") for label in labels]

    _, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.9)))
    bars = ax.barh(labels, biases, color=colors, edgecolor="white", height=0.55)
    ax.axvline(0, color="black", linewidth=1.2)

    for bar, value in zip(bars, biases):
        x_pos = value + (0.05 if value >= 0 else -0.05)
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f"{value:+.2f}", va="center", ha="left" if value >= 0 else "right", fontsize=10, fontweight="bold")

    ax.set_xlabel("Meðalskekkja: spáð - raunverulegt (UPDRS-III stig)", fontsize=11)
    ax.set_title("Hlutdrægni líkana\n+ = ofspár að meðaltali  |  - = vanspár að meðaltali", fontsize=12)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    save_plot("comparison_bias.png", OUTPUT_DIR)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 72)
    print("  PARKINSON - SAMANBURÐUR LÍKANA")
    print("  Grunnlíkan | Linear Regression | Random Forest | SVM | XGBoost")
    print("=" * 72)

    source_tables = load_source_tables()
    clinical_table = prepare_clinical_table(source_tables)
    pairs_df = add_longitudinal_features(build_longitudinal_pairs(clinical_table))

    X = pairs_df[LONGITUDINAL_FEATURE_COLS].copy()
    y = pairs_df["target_updrs3"].copy()
    groups = pairs_df["PATNO"].copy()

    train_patients, test_patients = split_by_patient_time(pairs_df)
    train_mask = groups.isin(train_patients)
    test_mask = groups.isin(test_patients)

    X_train = X[train_mask].reset_index(drop=True)
    y_train = y[train_mask].reset_index(drop=True)
    X_test = X[test_mask].reset_index(drop=True)
    y_test = y[test_mask].reset_index(drop=True)
    pairs_train = pairs_df[train_mask].copy().reset_index(drop=True)
    pairs_test = pairs_df[test_mask].copy().reset_index(drop=True)

    last_scores_train = pairs_train["traj_last_score"].to_numpy()
    last_scores_test = pairs_test["traj_last_score"].to_numpy()
    y_delta_train = y_train - last_scores_train

    print(f"\nTrain rows: {len(X_train):,} | patients: {len(train_patients):,}")
    print(f"Test rows:  {len(X_test):,}  | patients: {len(test_patients):,}")

    baseline_preds = last_scores_test
    results = [
        {
            "model": "Grunnlíkan",
            "mae": float(mean_absolute_error(y_test, baseline_preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, baseline_preds))),
            "r2": float(r2_score(y_test, baseline_preds)),
            "delta_mae": float(mean_absolute_error(y_test - last_scores_test, np.zeros(len(y_test)))),
            "mean_error": float(np.mean(baseline_preds - y_test.to_numpy())),
            "preds": baseline_preds,
        }
    ]

    model_specs = [
        ("Linear Regression", build_linear_pipeline, {}),
        ("Random Forest", build_rf_pipeline, RF_PARAMS),
        ("SVM (SVR)", build_svr_pipeline, SVR_PARAMS),
        ("XGBoost", build_xgb_pipeline, XGB_PARAMS),
    ]

    for model_name, builder, params in model_specs:
        print(f"\n{'-' * 60}\n  {model_name}\n{'-' * 60}")
        pipeline = builder().set_params(**{f"model__{key}": value for key, value in params.items()})
        result = evaluate_delta_pipeline(model_name, pipeline, X_train, y_delta_train, X_test, y_test, last_scores_test)
        result["mean_error"] = float(np.mean(result["preds"] - y_test.to_numpy()))
        result["params"] = params
        results.append(result)
        print(f"  MAE={result['mae']:.2f}  RMSE={result['rmse']:.2f}  R2={result['r2']:.3f}  delta-MAE={result['delta_mae']:.2f}")

    comparison_df = (
        pd.DataFrame([{key: value for key, value in result.items() if key not in ("preds", "pipeline", "delta_preds", "params")} for result in results])
        .sort_values("mae")
        .reset_index(drop=True)
    )
    comparison_df.to_csv(os.path.join(OUTPUT_DIR, "comparison_all_models.csv"), index=False)

    print("\n" + "=" * 72)
    print("Loka samanburður - prófunargögn")
    print("=" * 72)
    print(comparison_df.to_string(index=False))
    print("\nVistað: comparison_all_models.csv")

    print("\n% spáa innan við marka:")
    header = f"{'Model':<22}  {'<=5 stig':>8}  {'<=10 stig':>10}"
    print(header)
    print("-" * len(header))
    for entry in results:
        abs_errors = np.abs(entry["preds"] - y_test.to_numpy())
        print(f"  {entry['model']:<20}  {100 * np.mean(abs_errors <= 5):7.1f}%  {100 * np.mean(abs_errors <= 10):8.1f}%")

    baseline_mae = results[0]["mae"]
    print("\nBæting MAE yfir grunnlíkani:")
    for entry in results[1:]:
        improvement = 100 * (baseline_mae - entry["mae"]) / baseline_mae
        direction = "betra" if improvement > 0 else "verra"
        print(f"  {entry['model']:<22}  {abs(improvement):.1f}% {direction}")

    print("\nTeikna gröf...")
    plot_bar_chart(comparison_df)
    plot_bias_chart(results, y_test)
    plot_error_cdf(results, y_test)
    plot_scatter_grid(results, y_test)
    plot_error_boxplot(results, y_test)

    print("\nAllt vistað í:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
