"""
Random Forest líkan fyrir næstu UPDRS-III heimsókn.
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from init import (
    FEATURE_COLS,
    RF_PARAMS,
    add_longitudinal_features,
    build_feature_importance_df,
    build_longitudinal_pairs,
    build_rf_pipeline,
    evaluate_delta_pipeline,
    get_output_dir,
    load_source_tables,
    prepare_clinical_table,
    save_plot,
    split_by_patient_time,
)

warnings.filterwarnings("ignore")
OUTPUT_DIR = get_output_dir("random_forest")

PLOT_COLORS = {"actual": "steelblue", "predicted": "#6acc64"}


def plot_predicted_vs_actual(y_test, preds, mae, rmse, r2):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, preds, alpha=0.35, s=18, color=PLOT_COLORS["actual"])
    limits = [min(float(y_test.min()), float(preds.min())) - 1, max(float(y_test.max()), float(preds.max())) + 1]
    ax.plot(limits, limits, "r--", linewidth=1.5, label="Fullkomin spá")
    ax.set_xlabel("Raunverulegt UPDRS-III (næsta heimsókn)", fontsize=12)
    ax.set_ylabel("Spáð UPDRS-III", fontsize=12)
    ax.set_title(f"Random Forest: spáð vs raunverulegt\nMAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.3f}", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot("rf_predicted_vs_actual.png", OUTPUT_DIR)


def plot_feature_importance(importance_df):
    sorted_df = importance_df.sort_values("importance")
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(sorted_df["feature"], sorted_df["importance"], color=PLOT_COLORS["predicted"], edgecolor="white")
    for bar, value in zip(bars, sorted_df["importance"]):
        ax.text(value + 0.001, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Meðaltalsminnkun á óhreinu (MDI)", fontsize=12)
    ax.set_title("Random Forest: Vægi einkenna", fontsize=13)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    save_plot("rf_feature_importance.png", OUTPUT_DIR)


def plot_residuals(residuals):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=45, edgecolor="white", color=PLOT_COLORS["actual"])
    ax.axvline(0, linestyle="--", color="red", linewidth=2, label="Engin skekkja")
    ax.axvline(float(np.mean(residuals)), linestyle="--", color="orange", linewidth=1.5, label=f"Meðaltal = {float(np.mean(residuals)):.2f}")
    ax.set_xlabel("Skekkja (spáð - raunverulegt)", fontsize=12)
    ax.set_ylabel("Fjöldi", fontsize=12)
    ax.set_title("Skekkjudreifing - Random Forest", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot("rf_residuals.png", OUTPUT_DIR)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 72)
    print("  PARKINSON - RANDOM FOREST")
    print("  Markbreyta: UPDRS-III í næstu heimsókn")
    print("=" * 72)

    pairs_df = add_longitudinal_features(build_longitudinal_pairs(prepare_clinical_table(load_source_tables())))
    X = pairs_df[FEATURE_COLS].copy()
    y = pairs_df["target_updrs3"].copy()
    groups = pairs_df["PATNO"].copy()

    train_patients, test_patients = split_by_patient_time(pairs_df)
    train_mask = groups.isin(train_patients)
    test_mask = groups.isin(test_patients)

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    pairs_train = pairs_df[train_mask].copy().reset_index(drop=True)
    pairs_test = pairs_df[test_mask].copy().reset_index(drop=True)

    print(f"\nTrain rows: {len(X_train):,} | patients: {len(train_patients):,}")
    print(f"Test rows:  {len(X_test):,}  | patients: {len(test_patients):,}")

    baseline_preds = pairs_test["traj_last_score"].to_numpy()
    baseline_mae = float(mean_absolute_error(y_test, baseline_preds))
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_preds)))
    baseline_r2 = float(r2_score(y_test, baseline_preds))
    print("\nGrunnlíkan: síðasta mælda skor")
    print(f"  MAE:  {baseline_mae:.2f}")
    print(f"  RMSE: {baseline_rmse:.2f}")
    print(f"  R2:   {baseline_r2:.3f}")

    y_train_delta = y_train - pairs_train["traj_last_score"].to_numpy()
    print(f"\nStillingar líkans: {RF_PARAMS}")

    result = evaluate_delta_pipeline(
        "Random Forest",
        build_rf_pipeline().set_params(**{f"model__{key}": value for key, value in RF_PARAMS.items()}),
        X_train,
        y_train_delta,
        X_test,
        y_test,
        pairs_test["traj_last_score"].to_numpy(),
    )

    prediction_df = pairs_test.copy()
    prediction_df["actual"] = y_test.to_numpy()
    prediction_df["predicted"] = result["preds"]
    prediction_df["residual"] = result["preds"] - y_test.to_numpy()
    prediction_df["abs_error"] = np.abs(prediction_df["residual"])
    prediction_df.to_csv(os.path.join(OUTPUT_DIR, "rf_predictions.csv"), index=False)

    importance_df = build_feature_importance_df(result["pipeline"], FEATURE_COLS)
    importance_df.to_csv(os.path.join(OUTPUT_DIR, "rf_feature_importance.csv"), index=False)

    print("\nNiðurstöður Random Forest:")
    print(f"  MAE:        {result['mae']:.2f}  (baseline: {baseline_mae:.2f})")
    print(f"  RMSE:       {result['rmse']:.2f}  (baseline: {baseline_rmse:.2f})")
    print(f"  R2:         {result['r2']:.3f}  (baseline: {baseline_r2:.3f})")
    print(f"  Delta MAE:  {result['delta_mae']:.2f}")
    print("\n10 mikilvægustu eiginleikar:")
    print(importance_df.head(10).to_string(index=False))

    plot_predicted_vs_actual(y_test, result["preds"], result["mae"], result["rmse"], result["r2"])
    plot_feature_importance(importance_df)
    plot_residuals(prediction_df["residual"])


if __name__ == "__main__":
    main()
