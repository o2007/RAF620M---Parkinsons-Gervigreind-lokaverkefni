"""
SVM/SVR líkan fyrir næstu UPDRS-III heimsókn.
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from init import (
    FEATURE_COLS,
    MIN_PRIOR_VISITS,
    SVR_PARAMS,
    add_longitudinal_features,
    build_longitudinal_pairs,
    build_svr_pipeline,
    evaluate_delta_pipeline,
    get_output_dir,
    load_source_tables,
    prepare_clinical_table,
    save_plot,
    split_by_patient_time,
)

warnings.filterwarnings("ignore")
OUTPUT_DIR = get_output_dir("svm")

PLOT_COLORS = {"actual": "steelblue", "predicted": "#d65f5f"}


def plot_predicted_vs_actual(y_test, preds, mae, rmse, r2):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, preds, alpha=0.35, s=18, color=PLOT_COLORS["actual"])
    limits = [min(float(y_test.min()), float(preds.min())) - 1, max(float(y_test.max()), float(preds.max())) + 1]
    ax.plot(limits, limits, "r--", linewidth=1.5, label="Fullkomin spá")
    ax.set_xlabel("Raunverulegt UPDRS-III (næsta heimsókn)", fontsize=12)
    ax.set_ylabel("Spáð UPDRS-III", fontsize=12)
    ax.set_title(f"SVM (SVR): spáð vs raunverulegt\nMAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.3f}", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot("svm_predicted_vs_actual.png", OUTPUT_DIR)


def plot_support_vectors_summary(pipeline, n_test):
    model = pipeline.named_steps["model"]
    n_sv = model.n_support_vectors_ if hasattr(model, "n_support_vectors_") else len(model.support_)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Support Vectors", "Test Samples"], [n_sv, n_test], color=[PLOT_COLORS["predicted"], PLOT_COLORS["actual"]], edgecolor="white", width=0.5)
    ax.set_title("SVM: support vectors og prófunarraðir", fontsize=13)
    ax.set_ylabel("Fjöldi", fontsize=12)
    for i, value in enumerate([n_sv, n_test]):
        ax.text(i, value + 1, str(value), ha="center", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    save_plot("svm_support_vectors.png", OUTPUT_DIR)


def plot_residuals(residuals):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=45, edgecolor="white", color=PLOT_COLORS["actual"])
    ax.axvline(0, linestyle="--", color="red", linewidth=2, label="Engin skekkja")
    ax.axvline(float(np.mean(residuals)), linestyle="--", color="orange", linewidth=1.5, label=f"Meðaltal = {float(np.mean(residuals)):.2f}")
    ax.set_xlabel("Skekkja (spáð - raunverulegt)", fontsize=12)
    ax.set_ylabel("Fjöldi", fontsize=12)
    ax.set_title("Skekkjudreifing - SVM (SVR)", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot("svm_residuals.png", OUTPUT_DIR)


def plot_mean_updrs_by_year(prediction_df):
    plot_df = prediction_df.copy()
    plot_df["years_from_first_prediction"] = plot_df["traj_n_visits"] - MIN_PRIOR_VISITS

    grouped = (
        plot_df.groupby("years_from_first_prediction", as_index=False)[["actual", "predicted"]]
        .mean()
        .sort_values("years_from_first_prediction")
    )

    x = grouped["years_from_first_prediction"].to_numpy(dtype=int)
    actual = grouped["actual"].to_numpy()
    predicted = grouped["predicted"].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, actual, marker="o", linewidth=2.5, color="#4878cf", label="Mælt meðalgildi")
    ax.plot(x, predicted, marker="o", linewidth=2.5, color="#e87c2b", label="Spáð gildi")

    for xi, yi in zip(x, actual):
        ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points", xytext=(0, -14), ha="center", va="top", fontsize=9, color="#4878cf")
    for xi, yi in zip(x, predicted):
        ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points", xytext=(0, 14), ha="center", va="bottom", fontsize=9, color="#e87c2b")

    ax.set_title("Meðal UPDRS-III prófunarsjúklinga\nMælt vs spáð miðað við tíma frá fyrstu spáheimsókn", fontsize=17)
    ax.set_xlabel("Ár frá fyrstu spáheimsókn", fontsize=14)
    ax.set_ylabel("UPDRS-III skor (hærra = verra)", fontsize=14)
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    save_plot("svm_mean_updrs_by_year.png", OUTPUT_DIR)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 72)
    print("  PARKINSON - SVM")
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
    print(f"\nStillingar líkans: {SVR_PARAMS}")

    result = evaluate_delta_pipeline(
        "SVM (SVR)",
        build_svr_pipeline().set_params(**{f"model__{key}": value for key, value in SVR_PARAMS.items()}),
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
    prediction_df.to_csv(os.path.join(OUTPUT_DIR, "svm_predictions.csv"), index=False)

    perm = permutation_importance(result["pipeline"], X_test, y_test, n_repeats=10, random_state=42, scoring="neg_mean_absolute_error")
    importance_df = pd.DataFrame({"feature": FEATURE_COLS, "importance": perm.importances_mean, "std": perm.importances_std}).sort_values(
        "importance", ascending=False
    )
    importance_df.to_csv(os.path.join(OUTPUT_DIR, "svm_feature_importance.csv"), index=False)

    print("\nNiðurstöður SVM:")
    print(f"  MAE:        {result['mae']:.2f}  (baseline: {baseline_mae:.2f})")
    print(f"  RMSE:       {result['rmse']:.2f}  (baseline: {baseline_rmse:.2f})")
    print(f"  R2:         {result['r2']:.3f}  (baseline: {baseline_r2:.3f})")
    print(f"  Delta MAE:  {result['delta_mae']:.2f}")
    print("\nMikilvægustu eiginleikar samkvæmt permutation importance:")
    print(importance_df.head(50).to_string(index=False))

    plot_predicted_vs_actual(y_test, result["preds"], result["mae"], result["rmse"], result["r2"])
    plot_support_vectors_summary(result["pipeline"], len(X_test))
    plot_residuals(prediction_df["residual"])
    plot_mean_updrs_by_year(prediction_df)


if __name__ == "__main__":
    main()
