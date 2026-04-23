"""
XGBoost líkan með langtímaeiginleikum.
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from init import (
    FEATURE_COLS,
    MIN_PRIOR_VISITS,
    MODEL_COLORS,
    RANDOM_SEED,
    XGB_PARAMS,
    add_longitudinal_features,
    build_feature_importance_df,
    build_longitudinal_pairs,
    build_xgb_pipeline,
    evaluate_delta_pipeline,
    get_output_dir,
    load_source_tables,
    prepare_clinical_table,
    save_plot,
    split_by_patient_time,
)

warnings.filterwarnings("ignore")
OUTPUT_DIR = get_output_dir("xgboost_longitudinal")

N_BOOTSTRAPS = 20


def plot_model_comparison(comparison_df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metric_map = [("mae", "MAE"), ("rmse", "RMSE"), ("r2", "R2")]
    for ax, (metric_key, title) in zip(axes, metric_map):
        values = comparison_df[metric_key].tolist()
        labels = comparison_df["model"].tolist()
        colors = [MODEL_COLORS[label] for label in labels]
        bars = ax.bar(labels, values, color=colors, edgecolor="white")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02 * abs(bar.get_height()) + 0.03, f"{value:.3f}", ha="center", va="bottom", fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(min(0, min(values)) - 0.05, 1.05 if metric_key == "r2" else max(values) * 1.25)
    fig.suptitle("Samanburður líkana", fontsize=14)
    plt.tight_layout()
    save_plot("model_comparison.png", OUTPUT_DIR)


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
    save_plot("xgb_mean_updrs_by_year.png")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 72)
    print("  PARKINSON XGBOOST - Spálíkan")
    print("  Úttak = Næsta UPDRS skor, upplýsingar úr öllum fyrrum heimsóknum notaðar")
    print("=" * 72)

    pairs_df = add_longitudinal_features(build_longitudinal_pairs(prepare_clinical_table(load_source_tables())))
    X = pairs_df[FEATURE_COLS].copy()
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

    print(f"\nÞjálfunarraðir: {len(X_train):,} | sjúklingar: {len(train_patients):,}")
    print(f"Prófunarraðir:  {len(X_test):,}  | sjúklingar: {len(test_patients):,}")

    baseline_preds = pairs_test["traj_last_score"].to_numpy()
    baseline_metrics = {
        "mae": float(mean_absolute_error(y_test, baseline_preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, baseline_preds))),
        "r2": float(r2_score(y_test, baseline_preds)),
    }
    print("\nGrunnlíkan: síðasta mælda UPDRS-III skor")
    print(f"  MAE:  {baseline_metrics['mae']:.2f}")
    print(f"  RMSE: {baseline_metrics['rmse']:.2f}")
    print(f"  R2:   {baseline_metrics['r2']:.3f}")

    y_train_delta = y_train - pairs_train["traj_last_score"].to_numpy()
    print(f"\nStillingar líkans: {XGB_PARAMS}")

    model_results = [
        evaluate_delta_pipeline(
            "XGBoost",
            build_xgb_pipeline().set_params(**{f"model__{k}": v for k, v in XGB_PARAMS.items()}),
            X_train,
            y_train_delta,
            X_test,
            y_test,
            pairs_test["traj_last_score"].to_numpy(),
        ),
    ]

    xgb_result = model_results[0]
    comparison_df = pd.DataFrame(
        [
            {"model": "Baseline", "mae": baseline_metrics["mae"], "rmse": baseline_metrics["rmse"], "r2": baseline_metrics["r2"], "delta_mae": np.nan},
            {"model": xgb_result["model"], "mae": xgb_result["mae"], "rmse": xgb_result["rmse"], "r2": xgb_result["r2"], "delta_mae": xgb_result["delta_mae"]},
        ]
    ).sort_values("mae").reset_index(drop=True)
    comparison_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
    print("\nSamanburður á prófunargögnum:")
    print(comparison_df.to_string(index=False))

    residuals = xgb_result["preds"] - y_test.to_numpy()
    mse = float(np.mean(residuals**2))
    rng = np.random.default_rng(RANDOM_SEED)
    train_patient_ids = np.array(sorted(train_patients))
    grouped_train = {patno: group.reset_index(drop=True) for patno, group in pairs_train.groupby("PATNO", sort=False)}
    bootstrap_preds = []
    for bootstrap_idx in range(N_BOOTSTRAPS):
        sampled_patients = rng.choice(train_patient_ids, size=len(train_patient_ids), replace=True)
        boot_df = pd.concat([grouped_train[patno] for patno in sampled_patients], ignore_index=True)
        boot_model = clone(xgb_result["pipeline"])
        boot_model.set_params(model__random_state=RANDOM_SEED + bootstrap_idx + 1)
        boot_target = boot_df["target_updrs3"] - boot_df["traj_last_score"]
        boot_model.fit(boot_df[FEATURE_COLS], boot_target)
        boot_delta_preds = boot_model.predict(X_test)
        bootstrap_preds.append(pairs_test["traj_last_score"].to_numpy() + boot_delta_preds)

    bootstrap_preds = np.asarray(bootstrap_preds)
    mean_boot_pred = bootstrap_preds.mean(axis=0)
    pointwise_variance = bootstrap_preds.var(axis=0, ddof=1) if N_BOOTSTRAPS > 1 else np.zeros(len(X_test))
    signed_biases = mean_boot_pred - y_test.to_numpy()
    bias = float(np.mean(signed_biases))
    bias_sq = float(np.mean(signed_biases**2))
    variance = float(np.mean(pointwise_variance))


    print("\nXGBoost (longitudinal) niðurstöður:")
    print(f"  MAE:         {xgb_result['mae']:.2f}  (Grunnlíkan: {baseline_metrics['mae']:.2f})")
    print(f"  RMSE:        {xgb_result['rmse']:.2f}  (Grunnlíkan: {baseline_metrics['rmse']:.2f})")
    print(f"  R2:          {xgb_result['r2']:.3f}  (Grunnlíkan: {baseline_metrics['r2']:.3f})")
    print(f"  Delta MAE:   {xgb_result['delta_mae']:.2f}")
    print(f"  MAE bæting: {100 * (baseline_metrics['mae'] - xgb_result['mae']) / baseline_metrics['mae']:.1f}% yfir grunnlíkani")
    print("\nBias-variance (prófunargögn):")
    print(f"  Bias: {bias:+.3f}  (jákvætt = ofspá)")
    print(f"  Bias^2: {bias_sq:.3f}")
    print(f"  Meðal fervik líkans:        {variance:.3f}")
    print(f"  Bias^2 + variance:          {bias_sq + variance:.3f}")
    print(f"  Prófunar MSE:    {mse:.3f}")

    prediction_df = pairs_test.copy()
    prediction_df["actual"] = y_test.to_numpy()
    prediction_df["predicted"] = xgb_result["preds"]
    prediction_df["residual"] = prediction_df["predicted"] - prediction_df["actual"]
    prediction_df["abs_error"] = np.abs(prediction_df["residual"])
    prediction_df.to_csv(os.path.join(OUTPUT_DIR, "xgboost_longitudinal_predictions.csv"), index=False)

    importance_df = build_feature_importance_df(xgb_result["pipeline"], FEATURE_COLS)
    importance_df.to_csv(os.path.join(OUTPUT_DIR, "xgb_long_feature_importance.csv"), index=False)

    plot_model_comparison(comparison_df)
    plot_mean_updrs_by_year(prediction_df)
    print("\nMikilvægustu eiginleikar:")
    print(importance_df.head(50).to_string(index=False))
    print("\nAllt vistað í:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
