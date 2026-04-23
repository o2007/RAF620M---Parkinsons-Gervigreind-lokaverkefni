"""
Línulegt aðhvarf fyrir næstu UPDRS-III heimsókn.
"""

import os
import warnings

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from init import (
    LONGITUDINAL_FEATURE_COLS,
    add_longitudinal_features,
    build_coefficient_df,
    build_linear_pipeline,
    build_longitudinal_pairs,
    evaluate_delta_pipeline,
    get_output_dir,
    load_source_tables,
    prepare_clinical_table,
    split_by_patient_time,
)

warnings.filterwarnings("ignore")
OUTPUT_DIR = get_output_dir("linear_regression")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 72)
    print("  PARKINSON - LINEAR REGRESSION")
    print("  Markbreyta: UPDRS-III í næstu heimsókn")
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

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    pairs_train = pairs_df[train_mask].copy().reset_index(drop=True)
    pairs_test = pairs_df[test_mask].copy().reset_index(drop=True)

    baseline_preds = pairs_test["traj_last_score"].to_numpy()
    baseline_mae = float(mean_absolute_error(y_test, baseline_preds))
    baseline_rmse = float(mean_squared_error(y_test, baseline_preds) ** 0.5)
    baseline_r2 = float(r2_score(y_test, baseline_preds))

    print(f"\nTrain rows: {len(X_train):,} | patients: {len(train_patients):,}")
    print(f"Test rows:  {len(X_test):,}  | patients: {len(test_patients):,}")
    print("\nGrunnlíkan: síðasta mælda skor")
    print(f"  MAE:  {baseline_mae:.2f}")
    print(f"  RMSE: {baseline_rmse:.2f}")
    print(f"  R2:   {baseline_r2:.3f}")

    result = evaluate_delta_pipeline(
        "Linear Regression",
        build_linear_pipeline(),
        X_train,
        y_train.to_numpy() - pairs_train["traj_last_score"].to_numpy(),
        X_test,
        y_test,
        pairs_test["traj_last_score"].to_numpy(),
    )

    prediction_df = pairs_test.copy()
    prediction_df["actual"] = y_test.to_numpy()
    prediction_df["predicted"] = result["preds"]
    prediction_df["predicted_delta"] = result["delta_preds"]
    prediction_df.to_csv(os.path.join(OUTPUT_DIR, "linear_regression_predictions.csv"), index=False)

    coef_df = build_coefficient_df(result["pipeline"], LONGITUDINAL_FEATURE_COLS)
    coef_df.to_csv(os.path.join(OUTPUT_DIR, "linear_regression_coefficients.csv"), index=False)

    print("\nNiðurstöður Linear Regression:")
    print(f"  MAE:        {result['mae']:.2f}  (baseline: {baseline_mae:.2f})")
    print(f"  RMSE:       {result['rmse']:.2f}  (baseline: {baseline_rmse:.2f})")
    print(f"  R2:         {result['r2']:.3f}  (baseline: {baseline_r2:.3f})")
    print(f"  Delta MAE:  {result['delta_mae']:.2f}")
    print("\n10 stærstu stuðlar eftir algildi:")
    print(coef_df.head(10).to_string(index=False))
    print("\nVistaðar skrár:")
    print("  linear_regression_predictions.csv")
    print("  linear_regression_coefficients.csv")


if __name__ == "__main__":
    main()
