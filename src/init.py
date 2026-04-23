"""
Sameiginleg hjálparföll fyrir gagnavinnslu, eiginleika og líkön.
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


DATA_DIR = "./data/PPMI_data"
OUTPUT_DIR = "./ppmi_outputs"
RANDOM_SEED = 42
MIN_PRIOR_VISITS = 2

OUTPUT_SUBDIRS = {
    "clinical_only": "clinical_only",
    "clinical_only_delta": "clinical_only_delta",
    "xgboost_longitudinal": "xgboost_longitudinal",
    "random_forest": "random_forest",
    "svm": "svm",
    "linear_regression": "linear_regression",
    "comparison": "comparison",
    "eda": "eda",
    "misc": "misc",
}

VISIT_MAP = {
    "SC": -0.5,
    "BL": 0.0,
    "V04": 1.0,
    "V06": 2.0,
    "V08": 3.0,
    "V10": 4.0,
    "V12": 5.0,
    "V14": 6.0,
    "V16": 7.0,
    "V18": 8.0,
    "V20": 9.0,
}

RF_PARAMS = {"n_estimators": 500, "max_depth": 10, "min_samples_leaf": 4, "max_features": "sqrt"}

XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.025,
    "subsample": 0.85,
    "colsample_bytree": 0.75,
    "min_child_weight": 8,
    "reg_alpha": 0.6,
    "reg_lambda": 2.5,
}

SVR_PARAMS = {"C": 2.0, "epsilon": 0.3, "kernel": "rbf", "gamma": "scale"}

MODEL_COLORS = {
    "Baseline": "#9a9a9a",
    "Grunnlíkan": "#9a9a9a",
    "Linear Regression": "#c17d11",
    "SVM (SVR)": "#d65f5f",
    "Random Forest": "#6acc64",
    "XGBoost": "#4878cf",
}

LONGITUDINAL_FEATURE_COLS = [
    "traj_n_visits",
    "traj_first_score",
    "traj_last_score",
    "traj_mean_score",
    "traj_std_score",
    "traj_min_score",
    "traj_max_score",
    "traj_time_span",
    "traj_last_time",
    "traj_last_delta",
    "traj_last_dt",
    "traj_slope",
    "traj_intercept",
    "traj_r2",
    "traj_accel",
    "traj_ema_score",
    "traj_weighted_slope",
    "traj_quad_coef",
    "traj_score_velocity",
    "traj_recent_rate",
    "traj_visits_per_year",
    "traj_score_range",
    "traj_recent_vs_slope",
    "age_at_last",
    "is_male",
    "moca_last",
    "hoehn_yahr_last",
    "putamen_sbr_last",
    "caudate_sbr_last",
    "is_on_med_last",
    "pdmedyn_last",
    "pdstate_enc_last",
    "dt_to_target",
    "updrs1_last",
    "updrs2_last",
    "cohort",
    "prs_score",
    "trend_projected_updrs3",
    "ema_projected_updrs3",
    "last_score_resid",
    "total_updrs_last",
    "motor_to_total_ratio",
    "nonmotor_burden_last",
    "updrs3_moca_difference",
    "sbr_motor_gap",
    "nonmotor_hy_burden",
    "age_progression_interaction",
    "medication_trajectory_interaction",
    "nonmotor_moca_ratio",
]

# Eldra heiti, haldið inni fyrir eldri skrár.
FEATURE_COLS = LONGITUDINAL_FEATURE_COLS

CLINICAL_ONLY_FEATURE_COLS = [
    "dt_to_target",
    "age_at_last",
    "is_male",
    "moca_last",
    "putamen_sbr_last",
    "caudate_sbr_last",
    "is_on_med_last",
    "pdmedyn_last",
    "pdstate_enc_last",
    "updrs1_last",
    "updrs2_last",
    "prs_score",
    "nonmotor_burden_last",
    "updrs2_moca_difference",
    "datscan_ratio",
]


def get_output_dir(key):
    path = os.path.join(OUTPUT_DIR, OUTPUT_SUBDIRS.get(key, key))
    os.makedirs(path, exist_ok=True)
    return path


def save_plot(filename, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Vistað: {filename}")


def encode_sex(series):
    values = set(series.dropna().unique())
    if values.issubset({1.0, 2.0, 1, 2}):
        return series.map({1: 1, 2: 0, 1.0: 1, 2.0: 0})
    if values.issubset({0.0, 1.0, 0, 1}):
        return series.map({1: 1, 0: 0, 1.0: 1, 0.0: 0})
    return series.astype(str).str.strip().str.upper().map(
        {
            "MALE": 1,
            "M": 1,
            "1": 1,
            "1.0": 1,
            "FEMALE": 0,
            "F": 0,
            "2": 0,
            "2.0": 0,
            "0": 0,
            "0.0": 0,
        }
    )


def load_source_tables():
    return {
        "updrs": pd.read_csv(os.path.join(DATA_DIR, "MDS-UPDRS_Part_III_12Mar2026.csv"), low_memory=False),
        "demo": pd.read_csv(os.path.join(DATA_DIR, "Demographics_12Mar2026.csv"), low_memory=False),
        "moca": pd.read_csv(os.path.join(DATA_DIR, "Montreal_Cognitive_Assessment__MoCA__12Mar2026.csv"), low_memory=False),
        "age": pd.read_csv(os.path.join(DATA_DIR, "Age_at_visit_12Mar2026.csv"), low_memory=False),
        "sbr": pd.read_csv(os.path.join(DATA_DIR, "Xing_Core_Lab_-_Quant_SBR_12Mar2026.csv"), low_memory=False),
        "updrs1": pd.read_csv(os.path.join(DATA_DIR, "MDS-UPDRS_Part_I_12Mar2026.csv"), low_memory=False),
        "updrs2": pd.read_csv(os.path.join(DATA_DIR, "MDS_UPDRS_Part_II__Patient_Questionnaire_12Mar2026.csv"), low_memory=False),
        "prs": pd.read_csv(os.path.join(DATA_DIR, "PPMI_Project_9001_20250624_12Mar2026.csv"), low_memory=False),
        "participant": pd.read_csv(os.path.join(DATA_DIR, "Participant_Status_12Mar2026.csv"), low_memory=False),
    }


def prepare_clinical_table(source_tables):
    updrs = source_tables["updrs"].copy()
    updrs["visit_year"] = updrs["EVENT_ID"].map(VISIT_MAP)
    clinical = updrs.dropna(subset=["visit_year", "NP3TOT"]).copy()

    if "NHY" in clinical.columns:
        clinical["NHY"] = clinical["NHY"].replace(101.0, np.nan)

    clinical = clinical.merge(source_tables["age"][["PATNO", "EVENT_ID", "AGE_AT_VISIT"]], on=["PATNO", "EVENT_ID"], how="left")
    clinical = clinical.merge(source_tables["demo"][["PATNO", "SEX"]].drop_duplicates("PATNO"), on="PATNO", how="left")
    clinical["is_male"] = encode_sex(clinical["SEX"])
    clinical = clinical.merge(source_tables["moca"][["PATNO", "EVENT_ID", "MCATOT"]].dropna(subset=["MCATOT"]), on=["PATNO", "EVENT_ID"], how="left")

    sbr = source_tables["sbr"].copy()
    sbr["putamen_sbr"] = (sbr["PUTAMEN_L_REF_CWM"] + sbr["PUTAMEN_R_REF_CWM"]) / 2
    sbr["caudate_sbr"] = (sbr["CAUDATE_L_REF_CWM"] + sbr["CAUDATE_R_REF_CWM"]) / 2
    clinical = clinical.merge(sbr[["PATNO", "EVENT_ID", "putamen_sbr", "caudate_sbr"]].dropna(), on=["PATNO", "EVENT_ID"], how="left")

    clinical["PDSTATE"] = clinical["PDSTATE"].fillna("MISSING")
    clinical["is_on_med"] = (clinical["PDSTATE"] == "ON").astype(int)
    clinical["pdmedyn_clean"] = clinical["PDMEDYN"].fillna(0)
    clinical["pdstate_enc"] = clinical["PDSTATE"].map({"OFF": 0, "ON": 1, "ON_WITHOUT_DOPA": 2, "MISSING": 3}).fillna(4)

    clinical = clinical.merge(source_tables["updrs1"][["PATNO", "EVENT_ID", "NP1RTOT"]].dropna(subset=["NP1RTOT"]), on=["PATNO", "EVENT_ID"], how="left")
    clinical = clinical.merge(source_tables["updrs2"][["PATNO", "EVENT_ID", "NP2PTOT"]].dropna(subset=["NP2PTOT"]), on=["PATNO", "EVENT_ID"], how="left")
    clinical = clinical.merge(source_tables["participant"][["PATNO", "COHORT"]].drop_duplicates("PATNO"), on="PATNO", how="left")
    clinical["cohort"] = clinical["COHORT"].fillna(-1).astype(int)
    clinical = clinical.merge(
        source_tables["prs"][["PATNO", "Genetic_PRS_PRS88"]].drop_duplicates("PATNO").rename(columns={"Genetic_PRS_PRS88": "prs_score"}),
        on="PATNO",
        how="left",
    )

    return clinical.sort_values(["PATNO", "visit_year"]).reset_index(drop=True)


def trajectory_features(prior_rows):
    deduped = prior_rows.groupby("visit_year", as_index=False)["NP3TOT"].mean().sort_values("visit_year")
    scores = deduped["NP3TOT"].to_numpy(dtype=float)
    times = deduped["visit_year"].to_numpy(dtype=float)
    n_visits = len(scores)
    time_span = float(times[-1] - times[0])
    times_vary = float(np.ptp(times)) > 0

    features = {
        "traj_n_visits": float(n_visits),
        "traj_first_score": float(scores[0]),
        "traj_last_score": float(scores[-1]),
        "traj_mean_score": float(np.mean(scores)),
        "traj_std_score": float(np.std(scores)) if n_visits > 1 else 0.0,
        "traj_min_score": float(np.min(scores)),
        "traj_max_score": float(np.max(scores)),
        "traj_time_span": time_span,
        "traj_last_time": float(times[-1]),
        "traj_last_delta": float(scores[-1] - scores[-2]) if n_visits >= 2 else 0.0,
        "traj_last_dt": float(times[-1] - times[-2]) if n_visits >= 2 else 0.0,
    }

    if n_visits >= 2 and times_vary:
        slope, intercept, corr, _, _ = linregress(times, scores)
        features["traj_slope"] = float(slope)
        features["traj_intercept"] = float(intercept)
        features["traj_r2"] = float(corr**2)
    else:
        features["traj_slope"] = 0.0
        features["traj_intercept"] = float(scores[0])
        features["traj_r2"] = 0.0

    if n_visits >= 3 and times_vary:
        deltas = np.diff(scores)
        dt = np.diff(times)
        dt[dt == 0] = 1e-6
        rates = deltas / dt
        mid_times = (times[:-1] + times[1:]) / 2
        features["traj_accel"] = float(linregress(mid_times, rates).slope) if float(np.ptp(mid_times)) > 0 else 0.0
        features["traj_quad_coef"] = float(np.polyfit(times, scores, 2)[0])
    else:
        features["traj_accel"] = 0.0
        features["traj_quad_coef"] = 0.0

    ema = float(scores[0])
    for score in scores[1:]:
        ema = 0.5 * float(score) + 0.5 * ema
    features["traj_ema_score"] = ema

    if n_visits >= 2 and times_vary:
        weights = np.arange(1, n_visits + 1, dtype=float)
        mean_t = np.average(times, weights=weights)
        mean_s = np.average(scores, weights=weights)
        cov = np.sum(weights * (times - mean_t) * (scores - mean_s))
        var = np.sum(weights * (times - mean_t) ** 2)
        features["traj_weighted_slope"] = float(cov / var) if var > 0 else 0.0
    else:
        features["traj_weighted_slope"] = 0.0

    features["traj_score_velocity"] = float((scores[-1] - scores[0]) / time_span) if times_vary else 0.0
    if n_visits >= 2:
        last_dt = float(times[-1] - times[-2])
        features["traj_recent_rate"] = float((scores[-1] - scores[-2]) / last_dt) if last_dt > 0 else 0.0
    else:
        features["traj_recent_rate"] = 0.0
    features["traj_visits_per_year"] = float(n_visits / time_span) if times_vary else float(n_visits)
    features["traj_score_range"] = float(np.max(scores) - np.min(scores))
    features["traj_recent_vs_slope"] = float(features["traj_recent_rate"] / (abs(features["traj_slope"]) + 1e-6))
    return features


def build_longitudinal_pairs(clinical_table):
    rows = []
    for patno, patient_visits in clinical_table.groupby("PATNO"):
        patient_visits = patient_visits.sort_values("visit_year").reset_index(drop=True)
        visit_years = sorted(patient_visits["visit_year"].unique())
        if len(visit_years) < MIN_PRIOR_VISITS + 1:
            continue

        for idx in range(MIN_PRIOR_VISITS, len(visit_years)):
            prior_cutoff = visit_years[idx - 1]
            target_year = visit_years[idx]
            prior_rows = patient_visits[patient_visits["visit_year"] <= prior_cutoff]
            target_rows = patient_visits[patient_visits["visit_year"] == target_year]
            if prior_rows.empty or target_rows.empty:
                continue

            last_visit = prior_rows.sort_values("visit_year").iloc[-1]
            row = {
                "PATNO": patno,
                "target_visit_yr": float(target_year),
                "target_updrs3": float(target_rows["NP3TOT"].mean()),
                "age_at_last": last_visit.get("AGE_AT_VISIT", np.nan),
                "is_male": last_visit.get("is_male", np.nan),
                "moca_last": last_visit.get("MCATOT", np.nan),
                "hoehn_yahr_last": last_visit.get("NHY", np.nan),
                "putamen_sbr_last": last_visit.get("putamen_sbr", np.nan),
                "caudate_sbr_last": last_visit.get("caudate_sbr", np.nan),
                "is_on_med_last": last_visit.get("is_on_med", np.nan),
                "pdmedyn_last": last_visit.get("pdmedyn_clean", np.nan),
                "pdstate_enc_last": last_visit.get("pdstate_enc", np.nan),
                "dt_to_target": float(target_year - prior_cutoff),
                "updrs1_last": last_visit.get("NP1RTOT", np.nan),
                "updrs2_last": last_visit.get("NP2PTOT", np.nan),
                "cohort": last_visit.get("cohort", -1),
                "prs_score": last_visit.get("prs_score", np.nan),
            }
            row.update(trajectory_features(prior_rows))
            rows.append(row)

    return pd.DataFrame(rows).dropna(subset=["target_updrs3"]).reset_index(drop=True)


def add_longitudinal_features(pairs_df):
    pairs_df = pairs_df.copy()
    pairs_df["trend_projected_updrs3"] = pairs_df["traj_last_score"] + pairs_df["traj_slope"] * pairs_df["dt_to_target"]
    pairs_df["ema_projected_updrs3"] = pairs_df["traj_ema_score"] + pairs_df["traj_weighted_slope"] * pairs_df["dt_to_target"]
    pairs_df["last_score_resid"] = pairs_df["traj_last_score"] - (pairs_df["traj_intercept"] + pairs_df["traj_slope"] * pairs_df["traj_last_time"])
    pairs_df["total_updrs_last"] = pairs_df["traj_last_score"].fillna(0) + pairs_df["updrs1_last"].fillna(0) + pairs_df["updrs2_last"].fillna(0)
    pairs_df["motor_to_total_ratio"] = pairs_df["traj_last_score"] / (pairs_df["total_updrs_last"] + 1e-6)
    pairs_df["nonmotor_burden_last"] = pairs_df["updrs1_last"].fillna(0) + pairs_df["updrs2_last"].fillna(0)
    pairs_df["updrs3_moca_difference"] = pairs_df["traj_last_score"] - pairs_df["moca_last"].fillna(pairs_df["moca_last"].median())
    pairs_df["sbr_motor_gap"] = pairs_df["traj_last_score"] / (pairs_df["putamen_sbr_last"].fillna(pairs_df["putamen_sbr_last"].median()) + 1e-3)
    pairs_df["nonmotor_hy_burden"] = pairs_df["nonmotor_burden_last"] + pairs_df["hoehn_yahr_last"].fillna(0) * 5.0
    pairs_df["age_progression_interaction"] = pairs_df["age_at_last"].fillna(pairs_df["age_at_last"].median()) * pairs_df["traj_score_velocity"]
    pairs_df["medication_trajectory_interaction"] = pairs_df["is_on_med_last"].fillna(0) * pairs_df["traj_recent_rate"]
    pairs_df["nonmotor_moca_ratio"] = pairs_df["nonmotor_burden_last"] / (pairs_df["moca_last"].fillna(pairs_df["moca_last"].median()) + 1e-3)
    return pairs_df


# Eldri heiti, haldið inni fyrir eldri skrár.
build_prediction_pairs = build_longitudinal_pairs
add_derived_features = add_longitudinal_features


def build_clinical_only_pairs(clinical_table):
    rows = []
    for patno, patient_visits in clinical_table.groupby("PATNO"):
        patient_visits = patient_visits.sort_values("visit_year").reset_index(drop=True)
        visit_years = sorted(patient_visits["visit_year"].unique())
        if len(visit_years) < 2:
            continue

        for idx in range(1, len(visit_years)):
            prior_year = visit_years[idx - 1]
            target_year = visit_years[idx]
            prior_visit = patient_visits[patient_visits["visit_year"] == prior_year]
            target_visit = patient_visits[patient_visits["visit_year"] == target_year]
            if prior_visit.empty or target_visit.empty:
                continue

            last_visit = prior_visit.sort_values("visit_year").iloc[-1]
            rows.append(
                {
                    "PATNO": patno,
                    "target_visit_yr": float(target_year),
                    "target_updrs3": float(target_visit["NP3TOT"].mean()),
                    "dt_to_target": float(target_year - prior_year),
                    "baseline_prev_updrs3": float(last_visit.get("NP3TOT", np.nan)),
                    "age_at_last": last_visit.get("AGE_AT_VISIT", np.nan),
                    "is_male": last_visit.get("is_male", np.nan),
                    "moca_last": last_visit.get("MCATOT", np.nan),
                    "putamen_sbr_last": last_visit.get("putamen_sbr", np.nan),
                    "caudate_sbr_last": last_visit.get("caudate_sbr", np.nan),
                    "is_on_med_last": last_visit.get("is_on_med", np.nan),
                    "pdmedyn_last": last_visit.get("pdmedyn_clean", np.nan),
                    "pdstate_enc_last": last_visit.get("pdstate_enc", np.nan),
                    "updrs1_last": last_visit.get("NP1RTOT", np.nan),
                    "updrs2_last": last_visit.get("NP2PTOT", np.nan),
                    "prs_score": last_visit.get("prs_score", np.nan),
                }
            )
    return pd.DataFrame(rows).dropna(subset=["target_updrs3"]).reset_index(drop=True)


def add_clinical_only_features(pairs_df):
    pairs_df = pairs_df.copy()
    pairs_df["nonmotor_burden_last"] = pairs_df["updrs1_last"].fillna(0) + pairs_df["updrs2_last"].fillna(0)
    pairs_df["updrs2_moca_difference"] = pairs_df["updrs2_last"].fillna(0) - pairs_df["moca_last"].fillna(pairs_df["moca_last"].median())
    pairs_df["datscan_ratio"] = pairs_df["putamen_sbr_last"].fillna(pairs_df["putamen_sbr_last"].median()) / (pairs_df["caudate_sbr_last"].fillna(pairs_df["caudate_sbr_last"].median()) + 1e-3)
    return pairs_df


def split_by_patient_time(pairs_df):
    patient_median_year = pairs_df.groupby("PATNO")["target_visit_yr"].median().sort_values()
    cutoff_idx = int(len(patient_median_year) * 0.80)
    train_patients = set(patient_median_year.index[:cutoff_idx])
    test_patients = set(patient_median_year.index[cutoff_idx:])
    return train_patients, test_patients


def build_rf_pipeline():
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1))])


def build_xgb_pipeline():
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", XGBRegressor(objective="reg:squarederror", random_state=RANDOM_SEED, n_jobs=-1))])


def build_svr_pipeline():
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", SVR())])


def build_linear_pipeline():
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", LinearRegression())])


def evaluate_delta_pipeline(name, pipeline, X_train, y_train_delta, X_test, y_test, baseline_scores):
    pipeline.fit(X_train, y_train_delta)
    delta_preds = pipeline.predict(X_test)
    preds = baseline_scores + delta_preds
    return {
        "model": name,
        "pipeline": pipeline,
        "preds": preds,
        "delta_preds": delta_preds,
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "r2": float(r2_score(y_test, preds)),
        "delta_mae": float(mean_absolute_error(y_test - baseline_scores, delta_preds)),
    }
def build_feature_importance_df(pipeline, feature_cols):
    model = pipeline.named_steps["model"]
    return pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_}).sort_values("importance", ascending=False).reset_index(drop=True)


def build_coefficient_df(pipeline, feature_cols):
    model = pipeline.named_steps["model"]
    return pd.DataFrame({"feature": feature_cols, "coefficient": model.coef_}).sort_values("coefficient", key=lambda s: np.abs(s), ascending=False).reset_index(drop=True)
