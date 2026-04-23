"""
Myndir fyrir líkananiðurstöður og gagnayfirlit.
"""

import os
import warnings
from textwrap import fill

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from init import (
    CLINICAL_ONLY_FEATURE_COLS,
    FEATURE_COLS,
    MODEL_COLORS,
    add_clinical_only_features,
    add_longitudinal_features,
    build_clinical_only_pairs,
    build_longitudinal_pairs,
    get_output_dir,
    load_source_tables,
    prepare_clinical_table,
)

matplotlib.use("Agg")
warnings.filterwarnings("ignore")


EDA_DIR = get_output_dir("eda")
COMPARISON_DIR = get_output_dir("comparison")
MISC_DIR = get_output_dir("misc")


def save_fig(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Vistað: {path}")


def safe_read_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def slugify_model(name):
    mapping = {
        "Linear Regression": "linear_regression",
        "Random Forest": "random_forest",
        "SVM (SVR)": "svm",
        "XGBoost": "xgboost",
        "Baseline": "baseline",
    }
    return mapping.get(name, name.lower().replace(" ", "_").replace("(", "").replace(")", ""))


def infer_baseline_column(df):
    for col in ["traj_last_score", "baseline_prev_updrs3", "predicted_baseline"]:
        if col in df.columns:
            return col
    return None


def prepare_prediction_frame(df, pred_col):
    frame = df.copy()
    frame["actual"] = frame["actual"] if "actual" in frame.columns else frame["target_updrs3"]
    frame["predicted"] = frame[pred_col]
    baseline_col = infer_baseline_column(frame)
    if baseline_col is not None:
        frame["baseline_score"] = frame[baseline_col]
        frame["actual_change"] = frame["actual"] - frame["baseline_score"]
        if pred_col.endswith("_delta"):
            frame["predicted_change"] = frame[pred_col]
        else:
            frame["predicted_change"] = frame["predicted"] - frame["baseline_score"]
    elif "predicted_delta" in frame.columns:
        frame["predicted_change"] = frame["predicted_delta"]
    frame["residual"] = frame["predicted"] - frame["actual"]
    frame["abs_error"] = np.abs(frame["residual"])
    return frame


def feature_group(name):
    if name.startswith("traj_") or name in {
        "trend_projected_updrs3",
        "ema_projected_updrs3",
        "last_score_resid",
        "total_updrs_last",
        "motor_to_total_ratio",
    }:
        return "Hreyfiferill"
    if "moca" in name or "cognitive" in name:
        return "Vitsmunir"
    if "putamen" in name or "caudate" in name or "sbr" in name or "datscan" in name:
        return "Myndgreining"
    if "med" in name or "pdstate" in name or "pdmed" in name:
        return "Lyf"
    if "age" in name or "male" in name or "cohort" in name or "prs" in name:
        return "Lydfraedi og erfdir"
    if "updrs1" in name or "updrs2" in name or "nonmotor" in name:
        return "Klinisk einkenni"
    if "dt_to_target" in name:
        return "Spatimi"
    return "Annad"


def prettify_feature_name(name):
    return fill(name.replace("_", " "), width=24)


def plot_missingness_overview(clinical_table):
    miss = (clinical_table.isna().mean() * 100).sort_values(ascending=False)
    labels = [prettify_feature_name(col) for col in miss.index]
    values = miss.values
    n_rows = len(miss)

    fig_height = max(8, 0.28 * n_rows + 2.8)
    fig, ax = plt.subplots(figsize=(12.5, fig_height))
    fig.patch.set_facecolor("#f6f1e8")
    ax.set_facecolor("#fffaf2")

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "missingness", ["#2a9d8f", "#e9c46a", "#f4a261", "#d1495b"]
    )
    norm = matplotlib.colors.Normalize(vmin=0, vmax=100)
    colors = cmap(norm(values))

    y = np.arange(n_rows)
    bars = ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.9, height=0.78)

    for x in [25, 50, 75]:
        ax.axvline(x, color="#c9b79c", linestyle="--", linewidth=0.8, alpha=0.8, zorder=0)

    for bar, value in zip(bars, values):
        x = bar.get_width()
        y_mid = bar.get_y() + bar.get_height() / 2
        label = f"{value:.1f}%"
        if value >= 12:
            ax.text(x - 1.2, y_mid, label, va="center", ha="right", fontsize=8.5, color="#1f1f1f")
        else:
            ax.text(min(x + 1.2, 99.2), y_mid, label, va="center", ha="left", fontsize=8.5, color="#1f1f1f")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Vantar gildi (%)", fontsize=11)
    ax.set_title("Vantar Gildi Fyrir Allar Breytur", fontsize=18, weight="bold", pad=14)
    ax.text(
        0.0,
        1.01,
        f"Undirhópur: allar heimsóknir með gilt UPDRS-III skor (n = {len(clinical_table):,} raðir, {clinical_table['PATNO'].nunique():,} sjúklingar)",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#5f584f",
    )
    ax.grid(True, axis="x", color="#ddcfb7", alpha=0.65, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8b7e6b")
    ax.spines["bottom"].set_color("#8b7e6b")
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", length=0)

    save_fig(os.path.join(EDA_DIR, "vantandi_gildi_updrs3_hopur.png"))


def plot_missingness_report_style(clinical_table, source_tables):
    sbr = source_tables["sbr"].copy()
    sbr["striatum_sbr"] = (sbr["STRIATUM_L_REF_CWM"] + sbr["STRIATUM_R_REF_CWM"]) / 2
    clinical_with_striatum = clinical_table.merge(
        sbr[["PATNO", "EVENT_ID", "striatum_sbr"]].dropna(),
        on=["PATNO", "EVENT_ID"],
        how="left",
    )

    selected = [
        ("striatum_sbr", "DaTscan: Striatum SBR"),
        ("caudate_sbr", "DaTscan: Caudate SBR"),
        ("putamen_sbr", "DaTscan: Putamen SBR"),
        ("PDMEDDT", "Heildar-LEDD"),
        ("MCATOT", "MoCA heildarskor"),
        ("NP3TOT", "UPDRS III heildarskor"),
        ("NHY", "Hoehn og Yahr stig"),
        ("AGE_AT_VISIT", "Aldur við heimsókn"),
        ("NP2PTOT", "UPDRS II heildarskor"),
        ("NP1RTOT", "UPDRS I heildarskor"),
    ]

    missing_df = pd.DataFrame(
        [
            {"label": label, "missing_pct": clinical_with_striatum[column].isna().mean() * 100}
            for column, label in selected
            if column in clinical_with_striatum.columns
        ]
    ).sort_values("missing_pct", ascending=False)

    fig, ax = plt.subplots(figsize=(13, 7))
    bars = ax.bar(
        missing_df["label"],
        missing_df["missing_pct"],
        color="#1c7fb6",
        edgecolor="white",
        linewidth=1.2,
    )

    ax.set_title("Auð gögn eftir breytum", fontsize=20, pad=10)
    ax.set_ylabel("Vantar (%)", fontsize=14)
    ax.set_ylim(0, max(85, float(missing_df["missing_pct"].max()) + 5))
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=10)
    ax.tick_params(axis="y", labelsize=11)

    for bar, value in zip(bars, missing_df["missing_pct"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.0,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#1f1f1f",
        )

    save_fig(os.path.join(EDA_DIR, "vantandi_gildi_skyrsla.png"))


def plot_workflow_diagram():
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")
    steps = [
        ("PPMI gogn", 0.08),
        ("Forvinnsla", 0.28),
        ("Eiginleikar", 0.48),
        ("Þjálfun", 0.68),
        ("Nidurstodur", 0.88),
    ]
    for label, xpos in steps:
        ax.text(xpos, 0.5, label, ha="center", va="center", fontsize=14, bbox=dict(boxstyle="round,pad=0.5", facecolor="#e9f2ff", edgecolor="#4878cf"))
    for idx in range(len(steps) - 1):
        ax.annotate("", xy=(steps[idx + 1][1] - 0.08, 0.5), xytext=(steps[idx][1] + 0.08, 0.5), arrowprops=dict(arrowstyle="->", lw=2))
    save_fig(os.path.join(MISC_DIR, "verkfladi_likansins.png"))


def plot_eda(clinical_table, longitudinal_pairs, source_tables):
    cohort_counts = clinical_table["cohort"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(cohort_counts.index.astype(str), cohort_counts.values, color="#4878cf", edgecolor="white")
    ax.set_title("Dreifing markhópa", fontsize=13)
    ax.set_xlabel("Kohorta")
    ax.set_ylabel("Fjoldi")
    ax.grid(True, alpha=0.3, axis="y")
    save_fig(os.path.join(EDA_DIR, "kohortudreifing.png"))

    sex_counts = clinical_table["is_male"].map({1: "Karl", 0: "Kona"}).fillna("Vantar").value_counts()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(sex_counts.index, sex_counts.values, color="#6acc64", edgecolor="white")
    ax.set_title("Kynjaskipting", fontsize=13)
    ax.set_ylabel("Fjoldi")
    ax.grid(True, alpha=0.3, axis="y")
    save_fig(os.path.join(EDA_DIR, "kynjaskipting.png"))

    fig, ax = plt.subplots(figsize=(8, 5))
    clinical_table["AGE_AT_VISIT"].dropna().hist(ax=ax, bins=30, color="#e87c2b", edgecolor="white")
    ax.set_title("Aldursdreifing vid heimsokn", fontsize=13)
    ax.set_xlabel("Aldur")
    ax.set_ylabel("Fjoldi")
    save_fig(os.path.join(EDA_DIR, "aldursdreifing.png"))

    fig, ax = plt.subplots(figsize=(8, 5))
    clinical_table.groupby("PATNO").size().hist(ax=ax, bins=20, color="#d65f5f", edgecolor="white")
    ax.set_title("Fjoldi heimsokna a hvern sjukling", fontsize=13)
    ax.set_xlabel("Fjoldi heimsokna")
    ax.set_ylabel("Fjoldi sjuklinga")
    save_fig(os.path.join(EDA_DIR, "heimsoknir_a_sjukling.png"))

    fig, ax = plt.subplots(figsize=(8, 5))
    longitudinal_pairs["target_updrs3"].hist(ax=ax, bins=35, color="#4878cf", edgecolor="white")
    ax.set_title("Dreifing markbreytu: naesta UPDRS-III", fontsize=13)
    ax.set_xlabel("UPDRS-III")
    ax.set_ylabel("Fjoldi")
    save_fig(os.path.join(EDA_DIR, "dreifing_markbreytu.png"))

    fig, ax = plt.subplots(figsize=(8, 5))
    (longitudinal_pairs["target_updrs3"] - longitudinal_pairs["traj_last_score"]).hist(ax=ax, bins=35, color="#6acc64", edgecolor="white")
    ax.set_title("Dreifing breytingar a UPDRS-III", fontsize=13)
    ax.set_xlabel("Breyting")
    ax.set_ylabel("Fjoldi")
    save_fig(os.path.join(EDA_DIR, "dreifing_breytingar.png"))

    plot_missingness_overview(clinical_table)
    plot_missingness_report_style(clinical_table, source_tables)

    corr_cols = [col for col in ["NP3TOT", "NP2PTOT", "NP1RTOT", "AGE_AT_VISIT", "MCATOT", "NHY", "putamen_sbr", "caudate_sbr"] if col in clinical_table.columns]
    corr = clinical_table[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_cols)))
    ax.set_xticklabels(corr_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(corr_cols)))
    ax.set_yticklabels(corr_cols)
    ax.set_title("Fylgnifylki kliniskra breyta", fontsize=13)
    plt.colorbar(im, ax=ax)
    save_fig(os.path.join(EDA_DIR, "fylgnifylki.png"))


def plot_heldout_model_comparison():
    comparison_df = safe_read_csv(os.path.join(COMPARISON_DIR, "comparison_all_models.csv"))
    if comparison_df is None:
        return
    metrics = [("mae", "MAE"), ("rmse", "RMSE"), ("r2", "R2")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (metric, title) in zip(axes, metrics):
        vals = comparison_df[metric].tolist()
        labels = comparison_df["model"].tolist()
        colors = [MODEL_COLORS.get(label, "#888888") for label in labels]
        ax.bar(labels, vals, color=colors, edgecolor="white")
        ax.set_title(f"Haldið prófunarsett: {title}", fontsize=12)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, alpha=0.3, axis="y")
    save_fig(os.path.join(COMPARISON_DIR, "haldud_samanburdur_likana.png"))


def plot_clinical_vs_full():
    full_df = safe_read_csv(os.path.join(COMPARISON_DIR, "comparison_all_models.csv"))
    clinical_df = safe_read_csv(os.path.join(get_output_dir("clinical_only_delta"), "clinical_only_delta_model_comparison.csv"))
    if full_df is None or clinical_df is None:
        return

    full_best = full_df[full_df["model"] != "Grunnlíkan"].sort_values("mae").iloc[0]
    clinical_best = clinical_df[clinical_df["model"] != "Baseline"].sort_values("mae").iloc[0]
    plot_df = pd.DataFrame(
        [
            {"uppsetning": "Kliniskt einungis", "mae": clinical_best["mae"], "rmse": clinical_best["rmse"], "r2": clinical_best["r2"]},
            {"uppsetning": "Fullt likan", "mae": full_best["mae"], "rmse": full_best["rmse"], "r2": full_best["r2"]},
        ]
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric in zip(axes, ["mae", "rmse", "r2"]):
        ax.bar(plot_df["uppsetning"], plot_df[metric], color=["#e87c2b", "#4878cf"], edgecolor="white")
        ax.set_title(metric.upper(), fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Kliniskt likan vs fullt likan", fontsize=13)
    save_fig(os.path.join(COMPARISON_DIR, "klinisk_vs_fullt_likan.png"))


def plot_model_frame(frame, output_dir, prefix, title_prefix, importance_df=None, importance_kind="importance"):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(frame["actual"], frame["predicted"], alpha=0.35, s=18, color="#4878cf")
    lims = [min(float(frame["actual"].min()), float(frame["predicted"].min())) - 1, max(float(frame["actual"].max()), float(frame["predicted"].max())) + 1]
    ax.plot(lims, lims, "r--", linewidth=1.5)
    ax.set_xlabel("Raunverulegt UPDRS-III")
    ax.set_ylabel("Spáð UPDRS-III")
    ax.set_title(f"{title_prefix}: raun vs spá", fontsize=13)
    ax.grid(True, alpha=0.3)
    save_fig(os.path.join(output_dir, f"{prefix}_raun_vs_spa.png"))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(frame["actual"], bins=30, alpha=0.7, label="Raun", color="#4878cf", edgecolor="white")
    ax.hist(frame["predicted"], bins=30, alpha=0.7, label="Spá", color="#e87c2b", edgecolor="white")
    ax.set_title(f"{title_prefix}: dreifing raun og spár", fontsize=13)
    ax.legend()
    save_fig(os.path.join(output_dir, f"{prefix}_dreifing_raun_og_spa.png"))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(frame["residual"], bins=35, color="#6acc64", edgecolor="white")
    ax.axvline(0, color="red", linestyle="--")
    ax.set_title(f"{title_prefix}: skekkjudreifing", fontsize=13)
    ax.set_xlabel("Skekkja (spá - raun)")
    save_fig(os.path.join(output_dir, f"{prefix}_skekkjudreifing.png"))

    abs_errors = np.sort(frame["abs_error"].to_numpy())
    cdf = np.arange(1, len(abs_errors) + 1) / len(abs_errors) * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(abs_errors, cdf, color="#4878cf", linewidth=2)
    ax.axvline(5, linestyle="--", color="orange")
    ax.axvline(10, linestyle="--", color="red")
    ax.set_title(f"{title_prefix}: uppsöfnuð skekkjudreifing", fontsize=13)
    ax.set_xlabel("Algild skekkja")
    ax.set_ylabel("Prósenta spáa")
    ax.grid(True, alpha=0.3)
    save_fig(os.path.join(output_dir, f"{prefix}_uppsofnud_skekkja.png"))

    if "dt_to_target" in frame.columns:
        horizon_bins = pd.cut(frame["dt_to_target"], bins=[0, 0.75, 1.25, 2.25, 3.5, 100], labels=["<9 mán", "~1 ár", "~2 ár", "~3 ár", ">3 ár"])
        stats = frame.groupby(horizon_bins, observed=True)["abs_error"].mean()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(stats.index.astype(str), stats.values, color="#6acc64", edgecolor="white")
        ax.set_title(f"{title_prefix}: villa eftir spátíma", fontsize=13)
        ax.set_ylabel("MAE")
        save_fig(os.path.join(output_dir, f"{prefix}_villa_eftir_spatima.png"))

    severity = pd.cut(frame.get("baseline_score", frame["actual"]), bins=4, labels=["Lágt", "Miðlungs lágt", "Miðlungs hátt", "Hátt"])
    sev_stats = frame.groupby(severity, observed=True)["abs_error"].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(sev_stats.index.astype(str), sev_stats.values, color="#d65f5f", edgecolor="white")
    ax.set_title(f"{title_prefix}: villa eftir sjukdomsstigi", fontsize=13)
    ax.set_ylabel("MAE")
    save_fig(os.path.join(output_dir, f"{prefix}_villa_eftir_sjukdomsstigi.png"))

    if "cohort" in frame.columns:
        cohort_stats = frame.groupby("cohort")["abs_error"].mean()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(cohort_stats.index.astype(str), cohort_stats.values, color="#4878cf", edgecolor="white")
        ax.set_title(f"{title_prefix}: villa eftir kohortu", fontsize=13)
        ax.set_ylabel("MAE")
        save_fig(os.path.join(output_dir, f"{prefix}_villa_eftir_kohortu.png"))

    if "actual_change" in frame.columns and "predicted_change" in frame.columns:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(frame["actual_change"], frame["predicted_change"], alpha=0.35, s=18, color="#e87c2b")
        lims = [min(float(frame["actual_change"].min()), float(frame["predicted_change"].min())) - 1, max(float(frame["actual_change"].max()), float(frame["predicted_change"].max())) + 1]
        ax.plot(lims, lims, "k--", linewidth=1.2)
        ax.set_title(f"{title_prefix}: raunveruleg breyting vs spáð breyting", fontsize=13)
        ax.set_xlabel("Raunveruleg breyting")
        ax.set_ylabel("Spáð breyting")
        save_fig(os.path.join(output_dir, f"{prefix}_breyting_raun_vs_spa.png"))

    if "PATNO" in frame.columns:
        patient_mae = frame.groupby("PATNO")["abs_error"].mean()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(patient_mae, bins=30, color="#6acc64", edgecolor="white")
        ax.set_title(f"{title_prefix}: MAE á hvern sjúkling", fontsize=13)
        ax.set_xlabel("Meðal algild villa (UPDRS-III stig)")
        ax.set_ylabel("Fjöldi sjúklinga")
        ax.grid(True, alpha=0.3)
        save_fig(os.path.join(output_dir, f"{prefix}_mae_a_sjukling.png"))

        best = patient_mae.sort_values().head(10)
        worst = patient_mae.sort_values(ascending=False).head(10)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].barh(best.index.astype(str), best.values, color="#6acc64")
        axes[0].set_title("10 bestu sjúklingar")
        axes[0].set_xlabel("Meðal algild villa (UPDRS-III stig)")
        axes[0].set_ylabel("Sjúklingsnúmer")
        axes[1].barh(worst.index.astype(str), worst.values, color="#d65f5f")
        axes[1].set_title("10 verstu sjúklingar")
        axes[1].set_xlabel("Meðal algild villa (UPDRS-III stig)")
        axes[1].set_ylabel("Sjúklingsnúmer")
        fig.suptitle(f"{title_prefix}: bestu og verstu sjúklingar", fontsize=13)
        save_fig(os.path.join(output_dir, f"{prefix}_bestu_og_verstu_sjuklingar.png"))

        if "target_visit_yr" in frame.columns:
            patient_counts = frame.groupby("PATNO").size()
            selected = patient_counts[patient_counts >= 2].sort_values(ascending=False).index[:4].tolist()
            if selected:
                # Sama y-svið í öllum líkönum svo myndirnar séu samanburðarhæfar.
                y_fixed_min = 0
                y_fixed_max = 50
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                axes = axes.flatten()
                for ax, patno in zip(axes, selected):
                    pt = frame[frame["PATNO"] == patno].sort_values("target_visit_yr")
                    ax.plot(pt["target_visit_yr"], pt["actual"], "o-", label="Raunverulegt", color="#4878cf")
                    ax.plot(pt["target_visit_yr"], pt["predicted"], "s--", label="Spáð", color="#e87c2b")
                    ax.set_title(f"Sjúklingur {patno}")
                    ax.set_xlabel("Ár heimsóknar")
                    ax.set_ylabel("UPDRS-III skor")
                    ax.set_ylim(y_fixed_min, y_fixed_max)
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=8)
                for ax in axes[len(selected) :]:
                    ax.axis("off")
                fig.suptitle(f"{title_prefix}: dæmi um sjúklingaferla (PD sjúklingar, hópur 1)", fontsize=13)
                save_fig(os.path.join(output_dir, f"{prefix}_daemi_um_sjuklingaferla.png"))

    # Skiptum spám í hópa og berum meðalspá saman við raunverulegt meðaltal.
    bins = pd.qcut(frame["predicted"], q=min(10, max(3, frame["predicted"].nunique() // 5)), duplicates="drop")
    calib = frame.groupby(bins, observed=True).agg(spa=("predicted", "mean"), raun=("actual", "mean")).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(calib["spa"], calib["raun"], "o-", color="#4878cf", label="Líkanið")
    lims = [min(calib["spa"].min(), calib["raun"].min()) - 1, max(calib["spa"].max(), calib["raun"].max()) + 1]
    ax.plot(lims, lims, "k--", linewidth=1.2, label="Fullkomin spá")
    ax.set_title(
        f"{title_prefix}: kvörðunarrit\n"
        "Meðalspá og meðalraun í hverjum hópi",
        fontsize=11,
    )
    ax.set_xlabel("Meðalspá líkans í hverjum hópi (UPDRS-III stig)")
    ax.set_ylabel("Raunverulegt meðalgildi í sama hópi (UPDRS-III stig)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    save_fig(os.path.join(output_dir, f"{prefix}_kvorad_samanburdur.png"))

    if importance_df is not None and not importance_df.empty:
        value_col = importance_kind
        top = importance_df.copy()
        top["_abs"] = top[value_col].abs()
        top = top.nlargest(50, "_abs").sort_values("_abs")
        top = top.drop(columns=["_abs"])
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(top["feature"], top[value_col].abs(), color="#6acc64", edgecolor="white")
        ax.set_title(f"{title_prefix}: {'stuðlar (algilt gildi)' if value_col == 'coefficient' else 'eiginleikavægi'}", fontsize=13)
        ax.set_xlabel("Algilt vægi (áhrif á spá, óháð stefnu)")
        ax.grid(True, alpha=0.3, axis="x")
        save_fig(os.path.join(output_dir, f"{prefix}_{'studlar' if value_col == 'coefficient' else 'eiginleikavaegi'}.png"))

        group_df = importance_df.copy()
        group_df["group"] = group_df["feature"].map(feature_group)
        vals = group_df.groupby("group")[value_col].apply(lambda s: np.abs(s).sum()).sort_values()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(vals.index, vals.values, color="#4878cf", edgecolor="white")
        ax.set_title(f"{title_prefix}: vægi eiginleikahópa", fontsize=13)
        ax.set_xlabel("Samtals algilt vægi")
        save_fig(os.path.join(output_dir, f"{prefix}_eiginleikahopar.png"))

def plot_selected_patient_predictions(patient_ids=(3822, 3701)):
    model_specs = [
        {
            "label": "Linear Regression",
            "dir": get_output_dir("linear_regression"),
            "prediction_path": os.path.join(get_output_dir("linear_regression"), "linear_regression_predictions.csv"),
            "pred_col": "predicted",
            "color": "#9c9c9c",
        },
        {
            "label": "Random Forest",
            "dir": get_output_dir("random_forest"),
            "prediction_path": os.path.join(get_output_dir("random_forest"), "rf_predictions.csv"),
            "pred_col": "predicted",
            "color": "#6acc64",
        },
        {
            "label": "SVM (SVR)",
            "dir": get_output_dir("svm"),
            "prediction_path": os.path.join(get_output_dir("svm"), "svm_predictions.csv"),
            "pred_col": "predicted",
            "color": "#d65f5f",
        },
        {
            "label": "XGBoost",
            "dir": get_output_dir("xgboost_longitudinal"),
            "prediction_path": os.path.join(get_output_dir("xgboost_longitudinal"), "xgboost_longitudinal_predictions.csv"),
            "pred_col": "predicted",
            "color": "#4878cf",
        },
    ]

    for spec in model_specs:
        pred_df = safe_read_csv(spec["prediction_path"])
        if pred_df is None or spec["pred_col"] not in pred_df.columns:
            continue

        frame = prepare_prediction_frame(pred_df, spec["pred_col"])
        for patient_id in patient_ids:
            pt = frame[frame["PATNO"] == patient_id].sort_values("target_visit_yr")
            if pt.empty:
                continue

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(pt["target_visit_yr"], pt["actual"], "o-", label="Raunverulegt", color="black", linewidth=1.8, markersize=5)
            ax.plot(
                pt["target_visit_yr"],
                pt["predicted"],
                "s--",
                label="Spáð",
                color=spec["color"],
                linewidth=1.8,
                markersize=5,
            )
            ax.set_title(f"{spec['label']} - sjúklingur {patient_id}", fontsize=13, fontweight="bold")
            ax.set_xlabel("Ár frá grunnlínu")
            ax.set_ylabel("UPDRS-III stig")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            save_fig(os.path.join(spec["dir"], f"{slugify_model(spec['label'])}_patient_{patient_id}.png"))


def plot_comparison_patient_grids(patient_ids=(3822, 3701)):
    model_specs = [
        {
            "label": "XGBoost",
            "prediction_path": os.path.join(get_output_dir("xgboost_longitudinal"), "xgboost_longitudinal_predictions.csv"),
            "pred_col": "predicted",
            "color": "#5b8def",
        },
        {
            "label": "Random Forest",
            "prediction_path": os.path.join(get_output_dir("random_forest"), "rf_predictions.csv"),
            "pred_col": "predicted",
            "color": "#63d06b",
        },
        {
            "label": "Línuleg Aðhvarfsgreining",
            "prediction_path": os.path.join(get_output_dir("linear_regression"), "linear_regression_predictions.csv"),
            "pred_col": "predicted",
            "color": "#9a9a9a",
        },
        {
            "label": "SVM (SVR)",
            "prediction_path": os.path.join(get_output_dir("svm"), "svm_predictions.csv"),
            "pred_col": "predicted",
            "color": "#e56b6f",
        },
    ]

    frames = []
    for spec in model_specs:
        pred_df = safe_read_csv(spec["prediction_path"])
        if pred_df is None or spec["pred_col"] not in pred_df.columns:
            continue
        frames.append((spec, prepare_prediction_frame(pred_df, spec["pred_col"])))

    for patient_id in patient_ids:
        patient_frames = []
        y_values = []
        for spec, frame in frames:
            pt = frame[frame["PATNO"] == patient_id].sort_values("target_visit_yr")
            if pt.empty:
                continue
            patient_frames.append((spec, pt))
            y_values.extend(pt["actual"].tolist())
            y_values.extend(pt["predicted"].tolist())

        if not patient_frames:
            continue

        y_min = max(0, int(np.floor(min(y_values) - 1)))
        y_max = int(np.ceil(max(y_values) + 1))

        fig, axes = plt.subplots(2, 2, figsize=(13, 11), sharey=True)
        axes = axes.flatten()
        for ax, (spec, pt) in zip(axes, patient_frames):
            ax.plot(
                pt["target_visit_yr"],
                pt["actual"],
                "o-",
                label="Raunverulegt",
                color="black",
                linewidth=1.8,
                markersize=5,
            )
            ax.plot(
                pt["target_visit_yr"],
                pt["predicted"],
                "s--",
                label="Spáð",
                color=spec["color"],
                linewidth=1.8,
                markersize=5,
            )
            ax.set_title(spec["label"], fontsize=14, fontweight="bold")
            ax.set_xlabel("Ár frá grunnlínu", fontsize=12)
            ax.set_xlim(0.7, 9.3)
            ax.set_xticks(range(1, 10))
            ax.set_ylim(y_min, y_max)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9, loc="upper left")

        for ax in axes[2:]:
            ax.set_xlabel("Ár frá grunnlínu", fontsize=12)
        axes[0].set_ylabel(f"Sjúklingur {patient_id}\nUPDRS-III stig", fontsize=12)
        axes[2].set_ylabel(f"Sjúklingur {patient_id}\nUPDRS-III stig", fontsize=12)
        fig.suptitle(
            f"Spáð vs raunverulegt UPDRS-III - sjúklingur {patient_id}, fjögur líkön",
            fontsize=18,
            fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        save_fig(os.path.join(COMPARISON_DIR, f"comparison_patient_{patient_id}_models.png"))


def plot_mean_updrs_by_year_from_predictions(prediction_path, output_dir, output_filename):
    pred_df = safe_read_csv(prediction_path)
    if pred_df is None or "traj_n_visits" not in pred_df.columns or "predicted" not in pred_df.columns:
        return

    plot_df = pred_df.copy()
    actual_col = "actual" if "actual" in plot_df.columns else "target_updrs3"
    plot_df["actual"] = plot_df[actual_col]
    plot_df["years_from_first_prediction"] = plot_df["traj_n_visits"] - 2

    grouped = (
        plot_df.groupby("years_from_first_prediction", as_index=False)[["actual", "predicted"]]
        .mean()
        .sort_values("years_from_first_prediction")
    )
    if grouped.empty:
        return

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
    save_fig(os.path.join(output_dir, output_filename))


def generate_model_plots():
    model_specs = [
        {
            "label": "Linear Regression",
            "dir": get_output_dir("linear_regression"),
            "prediction_path": os.path.join(get_output_dir("linear_regression"), "linear_regression_predictions.csv"),
            "pred_col": "predicted",
            "importance_path": os.path.join(get_output_dir("linear_regression"), "linear_regression_coefficients.csv"),
            "importance_kind": "coefficient",
        },
        {
            "label": "Random Forest",
            "dir": get_output_dir("random_forest"),
            "prediction_path": os.path.join(get_output_dir("random_forest"), "rf_predictions.csv"),
            "pred_col": "predicted",
            "importance_path": os.path.join(get_output_dir("random_forest"), "rf_feature_importance.csv"),
            "importance_kind": "importance",
        },
        {
            "label": "SVR",
            "dir": get_output_dir("svm"),
            "prediction_path": os.path.join(get_output_dir("svm"), "svm_predictions.csv"),
            "pred_col": "predicted",
            "importance_path": os.path.join(get_output_dir("svm"), "svm_feature_importance.csv"),
            "importance_kind": "importance",
        },
        {
            "label": "XGBoost",
            "dir": get_output_dir("xgboost_longitudinal"),
            "prediction_path": os.path.join(get_output_dir("xgboost_longitudinal"), "xgboost_longitudinal_predictions.csv"),
            "pred_col": "predicted",
            "importance_path": os.path.join(get_output_dir("xgboost_longitudinal"), "xgb_long_feature_importance.csv"),
            "importance_kind": "importance",
        },
    ]

    for spec in model_specs:
        pred_df = safe_read_csv(spec["prediction_path"])
        if pred_df is None or spec["pred_col"] not in pred_df.columns:
            continue
        frame = prepare_prediction_frame(pred_df, spec["pred_col"])
        importance_df = safe_read_csv(spec["importance_path"]) if spec.get("importance_path") else None
        plot_model_frame(
            frame,
            spec["dir"],
            slugify_model(spec["label"]),
            spec["label"],
            importance_df=importance_df,
            importance_kind=spec.get("importance_kind", "importance"),
        )

    clinical_path = os.path.join(get_output_dir("clinical_only_delta"), "clinical_only_delta_predictions.csv")
    clinical_df = safe_read_csv(clinical_path)
    if clinical_df is not None:
        importance_rf = safe_read_csv(os.path.join(get_output_dir("clinical_only_delta"), "clinical_only_delta_rf_feature_importance.csv"))
        importance_xgb = safe_read_csv(os.path.join(get_output_dir("clinical_only_delta"), "clinical_only_delta_xgb_feature_importance.csv"))
        for pred_col, label, imp_df in [
            ("predicted_baseline", "Clinical-only Baseline", None),
            ("predicted_linear", "Clinical-only Linear Regression", None),
            ("predicted_rf", "Clinical-only Random Forest", importance_rf),
            ("predicted_xgb", "Clinical-only XGBoost", importance_xgb),
            ("predicted_svr", "Clinical-only SVR", None),
        ]:
            if pred_col not in clinical_df.columns:
                continue
            frame = prepare_prediction_frame(clinical_df, pred_col)
            plot_model_frame(
                frame,
                get_output_dir("clinical_only_delta"),
                pred_col.replace("predicted_", ""),
                label,
                importance_df=imp_df,
                importance_kind="importance",
            )


def main():
    source_tables = load_source_tables()
    clinical_table = prepare_clinical_table(source_tables)
    longitudinal_pairs = add_longitudinal_features(build_longitudinal_pairs(clinical_table))
    clinical_only_pairs = add_clinical_only_features(build_clinical_only_pairs(clinical_table))

    plot_workflow_diagram()
    plot_eda(clinical_table, longitudinal_pairs, source_tables)
    plot_heldout_model_comparison()
    plot_clinical_vs_full()
    generate_model_plots()

    summary = {
        "eda_rows": len(clinical_table),
        "longitudinal_pairs": len(longitudinal_pairs),
        "clinical_only_pairs": len(clinical_only_pairs),
        "longitudinal_features": len(FEATURE_COLS),
        "clinical_only_features": len(CLINICAL_ONLY_FEATURE_COLS),
    }
    pd.DataFrame([summary]).to_csv(os.path.join(MISC_DIR, "plot_yfirlit.csv"), index=False)
    print("\nBúið að búa til sameiginlegt myndasafn fyrir skýrslu og kynningu.")


if __name__ == "__main__":
    main()
