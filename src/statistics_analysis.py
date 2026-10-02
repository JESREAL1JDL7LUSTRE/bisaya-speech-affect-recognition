"""Missing-aware exploratory statistics grounded in self-reported affect ratings."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from scipy import stats


def benjamini_hochberg(p_values) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    result = np.full(values.shape, np.nan)
    valid = np.isfinite(values)
    if not valid.any():
        return result
    p = values[valid]
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0, 1)
    result[valid] = restored
    return result


def compute_univariate_stats(series: pd.Series) -> dict:
    clean = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if clean.size == 0:
        return {key: np.nan for key in ["N", "Missing", "Mean", "Std", "Median", "IQR", "Min", "Max", "Skewness", "Kurtosis"]}
    q25, q75 = np.percentile(clean, [25, 75])
    std = float(np.std(clean, ddof=1)) if clean.size > 1 else 0.0
    return {
        "N": int(clean.size), "Missing": int(series.size - clean.size),
        "Mean": round(float(np.mean(clean)), 4), "Std": round(std, 4),
        "Median": round(float(np.median(clean)), 4), "IQR": round(float(q75 - q25), 4),
        "Min": round(float(np.min(clean)), 4), "Max": round(float(np.max(clean)), 4),
        "Skewness": round(float(stats.skew(clean)), 4) if clean.size > 2 and std > 1e-12 else 0.0,
        "Kurtosis": round(float(stats.kurtosis(clean)), 4) if clean.size > 3 and std > 1e-12 else 0.0,
    }


def _pairwise(df: pd.DataFrame, a: str, b: str):
    pair = df[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(pair) < 3 or pair[a].nunique() < 2 or pair[b].nunique() < 2:
        return len(pair), np.nan, np.nan, np.nan, np.nan
    pearson = stats.pearsonr(pair[a], pair[b])
    spearman = stats.spearmanr(pair[a], pair[b])
    return len(pair), pearson.statistic, pearson.pvalue, spearman.statistic, spearman.pvalue


def _save_summary(df: pd.DataFrame, columns: list[str], path: Path) -> pd.DataFrame:
    records = []
    for column in columns:
        if column in df:
            record = compute_univariate_stats(df[column])
            record["Feature"] = column
            records.append(record)
    result = pd.DataFrame(records)
    result = result[["Feature", "N", "Missing", "Mean", "Std", "Median", "IQR", "Min", "Max", "Skewness", "Kurtosis"]]
    result.to_csv(path, index=False)
    return result


def _session_comparison(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        if feature not in df:
            continue
        am = pd.to_numeric(df.loc[df.session == "AM", feature], errors="coerce").dropna().to_numpy()
        pm = pd.to_numeric(df.loc[df.session == "PM", feature], errors="coerce").dropna().to_numpy()
        if len(am) < 2 or len(pm) < 2:
            continue
        welch = stats.ttest_ind(am, pm, equal_var=False)
        mwu = stats.mannwhitneyu(am, pm, alternative="two-sided")
        pooled = np.sqrt(((len(am) - 1) * np.var(am, ddof=1) + (len(pm) - 1) * np.var(pm, ddof=1)) / (len(am) + len(pm) - 2))
        rows.append({
            "Feature": feature, "AM_N": len(am), "PM_N": len(pm),
            "AM_Mean": np.mean(am), "AM_Std": np.std(am, ddof=1),
            "PM_Mean": np.mean(pm), "PM_Std": np.std(pm, ddof=1),
            "Welch_T": welch.statistic, "Welch_p": welch.pvalue,
            "MannWhitney_U": mwu.statistic, "MannWhitney_p": mwu.pvalue,
            "Cohens_d": (np.mean(am) - np.mean(pm)) / pooled if pooled > 1e-12 else 0.0,
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["Welch_q_BH"] = benjamini_hochberg(result.Welch_p)
        result["Welch_FDR_05"] = result.Welch_q_BH < 0.05
        numeric = result.select_dtypes(include=np.number).columns
        result[numeric] = result[numeric].round(6)
    return result


def _correlations(df: pd.DataFrame, left: list[str], right: list[str], left_name: str, right_name: str) -> pd.DataFrame:
    rows = []
    for a in left:
        for b in right:
            if a not in df or b not in df:
                continue
            n, r, p, rho, sp = _pairwise(df, a, b)
            rows.append({left_name: a, right_name: b, "N": n, "Pearson_r": r, "Pearson_p": p, "Spearman_rho": rho, "Spearman_p": sp})
    result = pd.DataFrame(rows)
    if not result.empty:
        result["Pearson_q_BH"] = benjamini_hochberg(result.Pearson_p)
        result["Pearson_FDR_05"] = result.Pearson_q_BH < 0.05
        numeric = result.select_dtypes(include=np.number).columns
        result[numeric] = result[numeric].round(6)
        result = result.sort_values(["Pearson_q_BH", "Pearson_p"], na_position="last")
    return result


def run_descriptive_statistics(
    dataset_csv: str = "datasets/multimodal/cleaned_multimodal_dataset.csv",
    output_dir: str = "outputs/descriptive_statistics",
):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(dataset_csv)

    audio_features = [
        "audio_duration_sec", "audio_trimmed_dur_sec", "audio_f0_mean_hz", "audio_f0_std_hz",
        "audio_f0_range_hz", "audio_voiced_ratio", "audio_period_variation_proxy",
        "audio_rms_mean", "audio_rms_std", "audio_rms_frame_variation_proxy", "audio_zcr_mean",
        "audio_spec_centroid_mean", "audio_spec_bandwidth_mean", "audio_spec_rolloff85_mean",
        "audio_spec_flatness_mean", "audio_spec_contrast_mean", "audio_chroma_mean",
    ] + [f"audio_mfcc_{i}_mean" for i in range(1, 14)]
    facial_features = [
        "face_smile_mean", "face_smile_max", "face_frown_mean", "face_frown_max",
        "face_brow_lowerer_mean", "face_brow_inner_raiser_mean", "face_jaw_open_mean",
        "face_eye_squint_mean", "face_disgust_proxy_mean", "face_expressiveness_proxy",
        "face_ear_mean", "face_ear_min", "face_mar_mean", "face_mar_max",
        "face_pitch_mean_deg", "face_yaw_mean_deg", "face_roll_mean_deg", "face_detection_rate",
    ]
    audio_stats = _save_summary(df, audio_features, output / "summary_statistics_audio.csv")
    facial_stats = _save_summary(df, facial_features, output / "summary_statistics_facial.csv")

    session_features = ["valence_score", "arousal_score"] + audio_features[:15] + facial_features[:16]
    session = _session_comparison(df, session_features)
    session.to_csv(output / "session_am_pm_comparison.csv", index=False)

    affect = df.groupby(["valence_group", "arousal_group", "affect_category"], dropna=False).agg(
        Count=("participant_code", "size"),
        Mean_Valence=("valence_score", "mean"), Mean_Arousal=("arousal_score", "mean"),
        Pitch_Available=("audio_pitch_detected", "sum"), Mean_F0_Hz=("audio_f0_mean_hz", "mean"),
        Mean_RMS=("audio_rms_mean", "mean"), Mean_Smile_Proxy=("face_smile_mean", "mean"),
    ).reset_index()
    affect["Percentage"] = affect.Count / len(df) * 100
    affect.round(4).to_csv(output / "affect_taxonomy_summary.csv", index=False)

    kapoy_columns = [
        "participant_code", "session", "class_activity", "self_reported_feeling",
        "self_reported_feeling_source", "valence_score", "arousal_score", "affect_category",
        "audio_pitch_detected", "audio_duration_sec", "audio_f0_mean_hz", "audio_rms_mean",
        "audio_spec_centroid_mean", "face_smile_mean", "face_frown_mean", "face_brow_lowerer_mean",
        "face_ear_mean", "face_mar_mean",
    ]
    kapoy = df.loc[df.spoken_word.eq("Kapoy"), [c for c in kapoy_columns if c in df]].sort_values(["valence_score", "arousal_score", "participant_code"])
    kapoy.to_csv(output / "lexically_controlled_kapoy.csv", index=False)

    context_metrics = ["valence_score", "arousal_score", "audio_f0_mean_hz", "audio_rms_mean", "face_smile_mean"]
    activity = df.groupby("class_activity").agg(Count=("participant_code", "size"), **{f"Mean_{c}": (c, "mean") for c in context_metrics}).reset_index()
    activity.round(4).to_csv(output / "class_activity_summary.csv", index=False)
    words = df.groupby("spoken_word").agg(Count=("participant_code", "size"), Mean_Valence=("valence_score", "mean"), Mean_Arousal=("arousal_score", "mean"), Pitch_Available=("audio_pitch_detected", "sum"), Mean_F0_Hz=("audio_f0_mean_hz", "mean")).reset_index()
    words.round(4).to_csv(output / "spoken_word_affect_summary.csv", index=False)
    years = df.groupby("year_level").agg(Count=("participant_code", "size"), Mean_Valence=("valence_score", "mean"), Mean_Arousal=("arousal_score", "mean")).reset_index()
    years.round(4).to_csv(output / "year_level_affect_summary.csv", index=False)

    audio_corr = [c for c in ["audio_f0_mean_hz", "audio_f0_range_hz", "audio_rms_mean", "audio_zcr_mean", "audio_spec_centroid_mean", "audio_spec_rolloff85_mean", "audio_duration_sec", "audio_period_variation_proxy", "audio_rms_frame_variation_proxy"] if c in df]
    face_corr = [c for c in ["face_smile_mean", "face_frown_mean", "face_brow_lowerer_mean", "face_brow_inner_raiser_mean", "face_jaw_open_mean", "face_ear_mean", "face_mar_mean", "face_expressiveness_proxy", "face_pitch_mean_deg", "face_yaw_mean_deg"] if c in df]
    cross = _correlations(df, audio_corr, face_corr, "Audio_Feature", "Facial_Feature")
    cross.to_csv(output / "audio_facial_correlations.csv", index=False)

    ground_rows = []
    for feature in audio_corr + face_corr:
        for outcome in ["valence_score", "arousal_score"]:
            n, r, p, rho, sp = _pairwise(df, feature, outcome)
            ground_rows.append({"Feature": feature, "Outcome": outcome, "N": n, "Pearson_r": r, "Pearson_p": p, "Spearman_rho": rho, "Spearman_p": sp})
    ground = pd.DataFrame(ground_rows)
    ground["Pearson_q_BH"] = np.nan
    for outcome in ground.Outcome.unique():
        mask = ground.Outcome.eq(outcome)
        ground.loc[mask, "Pearson_q_BH"] = benjamini_hochberg(ground.loc[mask, "Pearson_p"])
    ground["Pearson_FDR_05"] = ground.Pearson_q_BH < 0.05
    ground = ground.round(6).sort_values(["Outcome", "Pearson_q_BH", "Pearson_p"])
    ground.to_csv(output / "ground_truth_affect_correlations.csv", index=False)

    report = generate_descriptive_report(df, audio_stats, facial_stats, session, affect, kapoy, cross, ground)
    (output / "descriptive_statistics_report.md").write_text(report, encoding="utf-8")
    return {
        "audio_stats": audio_stats, "facial_stats": facial_stats, "session_comp": session,
        "affect_summary": affect, "kapoy_analysis": kapoy, "correlations": cross,
        "ground_truth_correlations": ground,
    }


def _fmt_p(value) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.2e}" if value < 0.0001 else f"{value:.4f}"


def _markdown_table(df: pd.DataFrame, index: bool = False) -> str:
    try:
        return df.to_markdown(index=index)
    except Exception:
        cols = list(df.columns)
        header = "| " + " | ".join(str(c) for c in cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        body = ["| " + " | ".join(str(val) for val in row) + " |" for row in df.itertuples(index=False)]
        return "\n".join([header, sep] + body)


def generate_descriptive_report(df, audio_stats, facial_stats, session, affect, kapoy, cross, ground) -> str:
    valence_counts = df.valence_score.value_counts().sort_index().to_dict()
    arousal_counts = df.arousal_score.value_counts().sort_index().to_dict()
    pitch_n = int(df.audio_pitch_detected.sum())
    cross_fdr = int(cross.Pearson_FDR_05.sum()) if not cross.empty else 0
    ground_fdr = int(ground.Pearson_FDR_05.sum()) if not ground.empty else 0
    session_fdr = int(session.Welch_FDR_05.sum()) if not session.empty else 0
    activity_map = df.groupby("session").agg(Subjects=("subject", lambda x: ", ".join(sorted(set(x)))), Activities=("class_activity", lambda x: ", ".join(sorted(set(x)))), N=("participant_code", "size")).reset_index()
    lines = [
        "# Descriptive statistics and exploratory analysis",
        "",
        "## Scope and reference outcome",
        "",
        f"This analysis contains {len(df)} fourth-year participant observations. Affect is represented by independently recorded valence and arousal ratings. Spoken-word categories are lexical context and are not treated as participant emotion.",
        "",
        f"Valence counts (scores 1–5): {valence_counts}. Arousal counts (scores 1–5): {arousal_counts}.",
        f"Pitch was detected for {pitch_n}/{len(df)} recordings; pitch summaries exclude unavailable observations rather than treating them as 0 Hz.",
        "",
        "## Educational context",
        "",
        _markdown_table(activity_map, index=False),
        "",
        "Session, subject and activity are completely confounded in this group: PM contains Foreign Language / Lecture-Review, while AM contains Graph Theory / Prelim Exam and Numerical Analysis / Quiz. Session comparisons are descriptive and cannot isolate a time-of-day effect. Year-level differences cannot be evaluated because every observation is fourth year.",
        "",
        "## Session comparisons",
        "",
        f"Welch tests were treated as the primary session tests and adjusted as one family using Benjamini–Hochberg FDR. {session_fdr} of {len(session)} comparisons have q < .05. Mann–Whitney results are secondary sensitivity checks.",
        "",
        _markdown_table(session[["Feature", "AM_N", "PM_N", "AM_Mean", "PM_Mean", "Welch_p", "Welch_q_BH", "Cohens_d"]], index=False) if not session.empty else "No session comparisons were estimable.",
        "",
        "## Rating-derived affect groups",
        "",
        "Valence groups use scores 1–2 = Negative, 3 = Neutral, and 4–5 = Positive. Arousal groups use 1–2 = Low, 3 = Moderate, and 4–5 = High. These are transparent analytical bins, not clinical emotion diagnoses.",
        "",
        _markdown_table(affect.round(3), index=False),
        "",
        "## Same-word analysis",
        "",
        f"There are {len(kapoy)} Kapoy recordings. Their variation can describe within-word heterogeneity, but variation alone does not prove that features predict affect or control for speaker differences.",
        "",
        _markdown_table(kapoy[[c for c in ["participant_code", "valence_score", "arousal_score", "audio_pitch_detected", "audio_f0_mean_hz", "audio_rms_mean", "face_smile_mean"] if c in kapoy]], index=False),
        "",
        "## Exploratory associations",
        "",
        f"Among {len(cross)} audio-facial Pearson tests, {cross_fdr} survive BH FDR at .05. Among {len(ground)} feature/outcome tests, {ground_fdr} survive outcome-specific BH FDR at .05. Lack of an adjusted association does not prove absence of signal in this small pilot sample.",
        "",
        "The correlation analysis is participant-level. It should not be described as temporal synchronization because frame-level audio/video dynamics were not correlated.",
        "",
        "## Limitations",
        "",
        "The sample is small and unbalanced across sessions. Context variables are confounded. Missing pitch is substantial. Sixteen feeling-text entries are marked as unverified word fills pending provenance confirmation. Four recordings were intentionally trimmed to avoid background music; speech completeness and overlap status remain to be confirmed. MediaPipe blendshapes and derived composites are model-based expression proxies, not validated FACS coding or objective emotion measurements.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    run_descriptive_statistics()
