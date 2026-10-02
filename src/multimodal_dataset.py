"""Validated one-to-one multimodal merge, column roles, and quality reporting."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd


OUTCOME_COLUMNS = {
    "self_reported_feeling", "self_reported_feeling_missing", "self_reported_feeling_source",
    "valence_score", "valence_label", "valence_group", "arousal_score", "arousal_label",
    "arousal_group", "affect_category",
}
IDENTIFIER_COLUMNS = {"participant_code", "audio_filename", "video_filename", "survey_audio_filename"}
CONTEXT_COLUMNS = {
    "year_level", "year_code", "class_activity", "subject", "session", "session_name",
    "spoken_word", "lexical_category", "word_english", "date_collected", "consent_obtained",
}
QUALITY_TOKENS = (
    "detected", "detection", "quality", "boundary", "frame_length", "total_frames",
    "decoded_frames", "reported_frames", "sampling_rate", "delta_available", "delta_width",
    "intentional_bgm", "background_music", "speech_complete", "metadata_word_matches",
    "coordinate_system", "running_mode", "video_width", "video_height", "video_fps",
    "pitch_total_frames", "voiced_frames",
)


def column_role(column: str, series: pd.Series | None = None) -> str:
    if column in IDENTIFIER_COLUMNS:
        return "identifier"
    if column in OUTCOME_COLUMNS:
        return "reference_outcome"
    if column in CONTEXT_COLUMNS:
        return "context"
    if any(token in column for token in QUALITY_TOKENS):
        return "quality_control"
    if column.startswith("audio_"):
        return "acoustic_predictor"
    if column.startswith("face_"):
        return "facial_predictor"
    if column.startswith("video_"):
        return "quality_control"
    return "other"


def predictor_columns(df: pd.DataFrame, modality: str = "multimodal") -> list[str]:
    roles = {"acoustic_predictor", "facial_predictor"}
    if modality == "audio":
        roles = {"acoustic_predictor"}
    elif modality == "facial":
        roles = {"facial_predictor"}
    return [c for c in df.columns if column_role(c, df[c]) in roles and pd.api.types.is_numeric_dtype(df[c])]


def _validate_unique(df: pd.DataFrame, name: str):
    if "participant_code" not in df:
        raise ValueError(f"{name} is missing participant_code")
    if df["participant_code"].isna().any() or df["participant_code"].duplicated().any():
        duplicates = df.loc[df["participant_code"].duplicated(keep=False), "participant_code"].tolist()
        raise ValueError(f"{name} participant_code must be unique and nonmissing; duplicates={duplicates}")


def _metadata_columns(df_audio: pd.DataFrame, df_facial: pd.DataFrame) -> list[str]:
    shared = set(df_audio.columns) & set(df_facial.columns)
    return sorted(c for c in shared if not c.startswith(("audio_", "face_", "video_")))


def _unit_for(column: str, series: pd.Series) -> str:
    if column in {"valence_score", "arousal_score"}:
        return "ordinal score 1-5"
    if "f0_" in column and "hz" in column or "centroid" in column or "bandwidth" in column or "rolloff" in column:
        return "Hz"
    if column.endswith("_sec") or "duration_sec" in column:
        return "seconds"
    if column.endswith("_deg"):
        return "degrees"
    if column.endswith("_px"):
        return "pixels"
    if "rate" in column or "ratio" in column or "fraction" in column or column.startswith("face_"):
        return "dimensionless"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    return "coefficient or native unit"


def _write_quality_report(df: pd.DataFrame, csv_path: str, md_path: str):
    rows = []
    for _, row in df.iterrows():
        warnings = []
        if not bool(row.get("audio_pitch_detected", False)):
            warnings.append("pitch unavailable")
        if bool(row.get("audio_boundary_review_flag", False)):
            warnings.append("high-energy audio boundary")
        if str(row.get("intentional_bgm_boundary_trim", "No")).lower() == "yes":
            warnings.append("intentional trim to avoid background music")
        if str(row.get("background_music_overlap", "Not flagged")).lower() == "unknown":
            warnings.append("background-music overlap unverified")
        if str(row.get("speech_complete_status", "Not flagged")).lower() == "unknown":
            warnings.append("speech completeness unverified")
        if str(row.get("self_reported_feeling_source", "")) == "Unverified word fill":
            warnings.append("feeling-text provenance unverified")
        if not bool(row.get("face_detection_quality_pass", False)):
            warnings.append("face detection below 80%")
        rows.append({
            "participant_code": row["participant_code"],
            "consent_obtained": bool(row["consent_obtained"]),
            "metadata_word_matches_filename": bool(row["metadata_word_matches_filename"]),
            "audio_pitch_detected": bool(row["audio_pitch_detected"]),
            "face_detection_rate": row["face_detection_rate"],
            "recording_valid_for_non_pitch_eda": bool(row["consent_obtained"] and row["metadata_word_matches_filename"] and row["face_detection_quality_pass"]),
            "quality_warnings": "; ".join(warnings),
        })
    quality = pd.DataFrame(rows)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    quality.to_csv(csv_path, index=False)
    summary = [
        "# Data quality report",
        "",
        f"- Observations: {len(quality)}",
        f"- Valid for non-pitch EDA: {int(quality['recording_valid_for_non_pitch_eda'].sum())}/{len(quality)}",
        f"- Pitch available: {int(quality['audio_pitch_detected'].sum())}/{len(quality)}",
        f"- High-energy boundary flags: {int(df['audio_boundary_review_flag'].sum())}",
        f"- Intentional background-music boundary trims: {int(df['intentional_bgm_boundary_trim'].astype(str).str.lower().eq('yes').sum())}",
        f"- Unverified feeling-text provenance: {int(df['self_reported_feeling_source'].eq('Unverified word fill').sum())}",
        "",
        "Pitch-unavailable observations remain usable for analyses that do not require pitch. Missing pitch is preserved rather than replaced with 0 Hz.",
        "Intentional background-music boundary trims require manual confirmation of speech completeness and overlap status.",
    ]
    Path(md_path).write_text("\n".join(summary) + "\n", encoding="utf-8")


def build_cleaned_multimodal_dataset(
    audio_csv: str = "datasets/audio/audio_feature_dataset.csv",
    facial_csv: str = "datasets/facial/facial_feature_dataset.csv",
    output_raw_csv: str = "datasets/multimodal/cleaned_multimodal_dataset.csv",
    output_scaled_csv: str = "datasets/multimodal/cleaned_multimodal_dataset_scaled.csv",
    output_dictionary_csv: str = "datasets/multimodal/multimodal_data_dictionary.csv",
    output_scaler_csv: str = "datasets/multimodal/descriptive_scaling_parameters.csv",
    quality_csv: str = "outputs/data_quality/data_quality_report.csv",
    quality_md: str = "outputs/data_quality/data_quality_report.md",
):
    df_audio = pd.read_csv(audio_csv)
    df_facial = pd.read_csv(facial_csv)
    _validate_unique(df_audio, "audio dataset")
    _validate_unique(df_facial, "facial dataset")
    audio_codes, facial_codes = set(df_audio.participant_code), set(df_facial.participant_code)
    if audio_codes != facial_codes:
        raise ValueError(f"Audio/facial participant mismatch: audio_only={sorted(audio_codes-facial_codes)}, facial_only={sorted(facial_codes-audio_codes)}")

    shared_metadata = _metadata_columns(df_audio, df_facial)
    a_meta = df_audio.set_index("participant_code")[shared_metadata].sort_index()
    f_meta = df_facial.set_index("participant_code")[shared_metadata].sort_index()
    mismatches = []
    for column in shared_metadata:
        left = a_meta[column].fillna("<MISSING>").astype(str)
        right = f_meta[column].fillna("<MISSING>").astype(str)
        for code in left.index[left != right]:
            mismatches.append((code, column, left.loc[code], right.loc[code]))
    if mismatches:
        raise ValueError(f"Audio/facial metadata mismatch: {mismatches[:10]}")

    facial_only = [c for c in df_facial.columns if c == "participant_code" or c not in shared_metadata]
    df = pd.merge(df_audio, df_facial[facial_only], on="participant_code", how="inner", validate="one_to_one")
    audio_stems = df.audio_filename.map(lambda value: Path(value).stem.casefold())
    video_stems = df.video_filename.map(lambda value: Path(value).stem.casefold())
    if not audio_stems.equals(video_stems):
        raise ValueError("Audio/video recording stems do not match one-to-one")

    numeric = df.select_dtypes(include=np.number)
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError("Infinite numeric values detected")
    allowed_missing = set(c for c in df.columns if c.startswith("audio_f0_") or c == "audio_period_variation_proxy") | {"self_reported_feeling"}
    unexpected_missing = {c: int(n) for c, n in df.isna().sum().items() if n and c not in allowed_missing}
    if unexpected_missing:
        raise ValueError(f"Unexpected missing values: {unexpected_missing}")

    Path(output_raw_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_raw_csv, index=False)

    # EDA-only scaling: targets/context/quality stay unchanged. Model evaluation must fit transforms inside training folds.
    scaled = df.copy()
    parameters = []
    for column in predictor_columns(df):
        values = df[column].astype(float)
        mean, std = float(values.mean(skipna=True)), float(values.std(skipna=True, ddof=0))
        parameters.append({"column": column, "mean": mean, "population_std": std, "scope": "full dataset; descriptive EDA only"})
        scaled[column] = ((values - mean) / std).round(4) if std > 1e-12 else 0.0
    scaled.to_csv(output_scaled_csv, index=False)
    pd.DataFrame(parameters).to_csv(output_scaler_csv, index=False)

    dictionary = []
    for column in df.columns:
        role = column_role(column, df[column])
        modality = "Audio" if column.startswith("audio_") else "Facial" if column.startswith("face_") else "Video" if column.startswith("video_") else "Metadata"
        dictionary.append({
            "Column_Name": column,
            "Role": role,
            "Modality": modality,
            "Data_Type": str(df[column].dtype),
            "Unit": _unit_for(column, df[column]),
            "Missing_Count": int(df[column].isna().sum()),
            "Description": f"{role.replace('_', ' ').title()}: {column}",
        })
    df_dictionary = pd.DataFrame(dictionary)
    df_dictionary.to_csv(output_dictionary_csv, index=False)
    _write_quality_report(df, quality_csv, quality_md)
    print(f"Cleaned multimodal dataset: {df.shape}; predictors={len(predictor_columns(df))}")
    print("Scaled dataset is for descriptive EDA only; fit preprocessing inside training folds for ML.")
    return df, scaled, df_dictionary


if __name__ == "__main__":
    build_cleaned_multimodal_dataset()
