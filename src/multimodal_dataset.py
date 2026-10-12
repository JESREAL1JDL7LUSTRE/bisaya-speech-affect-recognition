"""Validated one-to-one multimodal merge, column roles, and quality reporting."""

from __future__ import annotations

# --- Standard Library Imports ---
import os                     # File checking and environment manipulation
import sys                    # Python runtime path configuration
from pathlib import Path       # Filesystem path abstraction

# Add project root directory to sys.path so modules can import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Scientific & Data Analysis Libraries ---
import numpy as np            # Numerical vector checking (infinities, NaNs)
import pandas as pd           # Tabular dataset manipulation and merging

# --- Explicit Column Role Categorization Sets ---
# DEFENSE POINT: Strict partition between predictor biomarkers and reference ground truths.
# Reference outcome targets (NEVER fed into models as features to prevent target leakage):
OUTCOME_COLUMNS = {
    "self_reported_feeling", "self_reported_feeling_missing", "self_reported_feeling_source",
    "valence_score", "valence_label", "valence_group", "arousal_score", "arousal_label",
    "arousal_group", "affect_category",
}

# Unique record identifiers:
IDENTIFIER_COLUMNS = {"participant_code", "audio_filename", "video_filename", "survey_audio_filename"}

# Contextual and demographic variables:
CONTEXT_COLUMNS = {
    "year_level", "year_code", "class_activity", "subject", "session", "session_name",
    "spoken_word", "lexical_category", "word_english", "date_collected", "consent_obtained",
}

# Tokens used to identify quality control and recording integrity flags:
QUALITY_TOKENS = (
    "detected", "detection", "quality", "boundary", "frame_length", "total_frames",
    "decoded_frames", "reported_frames", "sampling_rate", "delta_available", "delta_width",
    "intentional_bgm", "background_music", "speech_complete", "metadata_word_matches",
    "coordinate_system", "running_mode", "video_width", "video_height", "video_fps",
    "pitch_total_frames", "voiced_frames",
)


def column_role(column: str, series: pd.Series | None = None) -> str:
    """
    Classifies any column in the multimodal dataset into a standardized schema role:
    - 'identifier': Participant IDs and file paths.
    - 'reference_outcome': Ground-truth affect targets (Valence, Arousal).
    - 'context': Demographics, lexical word, class activity, session.
    - 'quality_control': Tracking pass flags, SNR boundaries, BGM trim flags.
    - 'acoustic_predictor': 108 acoustic and prosodic features from audio.
    - 'facial_predictor': 225 FACS blendshape and geometric features from video.
    """
    # 1. Primary keys and filenames
    if column in IDENTIFIER_COLUMNS:
        return "identifier"
    # 2. Ground-truth target variables
    if column in OUTCOME_COLUMNS:
        return "reference_outcome"
    # 3. Contextual and experimental metadata
    if column in CONTEXT_COLUMNS:
        return "context"
    # 4. Data audit and quality assurance flags
    if any(token in column for token in QUALITY_TOKENS):
        return "quality_control"
    # 5. Acoustic audio predictors (e.g., audio_f0_mean_hz, audio_rms_mean, audio_mfcc_1_mean)
    if column.startswith("audio_"):
        return "acoustic_predictor"
    # 6. Facial visual predictors (e.g., face_smile_mean, face_ear_mean, face_jaw_open_mean)
    if column.startswith("face_"):
        return "facial_predictor"
    # 7. Video capture metadata
    if column.startswith("video_"):
        return "quality_control"
    return "other"


def predictor_columns(df: pd.DataFrame, modality: str = "multimodal") -> list[str]:
    """
    Returns the list of valid numerical feature predictor column names for a given modality:
    - 'audio': 108 acoustic predictors only.
    - 'facial': 225 facial blendshape and geometric predictors only.
    - 'multimodal': Combined audio + facial predictors (333 total).
    Guarantees non-numeric columns and targets are excluded.
    """
    # Define active roles based on requested modality
    roles = {"acoustic_predictor", "facial_predictor"}
    if modality == "audio":
        roles = {"acoustic_predictor"}
    elif modality == "facial":
        roles = {"facial_predictor"}
        
    # Filter DataFrame columns matching role and numeric dtype
    return [
        c for c in df.columns 
        if column_role(c, df[c]) in roles and pd.api.types.is_numeric_dtype(df[c])
    ]


def _validate_unique(df: pd.DataFrame, name: str):
    """
    Enforces primary key integrity on participant_code:
    Ensures column exists, contains no NaNs, and has zero duplicate codes.
    """
    # Verify participant_code column is present
    if "participant_code" not in df:
        raise ValueError(f"{name} is missing participant_code")
    # Verify no missing or duplicated participant codes
    if df["participant_code"].isna().any() or df["participant_code"].duplicated().any():
        duplicates = df.loc[df["participant_code"].duplicated(keep=False), "participant_code"].tolist()
        raise ValueError(f"{name} participant_code must be unique and nonmissing; duplicates={duplicates}")


def _metadata_columns(df_audio: pd.DataFrame, df_facial: pd.DataFrame) -> list[str]:
    """
    Identifies shared non-feature metadata columns present in both audio and facial datasets
    (e.g., session, spoken_word, valence_group, consent_obtained).
    """
    # Set intersection of columns excluding primary key
    shared = (set(df_audio.columns) & set(df_facial.columns)) - {"participant_code"}
    # Filter out predictor and video quality columns
    return sorted(c for c in shared if not c.startswith(("audio_", "face_", "video_")))


def _unit_for(column: str, series: pd.Series) -> str:
    """
    Assigns the scientific physical or mathematical measurement unit for each variable:
    Used to generate the comprehensive Multimodal Data Dictionary.
    """
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
    """
    Generates an auditable Data Quality Report documenting:
    - Pitch availability across unvoiced vs voiced utterances
    - High-energy audio boundary cutoffs
    - Intentional background music (BGM) trimming
    - Face tracking success rates (>=80% threshold)
    Exports both tabular CSV and markdown summary documents.
    """
    rows = []
    # Evaluate data quality for each participant
    for _, row in df.iterrows():
        warnings = []
        # Check if pitch tracking failed (completely unvoiced utterance)
        if not bool(row.get("audio_pitch_detected", False)):
            warnings.append("pitch unavailable")
        # Check if audio ended abruptly at high volume
        if bool(row.get("audio_boundary_review_flag", False)):
            warnings.append("high-energy audio boundary")
        # Check if audio was trimmed specifically to excise background music
        if str(row.get("intentional_bgm_boundary_trim", "No")).lower() == "yes":
            warnings.append("intentional trim to avoid background music")
        # Check if background music overlap status was unverified
        if str(row.get("background_music_overlap", "Not flagged")).lower() == "unknown":
            warnings.append("background-music overlap unverified")
        # Check if speech completeness was verified
        if str(row.get("speech_complete_status", "Not flagged")).lower() == "unknown":
            warnings.append("speech completeness unverified")
        # Check if self-reported feeling text was unverified
        if str(row.get("self_reported_feeling_source", "")) == "Unverified word fill":
            warnings.append("feeling-text provenance unverified")
        # Check if video face detection fell below 80%
        if not bool(row.get("face_detection_quality_pass", False)):
            warnings.append("face detection below 80%")
            
        rows.append({
            "participant_code": row["participant_code"],
            "consent_obtained": bool(row["consent_obtained"]),
            "metadata_word_matches_filename": bool(row["metadata_word_matches_filename"]),
            "audio_pitch_detected": bool(row["audio_pitch_detected"]),
            "face_detection_rate": row["face_detection_rate"],
            "recording_valid_for_non_pitch_eda": bool(
                row["consent_obtained"] and 
                row["metadata_word_matches_filename"] and 
                row["face_detection_quality_pass"]
            ),
            "quality_warnings": "; ".join(warnings),
        })
        
    quality = pd.DataFrame(rows)
    # Ensure destination directory exists and write CSV
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    quality.to_csv(csv_path, index=False)
    
    # Construct Markdown executive summary
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
    """
    Master multimodal integration engine:
    1. Loads audio and facial feature tables.
    2. Validates 1-to-1 participant code correspondence (40 audio == 40 facial).
    3. Confirms identical survey metadata values between both tables.
    4. Merges into a single master tabular dataset (cleaned_multimodal_dataset.csv).
    5. Validates audio and video file stem correspondence.
    6. Checks for unexpected missing values and eliminates infinite values.
    7. Generates descriptive z-score scaled dataset for EDA plotting.
    8. Writes the authoritative Multimodal Data Dictionary and Quality Report.
    """
    # 1. Load both feature tables
    df_audio = pd.read_csv(audio_csv)
    df_facial = pd.read_csv(facial_csv)
    
    # 2. Enforce primary key uniqueness
    _validate_unique(df_audio, "audio dataset")
    _validate_unique(df_facial, "facial dataset")
    
    # 3. Check bidirectional set equality of participant codes
    audio_codes, facial_codes = set(df_audio.participant_code), set(df_facial.participant_code)
    if audio_codes != facial_codes:
        raise ValueError(f"Audio/facial participant mismatch: audio_only={sorted(audio_codes-facial_codes)}, facial_only={sorted(facial_codes-audio_codes)}")

    # 4. Enforce strict metadata consistency between audio and facial extractions
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

    # 5. Inner join on participant_code with strict one-to-one validation
    facial_only = [c for c in df_facial.columns if c == "participant_code" or c not in shared_metadata]
    df = pd.merge(df_audio, df_facial[facial_only], on="participant_code", how="inner", validate="one_to_one")
    
    # 6. Verify that audio filename stem matches video filename stem (e.g. Y4-011_Y4_AM_Kapoy)
    audio_stems = df.audio_filename.map(lambda value: Path(value).stem.casefold())
    video_stems = df.video_filename.map(lambda value: Path(value).stem.casefold())
    if not audio_stems.equals(video_stems):
        raise ValueError("Audio/video recording stems do not match one-to-one")

    # 7. Mathematical hygiene check: ensure zero infinite values in numeric columns
    numeric = df.select_dtypes(include=np.number)
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError("Infinite numeric values detected")
        
    # 8. Strict missing value policy: ONLY unvoiced pitch parameters and optional feeling text may be NaN
    allowed_missing = set(c for c in df.columns if c.startswith("audio_f0_") or c == "audio_period_variation_proxy") | {"self_reported_feeling"}
    unexpected_missing = {c: int(n) for c, n in df.isna().sum().items() if n and c not in allowed_missing}
    if unexpected_missing:
        raise ValueError(f"Unexpected missing values: {unexpected_missing}")

    # 9. Save unscaled raw multimodal dataset to disk
    Path(output_raw_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_raw_csv, index=False)

    # 10. Compute descriptive z-score standardization for EDA and visualization
    # DEFENSE POINT: Descriptive scaling across full dataset is for descriptive EDA only.
    # For machine learning, all scaling must be fit strictly INSIDE training folds to avoid data leakage!
    scaled = df.copy()
    parameters = []
    for column in predictor_columns(df):
        values = df[column].astype(float)
        # Compute sample mean and population standard deviation
        mean, std = float(values.mean(skipna=True)), float(values.std(skipna=True, ddof=0))
        parameters.append({
            "column": column, 
            "mean": mean, 
            "population_std": std, 
            "scope": "full dataset; descriptive EDA only"
        })
        # Standardize: z = (x - mean) / std
        scaled[column] = ((values - mean) / std).round(4) if std > 1e-12 else 0.0
        
    scaled.to_csv(output_scaled_csv, index=False)
    pd.DataFrame(parameters).to_csv(output_scaler_csv, index=False)

    # 11. Compile the master Data Dictionary describing every column
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
    
    # 12. Write the data quality audit reports
    _write_quality_report(df, quality_csv, quality_md)
    print(f"Cleaned multimodal dataset: {df.shape}; predictors={len(predictor_columns(df))}")
    print("Scaled dataset is for descriptive EDA only; fit preprocessing inside training folds for ML.")
    return df, scaled, df_dictionary


# Script entry point
if __name__ == "__main__":
    build_cleaned_multimodal_dataset()
