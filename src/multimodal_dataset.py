"""
Multimodal Dataset Construction and Cleaning Module.
Merges Audio and Facial feature datasets, validates data integrity,
computes cross-modal interaction indicators, and generates both unscaled
and standardized machine-learning-ready datasets along with a comprehensive data dictionary.
"""

import os
import sys

# Ensure workspace root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def build_cleaned_multimodal_dataset(
    audio_csv: str = "datasets/audio_feature_dataset.csv",
    facial_csv: str = "datasets/facial_feature_dataset.csv",
    output_raw_csv: str = "datasets/cleaned_multimodal_dataset.csv",
    output_scaled_csv: str = "datasets/cleaned_multimodal_dataset_scaled.csv",
    output_dictionary_csv: str = "datasets/multimodal_data_dictionary.csv"
):
    print("Loading Audio and Facial Feature Datasets...")
    df_audio = pd.read_csv(audio_csv)
    df_facial = pd.read_csv(facial_csv)
    
    print(f"Audio Dataset Shape: {df_audio.shape}")
    print(f"Facial Dataset Shape: {df_facial.shape}")
    
    # Common metadata columns
    shared_cols = [
        "participant_code", "year_level", "class_activity", "subject",
        "session", "session_name", "spoken_word", "self_reported_feeling",
        "valence_score", "valence_label", "arousal_score", "arousal_label",
        "affect_category", "word_english", "date_collected"
    ]
    
    # Check for matching participants
    assert set(df_audio["participant_code"]) == set(df_facial["participant_code"]), \
        "Mismatch between participant codes in audio and facial datasets!"
        
    # Drop shared metadata from facial to prevent collision on merge
    facial_feature_cols = [c for c in df_facial.columns if c not in shared_cols]
    
    # Merge on participant_code
    df_multimodal = pd.merge(
        df_audio,
        df_facial[["participant_code"] + facial_feature_cols],
        on="participant_code",
        how="inner"
    ).copy()
    
    if "self_reported_feeling" in df_multimodal.columns:
        df_multimodal["self_reported_feeling"] = df_multimodal["self_reported_feeling"].fillna(df_multimodal["spoken_word"])
    
    # Compute Cross-Modal Affective Indices
    # 1. Multimodal Expressiveness Index: normalized audio RMS energy * facial expressiveness score
    audio_rms_norm = (df_multimodal["audio_rms_mean"] - df_multimodal["audio_rms_mean"].min()) / \
                     (df_multimodal["audio_rms_mean"].max() - df_multimodal["audio_rms_mean"].min() + 1e-6)
    face_exp_norm = (df_multimodal["face_expressiveness_score"] - df_multimodal["face_expressiveness_score"].min()) / \
                    (df_multimodal["face_expressiveness_score"].max() - df_multimodal["face_expressiveness_score"].min() + 1e-6)
    df_multimodal["multimodal_expressiveness_index"] = round(0.5 * audio_rms_norm + 0.5 * face_exp_norm, 4)
    
    # 2. Facial Valence Proxy: Smile mean minus Frown mean
    df_multimodal["face_valence_proxy"] = round(df_multimodal["face_smile_mean"] - df_multimodal["face_frown_mean"], 4)
    
    # 3. Audio Brightness / Arousal Proxy: Spectral centroid normalized + F0 mean normalized
    f0_norm = (df_multimodal["audio_f0_mean_hz"] - df_multimodal["audio_f0_mean_hz"].min()) / \
              (df_multimodal["audio_f0_mean_hz"].max() - df_multimodal["audio_f0_mean_hz"].min() + 1e-6)
    sc_norm = (df_multimodal["audio_spec_centroid_mean"] - df_multimodal["audio_spec_centroid_mean"].min()) / \
              (df_multimodal["audio_spec_centroid_mean"].max() - df_multimodal["audio_spec_centroid_mean"].min() + 1e-6)
    df_multimodal["audio_arousal_proxy"] = round(0.5 * f0_norm + 0.5 * sc_norm, 4)
    
    # Data Cleaning and Quality Checks
    print("\n--- Performing Data Hygiene & Validation Checks ---")
    null_counts = df_multimodal.isnull().sum()
    total_nulls = null_counts.sum()
    print(f"Total Missing / Null Values: {total_nulls}")
    assert total_nulls == 0, f"Detected {total_nulls} null values in merged dataset!"
    
    # Check for infinite values in numeric columns
    numeric_cols = df_multimodal.select_dtypes(include=[np.number]).columns
    inf_counts = np.isinf(df_multimodal[numeric_cols]).sum().sum()
    print(f"Total Infinite Values: {inf_counts}")
    assert inf_counts == 0, "Detected infinite values in numeric columns!"
    
    print(f"Cleaned Multimodal Dataset: {df_multimodal.shape[0]} rows, {df_multimodal.shape[1]} columns")
    
    # Save Cleaned Raw Multimodal Dataset
    os.makedirs(os.path.dirname(os.path.abspath(output_raw_csv)), exist_ok=True)
    df_multimodal.to_csv(output_raw_csv, index=False)
    print(f"Cleaned Raw Multimodal Dataset saved to: {output_raw_csv}")
    
    # Generate Standard-Scaled Version (z-score normalized for ML)
    df_scaled = df_multimodal.copy()
    scaler = StandardScaler()
    df_scaled[numeric_cols] = scaler.fit_transform(df_multimodal[numeric_cols]).round(4)
    df_scaled.to_csv(output_scaled_csv, index=False)
    print(f"Standard-Scaled Multimodal Dataset saved to: {output_scaled_csv}")
    
    # Build Data Dictionary
    print("\nGenerating Multimodal Data Dictionary...")
    dict_records = []
    
    for col in df_multimodal.columns:
        if col in shared_cols or "filename" in col or "filepath" in col:
            modality = "Metadata"
            family = "Identifiers & Educational Context"
            desc = f"Participant / Context variable ({col})"
            unit = "categorical/string"
        elif col.startswith("audio_"):
            modality = "Acoustic / Speech"
            if "f0" in col or "jitter" in col or "voiced" in col:
                family = "Prosodic - Pitch & Vocal Stability"
                desc = f"Fundamental frequency or pitch variation measure: {col}"
                unit = "Hz" if "hz" in col else "ratio"
            elif "rms" in col or "shimmer" in col:
                family = "Prosodic - Intensity & Loudness"
                desc = f"Root-Mean-Square energy or amplitude stability: {col}"
                unit = "amplitude"
            elif "zcr" in col:
                family = "Temporal - Zero-Crossing Rate"
                desc = "Rate of sign-changes along the speech signal"
                unit = "rate"
            elif "mfcc" in col:
                family = "Timbral - Mel-Frequency Cepstral Coefficients"
                desc = f"Spectral envelope representation: {col}"
                unit = "coefficient"
            elif "spec" in col or "chroma" in col:
                family = "Spectral Characteristics"
                desc = f"Frequency distribution and brightness: {col}"
                unit = "Hz/index"
            else:
                family = "Audio Metadata"
                desc = f"Audio property: {col}"
                unit = "seconds/count"
        elif col.startswith("face_"):
            modality = "Visual / Facial"
            if "smile" in col or "frown" in col or "brow" in col or "jaw" in col or "disgust" in col:
                family = "FACS Action Unit Composite"
                desc = f"Facial Action Coding System affect indicator: {col}"
                unit = "score [0-1]"
            elif "ear" in col or "mar" in col:
                family = "Geometric Landmark Ratio"
                desc = f"Geometric landmark aspect ratio (EAR/MAR): {col}"
                unit = "ratio"
            elif "pitch" in col or "yaw" in col or "roll" in col:
                family = "Head Pose"
                desc = f"Head rotation angle: {col}"
                unit = "degrees"
            elif "blendshapes" in col or col.endswith("_mean") or col.endswith("_std") or col.endswith("_max"):
                family = "MediaPipe FACS Blendshape"
                desc = f"Standard 52 FACS Blendshape coefficient: {col}"
                unit = "score [0-1]"
            else:
                family = "Video Metadata"
                desc = f"Video property: {col}"
                unit = "frames/seconds/rate"
        elif col.startswith("multimodal_") or "proxy" in col:
            modality = "Multimodal Composite"
            family = "Cross-Modal Interaction"
            desc = f"Synthesized audio-visual affective index: {col}"
            unit = "normalized [0-1]"
        else:
            modality = "Other"
            family = "General"
            desc = col
            unit = "unspecified"
            
        dict_records.append({
            "Column_Name": col,
            "Modality": modality,
            "Feature_Family": family,
            "Data_Type": str(df_multimodal[col].dtype),
            "Unit": unit,
            "Description": desc
        })
        
    df_dict = pd.DataFrame(dict_records)
    df_dict.to_csv(output_dictionary_csv, index=False)
    print(f"Multimodal Data Dictionary saved to: {output_dictionary_csv} ({len(df_dict)} entries)")
    
    return df_multimodal, df_scaled, df_dict


if __name__ == "__main__":
    build_cleaned_multimodal_dataset()
