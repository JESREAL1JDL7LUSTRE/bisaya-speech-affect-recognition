"""
Part B: Single-Recording Preprocessing and Feature Extraction Demonstration
CS412 - Bisaya Speech Affect Recognition (Group 7)

Demonstrates:
1. Loading a raw .WAV audio recording (16 kHz mono)
2. Silence trimming and duration analysis
3. Extracting 108 acoustic and prosodic features using librosa (pYIN, RMS, ZCR, Spectral, MFCC)
4. Displaying the resulting structured feature vector and column values
"""

import os
import sys
import time
from pathlib import Path
import librosa
import numpy as np
import pandas as pd
from tabulate import tabulate

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.audio_features import extract_audio_features_from_file

def demonstrate_feature_extraction(audio_path: str = None):
    print("=" * 80)
    print(" PART B: PREPROCESSING & FEATURE EXTRACTION DEMO (CS412 GROUP 7)")
    print("=" * 80)

    if audio_path is None:
        # Default sample recording: Y4-011 (Prelim Exam, 'Kapoy')
        audio_path = "DATA/2AudioRecordings/Raw .wav/Y4-011_Y4_AM_Kapoy.WAV"

    if not os.path.exists(audio_path):
        print(f"[ERROR] Audio file not found: {audio_path}")
        return

    print(f"\n[1] Target Raw Recording: {os.path.basename(audio_path)}")
    print(f"    - Full Path: {os.path.abspath(audio_path)}")

    # 1. Inspect Raw Signal
    t0 = time.time()
    y_raw, sr = librosa.load(audio_path, sr=16000, mono=True)
    raw_duration = len(y_raw) / sr
    
    # 2. Preprocessing: Silence Trimming
    y_trimmed, trim_idx = librosa.effects.trim(y_raw, top_db=25)
    trimmed_duration = len(y_trimmed) / sr
    
    print("\n[2] Preprocessing Pipeline Applied:")
    print(f"    - Target Sampling Rate: {sr} Hz (Mono)")
    print(f"    - Raw Sample Count:     {len(y_raw):,} samples ({raw_duration:.4f} seconds)")
    print(f"    - Silence Trimming:     top_db=25 threshold")
    print(f"    - Trimmed Sample Count: {len(y_trimmed):,} samples ({trimmed_duration:.4f} seconds)")
    print(f"    - Frame Hop Length:     256 samples (16 ms step)")
    print(f"    - STFT Window Length:   2048 / 1024 samples")

    # 3. Extract Full Feature Vector
    print("\n[3] Extracting Acoustic & Prosodic Biomarkers...")
    feature_dict = extract_audio_features_from_file(audio_path, target_sr=16000)
    t_elapsed = time.time() - t0
    print(f"    - Feature Extraction Complete in: {t_elapsed:.3f} seconds!")

    # 4. Display Key Extracted Feature Values by Category
    print("\n[4] Structured Feature Vector Highlights (Sample Values):")
    
    cat_highlights = [
        ["Context & Target", "Participant Code", feature_dict["participant_code"]],
        ["Context & Target", "Spoken Word", feature_dict["spoken_word"]],
        ["Context & Target", "Session / Activity", f"{feature_dict['session']} / {feature_dict['class_activity']}"],
        ["Context & Target", "Valence (Likert -> Target)", f"{feature_dict['valence_score']} -> {feature_dict['valence_group']}"],
        ["Context & Target", "Arousal (Likert -> Target)", f"{feature_dict['arousal_score']} -> {feature_dict['arousal_group']}"],
        ["Prosody / Pitch", "Pitch Detected?", str(feature_dict["audio_pitch_detected"])],
        ["Prosody / Pitch", "F0 Mean (Hz)", f"{feature_dict['audio_f0_mean_hz']} Hz"],
        ["Prosody / Pitch", "F0 Range (Hz)", f"{feature_dict['audio_f0_range_hz']} Hz"],
        ["Prosody / Pitch", "Voiced Frame Ratio", f"{feature_dict['audio_voiced_ratio']:.4f}"],
        ["Energy / Intensity", "RMS Mean (Loudness)", f"{feature_dict['audio_rms_mean']:.5f}"],
        ["Energy / Intensity", "RMS Std Dev", f"{feature_dict['audio_rms_std']:.5f}"],
        ["Spectral Dynamics", "Zero Crossing Rate (Mean)", f"{feature_dict['audio_zcr_mean']:.5f}"],
        ["Spectral Dynamics", "Spectral Centroid (Brightness)", f"{feature_dict['audio_spec_centroid_mean']:.2f} Hz"],
        ["Spectral Dynamics", "Spectral Bandwidth (Spread)", f"{feature_dict['audio_spec_bandwidth_mean']:.2f} Hz"],
        ["Spectral Dynamics", "Spectral Rolloff 85%", f"{feature_dict['audio_spec_rolloff85_mean']:.2f} Hz"],
        ["Spectral Dynamics", "Spectral Contrast", f"{feature_dict['audio_spec_contrast_mean']:.4f}"],
        ["Timbre (MFCC)", "MFCC 1 (Mean)", f"{feature_dict['audio_mfcc_1_mean']:.4f}"],
        ["Timbre (MFCC)", "MFCC 2 (Mean)", f"{feature_dict['audio_mfcc_2_mean']:.4f}"],
        ["Timbre (MFCC)", "Delta MFCC 1 (Mean)", f"{feature_dict['audio_delta_mfcc_1_mean']:.4f}"],
        ["Timbre (MFCC)", "Delta-Delta MFCC 1 (Mean)", f"{feature_dict['audio_delta2_mfcc_1_mean']:.4f}"],
    ]
    print(tabulate(cat_highlights, headers=["Feature Domain", "Variable Name", "Extracted Value"], tablefmt="github"))

    # 5. Display Vector Structure Summary
    numeric_keys = [k for k in feature_dict.keys() if k.startswith("audio_")]
    print(f"\n[5] Feature Vector Dimensions:")
    print(f"    - Metadata & Context Variables: {len(feature_dict) - len(numeric_keys)} columns")
    print(f"    - Total Acoustic Predictors:    {len(numeric_keys)} numerical columns")
    print(f"    - Dimensionality:               1 x {len(feature_dict)} structured row vector")
    print("    - Fully compatible with the master dataset in: datasets/audio/audio_feature_dataset.csv")

    print("\n" + "=" * 80)
    print(" [OK] DEMONSTRATION COMPLETE: Raw audio successfully transformed to feature vector.")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    demonstrate_feature_extraction(path_arg)
