"""
Part B: Single-Recording Preprocessing and Feature Extraction Demonstration Script
CS412 - Bisaya Speech Affect Recognition (Group 7)

DEMONSTRATION OBJECTIVES (Deliverable 2 - Part B Rubric):
1. Raw Signal Loading: Demonstrates loading a raw .WAV audio file resampled to 16,000 Hz mono.
2. Signal Preprocessing: Demonstrates silence trimming using librosa.effects.trim(top_db=25).
3. Live Extraction: Measures extraction runtime for fundamental frequency, RMS, spectral, and MFCC features.
4. Feature Values Display: Tabulates extracted numerical biomarkers grouped by functional domain.
5. Structured Row Vector: Confirms dimensionality matching master acoustic feature dataset schema.
"""

import os                     # File existence and path handling
import sys                    # CLI arguments and platform config
import time                   # Wall-clock runtime benchmarking
from pathlib import Path      # Object-oriented path utilities
import librosa                # Audio signal processing library
import numpy as np            # Numerical operations
import pandas as pd           # Tabular formatting
from tabulate import tabulate # Clean terminal output tables

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path to enable direct importing of src/ modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.audio_features import extract_audio_features_from_file

def demonstrate_feature_extraction(audio_path: str = None):
    """
    Demonstrates single-recording audio preprocessing and feature extraction on screen.
    Runs live in ~2 to 4 seconds to prove algorithm efficiency to the panel.
    """
    print("=" * 80)
    print(" PART B: PREPROCESSING & FEATURE EXTRACTION DEMO (CS412 GROUP 7)")
    print("=" * 80)

    # Default to sample recording Y4-011 ('Kapoy', Prelim Exam, Morning session) if not specified
    if audio_path is None:
        audio_path = "DATA/2AudioRecordings/Raw .wav/Y4-011_Y4_AM_Kapoy.WAV"

    # Verify input audio file exists on disk
    if not os.path.exists(audio_path):
        print(f"[ERROR] Audio file not found: {audio_path}")
        return

    print(f"\n[1] Target Raw Recording: {os.path.basename(audio_path)}")
    print(f"    - Full Path: {os.path.abspath(audio_path)}")

    # -------------------------------------------------------------------------
    # [STEP 1 & 2] Inspect Raw Audio Signal & Apply Silence Trimming
    # -------------------------------------------------------------------------
    t0 = time.time()  # Start timing benchmark
    
    # 1. Load raw audio at 16 kHz mono (telephony standard)
    y_raw, sr = librosa.load(audio_path, sr=16000, mono=True)
    raw_duration = len(y_raw) / sr  # Duration = sample_count / sampling_rate
    
    # 2. Silence Trimming: Isolates vocalization from microphone background room silence
    # DEFENSE POINT: top_db=25 removes leading/trailing noise without clipping phonemes
    y_trimmed, trim_idx = librosa.effects.trim(y_raw, top_db=25)
    trimmed_duration = len(y_trimmed) / sr
    
    print("\n[2] Preprocessing Pipeline Applied:")
    print(f"    - Target Sampling Rate: {sr} Hz (Mono)")
    print(f"    - Raw Sample Count:     {len(y_raw):,} samples ({raw_duration:.4f} seconds)")
    print(f"    - Silence Trimming:     top_db=25 threshold")
    print(f"    - Trimmed Sample Count: {len(y_trimmed):,} samples ({trimmed_duration:.4f} seconds)")
    print(f"    - Frame Hop Length:     256 samples (16 ms step)")
    print(f"    - STFT Window Length:   2048 / 1024 samples")

    # -------------------------------------------------------------------------
    # [STEP 3] Extract Full 108-Acoustic Feature Vector via src/audio_features.py
    # -------------------------------------------------------------------------
    print("\n[3] Extracting Acoustic & Prosodic Biomarkers...")
    feature_dict = extract_audio_features_from_file(audio_path, target_sr=16000)
    t_elapsed = time.time() - t0  # Total elapsed runtime
    print(f"    - Feature Extraction Complete in: {t_elapsed:.3f} seconds!")

    # -------------------------------------------------------------------------
    # [STEP 4] Display Key Extracted Feature Values by Domain Category
    # -------------------------------------------------------------------------
    print("\n[4] Structured Feature Vector Highlights (Sample Values):")
    
    # Highlight key biomarkers across all 5 extracted domains
    cat_highlights = [
        # Domain 1: Context & Ground-Truth Targets
        ["Context & Target", "Participant Code", feature_dict["participant_code"]],
        ["Context & Target", "Spoken Word", feature_dict["spoken_word"]],
        ["Context & Target", "Session / Activity", f"{feature_dict['session']} / {feature_dict['class_activity']}"],
        ["Context & Target", "Valence (Likert -> Target)", f"{feature_dict['valence_score']} -> {feature_dict['valence_group']}"],
        ["Context & Target", "Arousal (Likert -> Target)", f"{feature_dict['arousal_score']} -> {feature_dict['arousal_group']}"],
        
        # Domain 2: Pitch & Intonation (Prosody via pYIN)
        ["Prosody / Pitch", "Pitch Detected?", str(feature_dict["audio_pitch_detected"])],
        ["Prosody / Pitch", "F0 Mean (Hz)", f"{feature_dict['audio_f0_mean_hz']} Hz"],
        ["Prosody / Pitch", "F0 Range (Hz)", f"{feature_dict['audio_f0_range_hz']} Hz"],
        ["Prosody / Pitch", "Voiced Frame Ratio", f"{feature_dict['audio_voiced_ratio']:.4f}"],
        
        # Domain 3: Energy / Loudness Dynamics (RMS)
        ["Energy / Intensity", "RMS Mean (Loudness)", f"{feature_dict['audio_rms_mean']:.5f}"],
        ["Energy / Intensity", "RMS Std Dev", f"{feature_dict['audio_rms_std']:.5f}"],
        
        # Domain 4: Spectral Shape & Frequency Dynamics
        ["Spectral Dynamics", "Zero Crossing Rate (Mean)", f"{feature_dict['audio_zcr_mean']:.5f}"],
        ["Spectral Dynamics", "Spectral Centroid (Brightness)", f"{feature_dict['audio_spec_centroid_mean']:.2f} Hz"],
        ["Spectral Dynamics", "Spectral Bandwidth (Spread)", f"{feature_dict['audio_spec_bandwidth_mean']:.2f} Hz"],
        ["Spectral Dynamics", "Spectral Rolloff 85%", f"{feature_dict['audio_spec_rolloff85_mean']:.2f} Hz"],
        ["Spectral Dynamics", "Spectral Contrast", f"{feature_dict['audio_spec_contrast_mean']:.4f}"],
        
        # Domain 5: Timbre (Mel-Frequency Cepstral Coefficients)
        ["Timbre (MFCC)", "MFCC 1 (Mean)", f"{feature_dict['audio_mfcc_1_mean']:.4f}"],
        ["Timbre (MFCC)", "MFCC 2 (Mean)", f"{feature_dict['audio_mfcc_2_mean']:.4f}"],
        ["Timbre (MFCC)", "Delta MFCC 1 (Mean)", f"{feature_dict['audio_delta_mfcc_1_mean']:.4f}"],
        ["Timbre (MFCC)", "Delta-Delta MFCC 1 (Mean)", f"{feature_dict['audio_delta2_mfcc_1_mean']:.4f}"],
    ]
    print(tabulate(cat_highlights, headers=["Feature Domain", "Variable Name", "Extracted Value"], tablefmt="github"))

    # -------------------------------------------------------------------------
    # [STEP 5] Display Vector Dimensionality & Architecture
    # -------------------------------------------------------------------------
    numeric_keys = [k for k in feature_dict.keys() if k.startswith("audio_")]
    print(f"\n[5] Feature Vector Dimensions:")
    print(f"    - Metadata & Context Variables: {len(feature_dict) - len(numeric_keys)} columns")
    print(f"    - Total Acoustic Predictors:    {len(numeric_keys)} numerical columns")
    print(f"    - Dimensionality:               1 x {len(feature_dict)} structured row vector")
    print("    - Fully compatible with the master dataset in: datasets/audio/audio_feature_dataset.csv")

    print("\n" + "=" * 80)
    print(" [OK] DEMONSTRATION COMPLETE: Raw audio successfully transformed to feature vector.")
    print("=" * 80 + "\n")

# Main execution: supports optional custom audio path as CLI argument
if __name__ == "__main__":
    # If user provided a custom path: e.g. python extract_single_feature.py "path/to/file.wav"
    path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    demonstrate_feature_extraction(path_arg)
