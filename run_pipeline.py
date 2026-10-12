"""
Master Execution Pipeline for Bisaya Speech and Affect Recognition Project.
CS412 - Group 7 (4th-Year Computer Science)

END-TO-END PIPELINE OVERVIEW:
This orchestrator executes the complete data engineering and exploratory research workflow:
1. Audio Feature Extraction: 40 raw .WAV files -> 108 acoustic predictors (audio_feature_dataset.csv)
2. Facial Feature Extraction: 40 raw .MP4 files -> 225 FACS blendshapes (facial_feature_dataset.csv)
3. Multimodal Dataset Construction: Merges audio & video with 1-to-1 participant correspondence
4. Descriptive Statistics: Computes missing-aware univariate stats, Welch t-tests, and Benjamini-Hochberg FDR
5. Exploratory Visualizations: Renders 300 DPI publication plots grounded in self-reported affect
"""

import os                     # Operating system utilities (path checking, resolving)
import sys                    # System-level interpreter configuration
import time                   # Wall-clock runtime profiling

# Ensure workspace root directory is in sys.path so modules in src/ can be cleanly imported
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Core pipeline stage imports from src/
from src.audio_features import extract_all_audio_features
from src.facial_features import extract_all_facial_features
from src.multimodal_dataset import build_cleaned_multimodal_dataset
from src.statistics_analysis import run_descriptive_statistics
from src.visualizations import generate_all_visualizations


def main():
    """
    Executes all 5 stages of the research pipeline sequentially.
    Profiles execution time and reports output file destinations.
    """
    t_start = time.time()  # Start master timer
    print("=" * 75)
    print(" BISAYA SPEECH & AFFECT RECOGNITION - SECOND DELIVERABLES PIPELINE")
    print("=" * 75)
    
    # -------------------------------------------------------------------------
    # [INPUTS & OUTPUTS CONFIGURATION]
    # -------------------------------------------------------------------------
    # Raw source directories on disk
    raw_audio_dir = "DATA/2AudioRecordings/Raw .wav"
    raw_video_dir = "DATA/3ImageRecording/Raw Cut Video"
    
    # Target dataset destination paths
    audio_csv = "datasets/audio/audio_feature_dataset.csv"
    facial_csv = "datasets/facial/facial_feature_dataset.csv"
    raw_multimodal_csv = "datasets/multimodal/cleaned_multimodal_dataset.csv"
    scaled_multimodal_csv = "datasets/multimodal/cleaned_multimodal_dataset_scaled.csv"
    data_dict_csv = "datasets/multimodal/multimodal_data_dictionary.csv"
    
    # Output artifact directories for plots and reports
    audio_dir = "outputs/audio"
    facial_dir = "outputs/facial"
    multimodal_dir = "outputs/multimodal"
    outputs_dir = "outputs"
    
    # -------------------------------------------------------------------------
    # [STEP 1/5] Extract Acoustic & Prosodic Biomarkers from Raw Audio
    # -------------------------------------------------------------------------
    # Reads 40 .WAV files; resamples to 16 kHz; trims silence; extracts pYIN F0, RMS, ZCR, MFCCs
    print("\n>>> STEP 1/5: Extracting Acoustic and Prosodic Features from Raw Audio...")
    df_audio = extract_all_audio_features(raw_audio_dir, audio_csv)
    print(f"[OK] Step 1 Complete: {df_audio.shape[0]} recordings, {df_audio.shape[1]} features.")
    
    # -------------------------------------------------------------------------
    # [STEP 2/5] Extract Facial Blendshapes & Landmarks from Raw Video
    # -------------------------------------------------------------------------
    # Reads 40 .MP4 videos; uses MediaPipe FaceMesh; extracts 225 FACS action units & eye/mouth aspect ratios
    print("\n>>> STEP 2/5: Extracting FACS Blendshapes & Landmarks from Raw Video...")
    df_facial = extract_all_facial_features(raw_video_dir, facial_csv)
    print(f"[OK] Step 2 Complete: {df_facial.shape[0]} recordings, {df_facial.shape[1]} features.")
    
    # -------------------------------------------------------------------------
    # [STEP 3/5] Multimodal Merging, Validation, Scaling, & Data Dictionary
    # -------------------------------------------------------------------------
    # Validates bidirectional 1-to-1 participant correspondence across modalities;
    # Produces unscaled dataset (physical units), scaled dataset (z-scores), and data dictionary
    print("\n>>> STEP 3/5: Validating one-to-one correspondence and building multimodal datasets...")
    df_multi, df_scaled, df_dict = build_cleaned_multimodal_dataset(
        audio_csv=audio_csv,
        facial_csv=facial_csv,
        output_raw_csv=raw_multimodal_csv,
        output_scaled_csv=scaled_multimodal_csv,
        output_dictionary_csv=data_dict_csv
    )
    print(f"[OK] Step 3 Complete: {df_multi.shape[0]} observations, {df_multi.shape[1]} multimodal variables.")
    
    # -------------------------------------------------------------------------
    # [STEP 4/5] Descriptive Statistics & FDR-Adjusted Hypothesis Testing
    # -------------------------------------------------------------------------
    # Missing-aware univariate stats (mean, std, IQR), AM vs PM Welch t-tests, Mann-Whitney U,
    # and Benjamini-Hochberg False Discovery Rate (FDR) adjustments for multiple comparisons
    print("\n>>> STEP 4/5: Computing missing-aware statistics and summaries...")
    stats_results = run_descriptive_statistics(
        raw_multimodal_csv,
        audio_output_dir=audio_dir,
        facial_output_dir=facial_dir,
        multimodal_output_dir=multimodal_dir,
    )
    print(f"[OK] Step 4 Complete: Statistics CSVs saved to {audio_dir}, {facial_dir}, and {multimodal_dir}.")
    
    # -------------------------------------------------------------------------
    # [STEP 5/5] Publication-Quality Exploratory Visualizations (300 DPI)
    # -------------------------------------------------------------------------
    # Generates high-resolution figures: affect distributions, boxplots, correlation heatmaps, PCA projections
    print("\n>>> STEP 5/5: Generating rating-grounded exploratory visualizations (300 DPI)...")
    generate_all_visualizations(raw_multimodal_csv, outputs_dir)
    print(f"[OK] Step 5 Complete: All figures saved directly to {audio_dir}, {facial_dir}, and {multimodal_dir}.")
    
    # Report total completion time and directory locations
    elapsed = time.time() - t_start
    print("\n" + "=" * 75)
    print(f" ALL DELIVERABLES SUCCESSFULLY PRODUCED IN {elapsed:.2f} SECONDS!")
    print("=" * 75)
    print(f"Datasets generated in:        {os.path.abspath('datasets')}")
    print(f"Audio Outputs in:             {os.path.abspath(audio_dir)}")
    print(f"Facial Outputs in:            {os.path.abspath(facial_dir)}")
    print(f"Multimodal Outputs in:        {os.path.abspath(multimodal_dir)}")


# Entry point trigger
if __name__ == "__main__":
    main()
