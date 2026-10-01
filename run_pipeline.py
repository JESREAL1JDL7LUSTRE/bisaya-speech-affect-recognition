"""
Master Execution Pipeline for Bisaya Speech and Affect Recognition Project.
Executes the complete Second Deliverables pipeline:
1. Audio Feature Extraction (Raw .wav)
2. Facial Feature Extraction (Raw Cut Video)
3. Multimodal Dataset Merging, Validation, Scaling, & Dictionary
4. Descriptive Statistics & Hypothesis Testing
5. Publication-Quality Visualizations
"""

import os
import sys
import time

# Ensure workspace root is in path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.audio_features import extract_all_audio_features
from src.facial_features import extract_all_facial_features
from src.multimodal_dataset import build_cleaned_multimodal_dataset
from src.statistics_analysis import run_descriptive_statistics
from src.visualizations import generate_all_visualizations


def main():
    t_start = time.time()
    print("=" * 75)
    print(" BISAYA SPEECH & AFFECT RECOGNITION - SECOND DELIVERABLES PIPELINE")
    print("=" * 75)
    
    raw_audio_dir = "DATA/2AudioRecordings/Raw .wav"
    raw_video_dir = "DATA/3ImageRecording/Raw Cut Video"
    
    audio_csv = "datasets/audio/audio_feature_dataset.csv"
    facial_csv = "datasets/facial/facial_feature_dataset.csv"
    raw_multimodal_csv = "datasets/multimodal/cleaned_multimodal_dataset.csv"
    scaled_multimodal_csv = "datasets/multimodal/cleaned_multimodal_dataset_scaled.csv"
    data_dict_csv = "datasets/multimodal/multimodal_data_dictionary.csv"
    stats_dir = "outputs/descriptive_statistics"
    vis_dir = "outputs/visualizations"
    
    # -------------------------------------------------------------
    # Step 1: Audio Feature Extraction
    # -------------------------------------------------------------
    print("\n>>> STEP 1/5: Extracting Acoustic and Prosodic Features from Raw Audio...")
    df_audio = extract_all_audio_features(raw_audio_dir, audio_csv)
    print(f"[OK] Step 1 Complete: {df_audio.shape[0]} recordings, {df_audio.shape[1]} features.")
    
    # -------------------------------------------------------------
    # Step 2: Facial Feature Extraction
    # -------------------------------------------------------------
    print("\n>>> STEP 2/5: Extracting FACS Blendshapes & Landmarks from Raw Video...")
    df_facial = extract_all_facial_features(raw_video_dir, facial_csv)
    print(f"[OK] Step 2 Complete: {df_facial.shape[0]} recordings, {df_facial.shape[1]} features.")
    
    # -------------------------------------------------------------
    # Step 3: Multimodal Dataset Construction & Cleaning
    # -------------------------------------------------------------
    print("\n>>> STEP 3/5: Merging, Validating, and Normalizing Multimodal Dataset...")
    df_multi, df_scaled, df_dict = build_cleaned_multimodal_dataset(
        audio_csv=audio_csv,
        facial_csv=facial_csv,
        output_raw_csv=raw_multimodal_csv,
        output_scaled_csv=scaled_multimodal_csv,
        output_dictionary_csv=data_dict_csv
    )
    print(f"[OK] Step 3 Complete: {df_multi.shape[0]} observations, {df_multi.shape[1]} multimodal variables.")
    
    # -------------------------------------------------------------
    # Step 4: Descriptive Statistics & Hypothesis Testing
    # -------------------------------------------------------------
    print("\n>>> STEP 4/5: Computing Descriptive Statistics & Statistical Tests...")
    stats_results = run_descriptive_statistics(raw_multimodal_csv, stats_dir)
    print(f"[OK] Step 4 Complete: Descriptive reports & CSV tables saved to {stats_dir}.")
    
    # -------------------------------------------------------------
    # Step 5: High-Resolution Visualizations
    # -------------------------------------------------------------
    print("\n>>> STEP 5/5: Generating Publication-Grade Visualizations (300 DPI)...")
    generate_all_visualizations(raw_multimodal_csv, vis_dir)
    print(f"[OK] Step 5 Complete: All 10 figures saved to {vis_dir}.")
    
    elapsed = time.time() - t_start
    print("\n" + "=" * 75)
    print(f" ALL DELIVERABLES SUCCESSFULLY PRODUCED IN {elapsed:.2f} SECONDS!")
    print("=" * 75)
    print(f"Datasets generated in:    {os.path.abspath('datasets')}")
    print(f"Statistics generated in:  {os.path.abspath(stats_dir)}")
    print(f"Visualizations generated in: {os.path.abspath(vis_dir)}")


if __name__ == "__main__":
    main()
