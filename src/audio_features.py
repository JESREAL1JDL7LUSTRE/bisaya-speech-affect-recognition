"""
Audio Feature Extraction Module for Bisaya Speech Affect Recognition.
Processes raw WAV recordings and extracts acoustic, prosodic, and spectral features.
"""

import os
import glob
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm


AFFECT_TAXONOMY = {
    # Fatigue / Physical & Mental Exhaustion
    "Kapoy": {"category": "Fatigue", "valence_group": "Negative", "arousal_group": "Low", "english": "Tired / Exhausted"},
    "Hangak": {"category": "Fatigue", "valence_group": "Negative", "arousal_group": "Moderate", "english": "Breathless / Gasping"},
    "Labad": {"category": "Fatigue", "valence_group": "Negative", "arousal_group": "Moderate", "english": "Headache / Stressed"},
    "Mamatay": {"category": "Fatigue", "valence_group": "Negative", "arousal_group": "Moderate", "english": "Dying / Drained"},
    "Lutang": {"category": "Fatigue", "valence_group": "Negative", "arousal_group": "Low", "english": "Spaced out / Floating"},
    
    # Stress / Anxiety / Cognitive Strain
    "Kulba": {"category": "Stress/Anxiety", "valence_group": "Negative", "arousal_group": "High", "english": "Nervous / Anxious"},
    "Kapuliki": {"category": "Stress/Anxiety", "valence_group": "Negative", "arousal_group": "High", "english": "Overwhelmed / Frantic"},
    "Lisod": {"category": "Stress/Anxiety", "valence_group": "Negative", "arousal_group": "Moderate", "english": "Difficult / Challenging"},
    "Libog": {"category": "Stress/Anxiety", "valence_group": "Negative", "arousal_group": "Moderate", "english": "Confused / Perplexed"},
    "Gaduha-duha": {"category": "Stress/Anxiety", "valence_group": "Negative", "arousal_group": "Moderate", "english": "Hesitant / In doubt"},
    "Pildi": {"category": "Stress/Anxiety", "valence_group": "Negative", "arousal_group": "Low", "english": "Defeated / Lost"},
    "Gakaguol": {"category": "Stress/Anxiety", "valence_group": "Negative", "arousal_group": "Low", "english": "Grieving / Gloomy"},
    "Kaguol": {"category": "Stress/Anxiety", "valence_group": "Negative", "arousal_group": "Low", "english": "Sad / Sorrowful"},
    
    # Neutral / Ambivalence / Reflective
    "Okay": {"category": "Neutral", "valence_group": "Neutral", "arousal_group": "Moderate", "english": "Okay / Fine"},
    "Ambot": {"category": "Neutral", "valence_group": "Neutral", "arousal_group": "Low", "english": "I don't know / Ambivalent"},
    "Hilom": {"category": "Neutral", "valence_group": "Neutral", "arousal_group": "Low", "english": "Quiet / Silent"},
    "Kamatuoran": {"category": "Neutral", "valence_group": "Neutral", "arousal_group": "Moderate", "english": "Truth / Acceptance"},
    
    # Positive / Relief / Joy / Satisfaction
    "Hapsay": {"category": "Positive", "valence_group": "Positive", "arousal_group": "Moderate", "english": "Smooth / Orderly"},
    "Nahuwasan": {"category": "Positive", "valence_group": "Positive", "arousal_group": "Moderate", "english": "Relieved"},
    "Relibo": {"category": "Positive", "valence_group": "Positive", "arousal_group": "Moderate", "english": "Relieved"},
    "Lingaw": {"category": "Positive", "valence_group": "Positive", "arousal_group": "High", "english": "Fun / Enjoyable"},
    "Chuy": {"category": "Positive", "valence_group": "Positive", "arousal_group": "Moderate", "english": "Cool / Chill"},
    "Thrilled": {"category": "Positive", "valence_group": "Positive", "arousal_group": "High", "english": "Excited / Thrilled"},
    "Gihigugma": {"category": "Positive", "valence_group": "Positive", "arousal_group": "Moderate", "english": "Loved / Appreciated"}
}


def parse_filename(filepath: str):
    """
    Parses ParticipantCode_YearLevel_Session_Word.wav
    Example: Y4-001_Y4_PM_Thrilled.WAV
    """
    filename = os.path.basename(filepath)
    stem, _ = os.path.splitext(filename)
    parts = stem.split("_")
    if len(parts) >= 4:
        participant_code = parts[0]
        year_level = parts[1]
        session = parts[2]
        spoken_word = parts[3]
    else:
        participant_code = stem
        year_level = "Unknown"
        session = "Unknown"
        spoken_word = "Unknown"
    
    tax = AFFECT_TAXONOMY.get(spoken_word, {
        "category": "Unknown",
        "valence_group": "Unknown",
        "arousal_group": "Unknown",
        "english": spoken_word
    })
    
    return {
        "participant_code": participant_code,
        "year_level": year_level,
        "session": session,
        "spoken_word": spoken_word,
        "affect_category": tax["category"],
        "valence_group": tax["valence_group"],
        "arousal_group": tax["arousal_group"],
        "word_english": tax["english"],
        "audio_filename": filename
    }


def extract_audio_features_from_file(filepath: str, target_sr: int = 16000) -> dict:
    """
    Extracts acoustic and prosodic features from a single WAV audio recording.
    """
    meta = parse_filename(filepath)
    
    # Load audio (downsampled to target_sr for consistent SER processing)
    y, sr = librosa.load(filepath, sr=target_sr, mono=True)
    
    # Remove leading/trailing silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    if len(y_trimmed) < 160: # fallback if trimmed too much
        y_trimmed = y
        
    duration = float(len(y) / sr)
    trimmed_duration = float(len(y_trimmed) / sr)
    
    features = {
        **meta,
        "audio_duration_sec": round(duration, 4),
        "audio_trimmed_dur_sec": round(trimmed_duration, 4),
        "audio_sampling_rate": sr,
    }
    
    # 1. Pitch / Fundamental Frequency (F0) using pyin
    # Human speech F0 typically between 50 Hz and 500 Hz
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y_trimmed,
        fmin=50,
        fmax=500,
        sr=sr,
        frame_length=1024,
        hop_length=256
    )
    
    voiced_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    voiced_ratio = float(len(voiced_f0) / len(f0)) if len(f0) > 0 else 0.0
    
    if len(voiced_f0) > 0:
        f0_mean = float(np.mean(voiced_f0))
        f0_std = float(np.std(voiced_f0))
        f0_min = float(np.min(voiced_f0))
        f0_max = float(np.max(voiced_f0))
        f0_median = float(np.median(voiced_f0))
        f0_range = float(f0_max - f0_min)
        
        # Local Jitter approximation (relative pitch period variation)
        if len(voiced_f0) > 1:
            periods = 1.0 / voiced_f0
            diff_periods = np.abs(np.diff(periods))
            jitter_local = float(np.mean(diff_periods) / np.mean(periods))
        else:
            jitter_local = 0.0
    else:
        f0_mean = 0.0
        f0_std = 0.0
        f0_min = 0.0
        f0_max = 0.0
        f0_median = 0.0
        f0_range = 0.0
        jitter_local = 0.0
        
    features.update({
        "audio_f0_mean_hz": round(f0_mean, 2),
        "audio_f0_std_hz": round(f0_std, 2),
        "audio_f0_min_hz": round(f0_min, 2),
        "audio_f0_max_hz": round(f0_max, 2),
        "audio_f0_median_hz": round(f0_median, 2),
        "audio_f0_range_hz": round(f0_range, 2),
        "audio_voiced_ratio": round(voiced_ratio, 4),
        "audio_jitter_local": round(jitter_local, 5),
    })
    
    # 2. Energy / Loudness (RMS)
    rms = librosa.feature.rms(y=y_trimmed, frame_length=1024, hop_length=256)[0]
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))
    rms_max = float(np.max(rms))
    rms_min = float(np.min(rms))
    
    # Local Shimmer approximation (relative amplitude difference between frames)
    if len(rms) > 1 and rms_mean > 1e-6:
        diff_rms = np.abs(np.diff(rms))
        shimmer_local = float(np.mean(diff_rms) / rms_mean)
    else:
        shimmer_local = 0.0
        
    features.update({
        "audio_rms_mean": round(rms_mean, 5),
        "audio_rms_std": round(rms_std, 5),
        "audio_rms_max": round(rms_max, 5),
        "audio_rms_min": round(rms_min, 5),
        "audio_shimmer_local": round(shimmer_local, 5),
    })
    
    # 3. Zero Crossing Rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y=y_trimmed, frame_length=1024, hop_length=256)[0]
    features.update({
        "audio_zcr_mean": round(float(np.mean(zcr)), 5),
        "audio_zcr_std": round(float(np.std(zcr)), 5),
        "audio_zcr_max": round(float(np.max(zcr)), 5),
    })
    
    # 4. Spectral Features
    sc = librosa.feature.spectral_centroid(y=y_trimmed, sr=sr, n_fft=1024, hop_length=256)[0]
    sb = librosa.feature.spectral_bandwidth(y=y_trimmed, sr=sr, n_fft=1024, hop_length=256)[0]
    srolloff = librosa.feature.spectral_rolloff(y=y_trimmed, sr=sr, roll_percent=0.85, n_fft=1024, hop_length=256)[0]
    sflatness = librosa.feature.spectral_flatness(y=y_trimmed, n_fft=1024, hop_length=256)[0]
    scontrast = librosa.feature.spectral_contrast(y=y_trimmed, sr=sr, n_fft=1024, hop_length=256)
    
    features.update({
        "audio_spec_centroid_mean": round(float(np.mean(sc)), 2),
        "audio_spec_centroid_std": round(float(np.std(sc)), 2),
        "audio_spec_bandwidth_mean": round(float(np.mean(sb)), 2),
        "audio_spec_bandwidth_std": round(float(np.std(sb)), 2),
        "audio_spec_rolloff85_mean": round(float(np.mean(srolloff)), 2),
        "audio_spec_rolloff85_std": round(float(np.std(srolloff)), 2),
        "audio_spec_flatness_mean": round(float(np.mean(sflatness)), 6),
        "audio_spec_flatness_std": round(float(np.std(sflatness)), 6),
        "audio_spec_contrast_mean": round(float(np.mean(scontrast)), 4),
        "audio_spec_contrast_std": round(float(np.std(scontrast)), 4),
    })
    
    # 5. Chroma features
    chroma = librosa.feature.chroma_stft(y=y_trimmed, sr=sr, n_fft=1024, hop_length=256)
    features.update({
        "audio_chroma_mean": round(float(np.mean(chroma)), 4),
        "audio_chroma_std": round(float(np.std(chroma)), 4),
    })
    
    # 6. MFCCs, Delta MFCCs, Delta-Delta MFCCs (13 coefficients each)
    n_mfcc = 13
    mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=n_mfcc, n_fft=1024, hop_length=256)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    
    for i in range(n_mfcc):
        features[f"audio_mfcc_{i+1}_mean"] = round(float(np.mean(mfcc[i])), 4)
        features[f"audio_mfcc_{i+1}_std"] = round(float(np.std(mfcc[i])), 4)
        
        features[f"audio_delta_mfcc_{i+1}_mean"] = round(float(np.mean(delta_mfcc[i])), 4)
        features[f"audio_delta_mfcc_{i+1}_std"] = round(float(np.std(delta_mfcc[i])), 4)
        
        features[f"audio_delta2_mfcc_{i+1}_mean"] = round(float(np.mean(delta2_mfcc[i])), 4)
        features[f"audio_delta2_mfcc_{i+1}_std"] = round(float(np.std(delta2_mfcc[i])), 4)
        
    return features


def extract_all_audio_features(raw_audio_dir: str, output_csv: str = None) -> pd.DataFrame:
    """
    Extracts acoustic features from all WAV files in raw_audio_dir.
    """
    raw_audio_dir = os.path.abspath(raw_audio_dir)
    files = sorted([
        os.path.join(raw_audio_dir, f)
        for f in os.listdir(raw_audio_dir)
        if f.lower().endswith((".wav"))
    ])
    
    print(f"Found {len(files)} audio recordings in {raw_audio_dir}")
    records = []
    for filepath in tqdm(files, desc="Extracting Audio Features"):
        try:
            feat = extract_audio_features_from_file(filepath)
            records.append(feat)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            
    df = pd.DataFrame(records)
    
    if output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Audio feature dataset saved to {output_csv} (Shape: {df.shape})")
        
    return df


if __name__ == "__main__":
    audio_dir = "DATA/2AudioRecordings/Raw .wav"
    out_csv = "datasets/audio_feature_dataset.csv"
    extract_all_audio_features(audio_dir, out_csv)
