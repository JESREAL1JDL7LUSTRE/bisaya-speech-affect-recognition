"""Validated acoustic feature extraction for the Bisaya affect dataset."""

from __future__ import annotations

# --- Standard Library Imports ---
import os                    # Operating system utilities (path checking, directory walking)
from pathlib import Path     # Object-oriented filesystem path manipulation

# --- Scientific & Signal Processing Libraries ---
import librosa               # Audio and speech signal processing (pYIN, STFT, RMS, ZCR, MFCC)
import numpy as np           # Numerical linear algebra and vectorized array operations
import pandas as pd          # Tabular dataset manipulation and CSV export
from tqdm import tqdm        # Progress bar for batch feature extraction

# --- File Paths for Data and Metadata Audit Trails ---
# Primary survey responses containing participant demographics, Likert ratings, and filenames
SURVEY_CSV = "DATA/RESEARCH MINI-PROJECT v2 - G7 - 4th year.csv"
# Provenance audit log documenting data provenance, corrections, and entry verification status
PROVENANCE_CSV = "DATA/metadata_provenance.csv"
# Audio quality flags documenting background music overlap and speech completeness
QUALITY_FLAGS_CSV = "DATA/recording_quality_flags.csv"

# --- Lexical Word Taxonomy Dictionary ---
# Maps each spoken Bisaya word to an academic lexical category and its English translation.
# CRITICAL RESEARCH DEFENSE POINT:
# The spoken word represents LEXICAL CONTEXT ONLY. It is NEVER used as the emotion target outcome!
# Emotional ground truth is strictly derived from the participant's independent self-reported ratings.
LEXICAL_TAXONOMY = {
    "Kapoy": ("Fatigue-related word", "Tired / Exhausted"),
    "Hangak": ("Fatigue-related word", "Breathless / Gasping"),
    "Labad": ("Fatigue-related word", "Headache / Stressed"),
    "Mamatay": ("Fatigue-related word", "Dying / Drained"),
    "Lutang": ("Fatigue-related word", "Spaced out / Floating"),
    "Kulba": ("Stress-related word", "Nervous / Anxious"),
    "Kapuliki": ("Stress-related word", "Overwhelmed / Frantic"),
    "Lisod": ("Stress-related word", "Difficult / Challenging"),
    "Libog": ("Stress-related word", "Confused / Perplexed"),
    "Gaduha-duha": ("Stress-related word", "Hesitant / In doubt"),
    "Duha-duha": ("Stress-related word", "Hesitant / In doubt"),
    "Pildi": ("Stress-related word", "Defeated / Lost"),
    "Gakaguol": ("Stress-related word", "Grieving / Gloomy"),
    "Kaguol": ("Stress-related word", "Sad / Sorrowful"),
    "Okay": ("Neutral/ambivalent word", "Okay / Fine"),
    "Ambot": ("Neutral/ambivalent word", "I don't know / Ambivalent"),
    "Hilom": ("Neutral/ambivalent word", "Quiet / Silent"),
    "Kamatuoran": ("Neutral/ambivalent word", "Truth / Acceptance"),
    "Hapsay": ("Positive/relief word", "Smooth / Orderly"),
    "Nahuwasan": ("Positive/relief word", "Relieved"),
    "Relibo": ("Positive/relief word", "Relieved"),
    "Lingaw": ("Positive/relief word", "Fun / Enjoyable"),
    "Chuy": ("Positive/relief word", "Cool / Chill"),
    "Thrilled": ("Positive/relief word", "Excited / Thrilled"),
    "Gihigugma": ("Positive/relief word", "Loved / Appreciated"),
}

# Structured dictionary representation of the lexical taxonomy for programmatic lookups
AFFECT_TAXONOMY = {
    word: {"category": category, "english": english}
    for word, (category, english) in LEXICAL_TAXONOMY.items()
}

# Alias mapping: handles minor spelling variations between filenames and survey entries
FILENAME_WORD_ALIASES = {"Gaduha-duha": "Duha-duha"}


def _read_csv(path: str) -> pd.DataFrame:
    """
    Robust CSV reader that handles Windows encoding variations.
    Tries UTF-8 first; if a UnicodeDecodeError occurs, falls back to CP1252 (Windows Latin-1).
    """
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252")


def _parse_rating(value, field: str) -> int:
    """
    Parses survey Likert ratings (e.g. '4 - Pleasant' or 4) into a strict integer from 1 to 5.
    DEFENSE POINT: Throws ValueError immediately if a rating is blank or outside [1, 5],
    ensuring zero corrupted or fabricated labels enter the system.
    """
    text = "" if pd.isna(value) else str(value).strip()
    if not text or not text[0].isdigit():
        raise ValueError(f"{field} must begin with an integer from 1 to 5; got {value!r}")
    score = int(text[0])
    if score not in range(1, 6):
        raise ValueError(f"{field} must be from 1 to 5; got {score}")
    return score


def _rating_group(score: int, dimension: str) -> str:
    """
    EXACT LINE OF TARGET RECLASSIFICATION (Hans' Question):
    Partitions the continuous 1-5 Likert response space into 3 balanced categorical target buckets:
    
    VALENCE:
      - Score 1 or 2 -> 'Negative' (Scores 1: Very Unpleasant, 2: Unpleasant)
      - Score 3      -> 'Neutral'  (Score 3: Neutral)
      - Score 4 or 5 -> 'Positive' (Scores 4: Pleasant, 5: Very Pleasant)
      
    AROUSAL:
      - Score 1 or 2 -> 'Low'      (Scores 1: Very Low, 2: Low physiological energy)
      - Score 3      -> 'Moderate' (Score 3: Moderate energy / modal baseline)
      - Score 4 or 5 -> 'High'     (Scores 4: High, 5: Very High activation / stress)
    """
    if dimension == "valence":
        return "Negative" if score <= 2 else "Neutral" if score == 3 else "Positive"
    return "Low" if score <= 2 else "Moderate" if score == 3 else "High"


def _load_auxiliary_map(path: str, value_columns: list[str]) -> dict[str, dict]:
    """
    Helper function to load provenance or quality flag tables into a fast lookup dictionary.
    Returns: {participant_code: {col_name: value}}
    """
    if not os.path.exists(path):
        return {}  # If audit file is absent, return empty mapping
    df = _read_csv(path)  # Read audit CSV with robust encoding
    if "Participant Code" not in df.columns:
        raise ValueError(f"{path} is missing Participant Code")  # Ensure foreign key exists
    return {
        # Key: Participant Code string (e.g. 'Y4-001')
        str(row["Participant Code"]).strip(): {
            # Value: dictionary of audit attributes (e.g. 'Intentional BGM Boundary Trim')
            col: ("" if pd.isna(row.get(col)) else str(row.get(col)).strip())
            for col in value_columns
        }
        for _, row in df.iterrows()
    }


def load_survey_metadata(csv_path: str = SURVEY_CSV) -> dict[str, dict]:
    """
    Loads, sanitizes, and verifies the 40 participant survey metadata records.
    
    DEFENSE POINTS:
    1. Zero Data Fabrication: Missing text entries remain NaN; ratings must be valid 1-5 integers.
    2. Strict Ethical Consent: All 40 participants must have verified informed consent ('Yes').
    3. Ground-Truth Construction: Valence and Arousal are mapped into analytical bins (Negative/Neutral/Positive).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Required survey metadata not found: {csv_path}")
        
    df = _read_csv(csv_path)  # Read primary survey CSV
    
    # 1. Enforce strict column schema requirements
    required = {
        "Participant Code", "Year Level", "Class Activity", "Subject", "Session",
        "Spoken Word", "Self-Reported Feeling", "Valence (1-5)", "Arousal (1-5)",
        "Audio Filename", "Consent Obtained (Y/N)", "Date Collected",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Survey metadata is missing columns: {missing}")
        
    # 2. Check for empty or duplicate participant codes
    codes = df["Participant Code"].astype(str).str.strip()
    if codes.eq("").any() or codes.duplicated().any():
        duplicates = codes[codes.duplicated(keep=False)].tolist()
        raise ValueError(f"Participant codes must be nonblank and unique; duplicates={duplicates}")

    # 3. Load supplementary audit trails (provenance of text notes & recording acoustic flags)
    provenance = _load_auxiliary_map(PROVENANCE_CSV, ["Field", "Status", "Note"])
    quality = _load_auxiliary_map(
        QUALITY_FLAGS_CSV,
        ["Intentional BGM Boundary Trim", "Background Music Overlap", "Speech Complete", "Notes"],
    )
    
    # Textual Likert descriptors for human-readable logging
    val_labels = {1: "Very Unpleasant", 2: "Unpleasant", 3: "Neutral", 4: "Pleasant", 5: "Very Pleasant"}
    aro_labels = {1: "Very Low", 2: "Low", 3: "Moderate", 4: "High", 5: "Very High"}
    
    records: dict[str, dict] = {}
    
    # 4. Iterate and validate each participant row
    for _, row in df.iterrows():
        code = str(row["Participant Code"]).strip()
        
        # Verify Informed Consent - Ethics requirement (CS412 PIT requirement)
        if str(row["Consent Obtained (Y/N)"]).strip().lower() not in {"yes", "y"}:
            raise ValueError(f"{code} does not have recorded consent")
            
        # Parse numerical ratings (1 to 5)
        valence = _parse_rating(row["Valence (1-5)"], f"{code} valence")
        arousal = _parse_rating(row["Arousal (1-5)"], f"{code} arousal")
        
        # Parse self-reported qualitative text
        feeling = "" if pd.isna(row["Self-Reported Feeling"]) else str(row["Self-Reported Feeling"]).strip()
        word = str(row["Spoken Word"]).strip()
        
        # Look up word taxonomy (Context only - not emotion outcome)
        lexical_category, word_english = LEXICAL_TAXONOMY.get(word, ("Unclassified word", word))
        
        # Segment ratings into 3 analytical categories (Negative/Neutral/Positive and Low/Moderate/High)
        v_group = _rating_group(valence, "valence")
        a_group = _rating_group(arousal, "arousal")
        
        prov = provenance.get(code, {})
        flags = quality.get(code, {})
        
        # Construct verified metadata record dictionary
        records[code] = {
            "participant_code": code,
            "year_level": str(row["Year Level"]).strip(),
            "class_activity": str(row["Class Activity"]).strip(),
            "subject": str(row["Subject"]).strip(),
            "session_name": str(row["Session"]).strip(),
            "survey_spoken_word": word,
            "self_reported_feeling": feeling if feeling else pd.NA,
            "self_reported_feeling_missing": not bool(feeling),
            "self_reported_feeling_source": prov.get("Status", "Survey entry"),
            "valence_score": valence,
            "valence_label": val_labels[valence],
            "valence_group": v_group,                          # Target 1: 3-Class Valence
            "arousal_score": arousal,
            "arousal_label": aro_labels[arousal],
            "arousal_group": a_group,                          # Target 2: 3-Class Arousal
            "affect_category": f"{v_group}-{a_group}",          # Compound 2D Affect Quadrant
            "lexical_category": lexical_category,
            "word_english": word_english,
            "date_collected": "" if pd.isna(row["Date Collected"]) else str(row["Date Collected"]).strip(),
            "consent_obtained": True,
            "survey_audio_filename": str(row["Audio Filename"]).strip(),
            "intentional_bgm_boundary_trim": flags.get("Intentional BGM Boundary Trim", "No"),
            "background_music_overlap": flags.get("Background Music Overlap", "Not flagged"),
            "speech_complete_status": flags.get("Speech Complete", "Not flagged"),
        }
    return records


# Preload and validate all survey metadata at module import time
SURVEY_METADATA = load_survey_metadata()


def parse_filename(filepath: str) -> dict:
    """
    Parses audio filename and enforces 1-to-1 consistency with survey metadata.
    
    NAMING SYNTAX:
    <ParticipantCode>_<YearLevel>_<Session>_<SpokenWord>.WAV
    Example: 'Y4-011_Y4_AM_Kapoy.WAV'
    
    DEFENSE POINT:
    Throws ValueError if the filename does not strictly correspond with its survey record,
    preventing any file-swapping or corrupted audio associations.
    """
    # Extract base filename from path (e.g., 'Y4-011_Y4_AM_Kapoy.WAV' from 'DATA/.../Y4-011_Y4_AM_Kapoy.WAV')
    filename = os.path.basename(filepath)
    # Strip .WAV extension using .stem, then split on underscore delimiter: ['Y4-011', 'Y4', 'AM', 'Kapoy']
    parts = Path(filename).stem.split("_")
    
    # Must have at least 4 tokens: Participant Code, Year Level, Session, and Spoken Word
    if len(parts) < 4:
        raise ValueError(f"Invalid recording filename: {filename}")
        
    # Unpack the first 3 tokens: 'Y4-011' (participant), 'Y4' (year), 'AM' (session)
    participant_code, year_code, session_code = parts[:3]
    # Rejoin any remaining tokens in case the word itself had underscores (e.g. 'Duha_duha')
    spoken_word = "_".join(parts[3:])
    
    # 1. Verify participant code exists in the survey metadata registry
    if participant_code not in SURVEY_METADATA:
        raise ValueError(f"No survey row for {participant_code}")
        
    # Make a shallow copy of the survey metadata dictionary to avoid mutating the master cache
    survey = SURVEY_METADATA[participant_code].copy()
    # Pop the self-reported spoken word recorded by the student in the survey
    survey_word = survey.pop("survey_spoken_word")
    # Resolve aliases (e.g., mapping 'Gaduha-duha' to canonical 'Duha-duha')
    canonical_filename_word = FILENAME_WORD_ALIASES.get(spoken_word, spoken_word)
    
    # 2. Case-insensitive comparison confirming the word spoken in the audio file matches the survey entry
    word_matches = canonical_filename_word.casefold() == survey_word.casefold()
    if not word_matches:
        raise ValueError(f"{participant_code}: filename word {spoken_word!r} != survey word {survey_word!r}")
        
    # 3. Verify the file stem on disk matches the exact Audio Filename documented in the survey CSV
    if Path(survey["survey_audio_filename"]).stem.casefold() != Path(filename).stem.casefold():
        raise ValueError(f"{participant_code}: filename does not match survey Audio Filename")
        
    # 4. Verify session code (AM vs PM) matches survey class schedule (Morning -> AM, Afternoon/Evening -> PM)
    expected_session = "AM" if survey["session_name"].lower().startswith("morning") else "PM"
    if session_code != expected_session:
        raise ValueError(f"{participant_code}: filename session {session_code} != survey session {survey['session_name']}")
        
    # Reconciled metadata ready for feature association with validated audit properties
    survey.update({
        "participant_code": participant_code,
        "year_code": year_code,
        "session": session_code,
        "spoken_word": canonical_filename_word,
        "metadata_word_matches_filename": word_matches,
        "audio_filename": filename,
    })
    return survey


def _round_or_nan(value: float, digits: int) -> float:
    """Helper: rounds finite floats; preserves NaN for unvoiced frames or undefined values."""
    return round(float(value), digits) if np.isfinite(value) else np.nan


def extract_audio_features_from_file(filepath: str, target_sr: int = 16000) -> dict:
    """
    Extracts 108 acoustic, prosodic, spectral, and timbral features from a single .WAV file.
    
    PIPELINE OVERVIEW:
    1. Parse Filename & Survey Metadata -> Link audio to participant demographics and self-reported ground truth.
    2. Resample & Load -> Convert audio to standard 16,000 Hz mono (telephony & SER benchmark standard).
    3. Silence Trimming -> librosa.effects.trim(top_db=25) removes non-speech microphone silence without clipping phonemes.
    4. Boundary Check -> Inspect final 50 ms to detect abrupt cutoffs or background music/noise bleed.
    5. Prosody / Pitch (pYIN) -> Probabilistic YIN tracks F0 between 50-500 Hz; unvoiced frames kept as NaN (not 0 Hz).
    6. Energy / Intensity (RMS) -> Moving window root-mean-square loudness mean, std, min, max, and jitter proxy.
    7. Spectral Dynamics -> Zero-crossing rate, spectral centroid (brightness), bandwidth, rolloff, flatness, contrast, chroma.
    8. Timbre (MFCC) -> 13 Mel-frequency cepstral coefficients + First Derivative (Delta/Velocity) + Second Derivative (Delta2/Acceleration).
    """
    # [STEP 1] Reconcile physical file with survey entry (participant code, session, word, valence, arousal)
    meta = parse_filename(filepath)

    # [STEP 2] Load audio file: resampled to 16 kHz mono (Nyquist = 8 kHz, perfectly capturing speech formants)
    y, sr = librosa.load(filepath, sr=target_sr, mono=True)
    if y.size < 512 or not np.isfinite(y).all():
        raise ValueError("Audio is too short or contains non-finite samples")

    # [STEP 3] Silence Trimming: removes leading/trailing room silence quieter than 25 dB below peak amplitude
    # DEFENSE POINT: top_db=25 preserves soft consonants (like /k/ in 'Kapoy' or /s/ in 'Lisod') while discarding mic noise
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    if y_trimmed.size < 1024:
        y_trimmed = y  # Fallback: if utterance was extremely quiet, keep original signal
    duration = len(y) / sr
    trimmed_duration = len(y_trimmed) / sr

    # [STEP 4] Boundary Check: inspect the last 50 ms (0.05 * 16000 = 800 samples)
    # DEFENSE POINT: flags recordings that abruptly cut off during vocalization or contain trailing background music
    end_window = max(1, int(0.05 * sr))
    chunks = [y[i:i + end_window] for i in range(0, len(y), end_window)]
    window_rms = [np.sqrt(np.mean(chunk ** 2)) for chunk in chunks]
    end_rms_ratio = float(np.sqrt(np.mean(y[-end_window:] ** 2)) / (max(window_rms) + 1e-12))

    features = {
        **meta,
        "audio_duration_sec": round(duration, 4),                     # Total raw duration in seconds
        "audio_trimmed_dur_sec": round(trimmed_duration, 4),           # Active vocal duration after trimming
        "audio_trimmed_fraction": round(trimmed_duration / duration, 4), # Percentage of recording containing vocal sound
        "audio_sampling_rate": sr,                                     # 16000 Hz
        "audio_end_boundary_rms_ratio": round(end_rms_ratio, 4),       # Ratio of end loudness to maximum loudness
        "audio_boundary_review_flag": bool(end_rms_ratio > 0.5),       # True if recording ends abruptly at >50% peak volume
    }

    # [STEP 5] Prosodic / Fundamental Frequency (F0) Extraction using pYIN (Probabilistic YIN algorithm)
    # DEFENSE POINT: fmin=50 Hz, fmax=500 Hz spans all human vocal pitch while ignoring 60 Hz hum and consonant hiss
    # STFT parameters: frame_length=2048 (128 ms resolution), hop_length=256 (16 ms step size)
    frame_length, hop_length = 2048, 256
    f0, _, _ = librosa.pyin(
        y_trimmed, fmin=50, fmax=500, sr=sr,
        frame_length=frame_length, hop_length=hop_length,
    )
    
    # Isolate voiced frames where vocal cords actively vibrate (finite numbers) vs unvoiced frames (NaN)
    valid = np.isfinite(f0) if f0 is not None else np.zeros(0, dtype=bool)
    voiced_f0 = f0[valid] if f0 is not None else np.array([])
    pitch_detected = voiced_f0.size > 0
    adjacent = valid[1:] & valid[:-1] if valid.size > 1 else np.zeros(0, dtype=bool)

    if pitch_detected:
        # Period variation proxy: measures pitch period jitter (relative frame-to-frame pitch instability)
        periods = 1.0 / f0
        period_variation = (
            float(np.mean(np.abs(np.diff(periods)[adjacent])) / np.mean(periods[valid]))
            if adjacent.any() else np.nan
        )
        # DEFENSE POINT: pitch statistics calculated ONLY on valid voiced frames
        # [mean, std, min, max, median, range/peak-to-peak]
        pitch_values = [
            np.mean(voiced_f0), np.std(voiced_f0), np.min(voiced_f0),
            np.max(voiced_f0), np.median(voiced_f0), np.ptp(voiced_f0)
        ]
    else:
        # DEFENSE POINT: if unvoiced, preserve as NaN rather than 0 Hz to avoid distorting mean and variance
        period_variation = np.nan
        pitch_values = [np.nan] * 6

    features.update({
        "audio_pitch_detected": bool(pitch_detected),                  # True if vocal cord vibration was detected
        "audio_voiced_frames": int(valid.sum()),                       # Total count of periodic speech frames
        "audio_pitch_total_frames": int(valid.size),                   # Total evaluated audio frames
        "audio_pitch_frame_length": frame_length,                      # 2048 samples
        "audio_f0_mean_hz": _round_or_nan(pitch_values[0], 2),         # Average fundamental frequency (Pitch level)
        "audio_f0_std_hz": _round_or_nan(pitch_values[1], 2),          # Pitch variability (Intonation contour)
        "audio_f0_min_hz": _round_or_nan(pitch_values[2], 2),          # Lowest vocal pitch
        "audio_f0_max_hz": _round_or_nan(pitch_values[3], 2),          # Highest vocal pitch
        "audio_f0_median_hz": _round_or_nan(pitch_values[4], 2),       # Median pitch (robust to octave jumps)
        "audio_f0_range_hz": _round_or_nan(pitch_values[5], 2),        # Pitch range (f0_max - f0_min)
        "audio_voiced_ratio": round(float(valid.mean()), 4) if valid.size else 0.0, # Fraction of speech that is voiced (vowels)
        "audio_period_variation_proxy": _round_or_nan(period_variation, 5),          # Pitch period jitter proxy
    })

    # [STEP 6] Energy & Loudness Dynamics via Root Mean Square (RMS)
    # Formula: RMS = sqrt(mean(x^2)) per 1024-sample frame (64 ms window, 16 ms hop)
    # DEFENSE POINT: High Arousal (stress, panic, excitement) causes high lung pressure -> directly increases RMS
    rms = librosa.feature.rms(y=y_trimmed, frame_length=1024, hop_length=hop_length)[0]
    rms_mean = float(np.mean(rms))
    # Frame-to-frame RMS variation proxy: measures loudness fluctuation / tremolo in voice
    rms_variation = float(np.mean(np.abs(np.diff(rms))) / rms_mean) if len(rms) > 1 and rms_mean > 1e-8 else np.nan
    features.update({
        "audio_rms_mean": round(rms_mean, 5),                          # Overall perceived speech loudness
        "audio_rms_std": round(float(np.std(rms)), 5),                 # Dynamic energy variation
        "audio_rms_max": round(float(np.max(rms)), 5),                 # Peak vocal burst energy
        "audio_rms_min": round(float(np.min(rms)), 5),                 # Minimum energy floor
        "audio_rms_frame_variation_proxy": _round_or_nan(rms_variation, 5), # Energy jitter / frame instability
    })

    # [STEP 7] Spectral Dynamics & Timbral Descriptors
    # Frame parameters: window=1024 samples, hop=256 samples
    zcr = librosa.feature.zero_crossing_rate(y=y_trimmed, frame_length=1024, hop_length=hop_length)[0] # Rate of sign-changes (articulatory speed)
    sc = librosa.feature.spectral_centroid(y=y_trimmed, sr=sr, n_fft=1024, hop_length=hop_length)[0]   # Spectral center of mass (timbral brightness/sharpness)
    sb = librosa.feature.spectral_bandwidth(y=y_trimmed, sr=sr, n_fft=1024, hop_length=hop_length)[0]  # Spectral spread around centroid
    rolloff = librosa.feature.spectral_rolloff(y=y_trimmed, sr=sr, roll_percent=0.85, n_fft=1024, hop_length=hop_length)[0] # Frequency below which 85% energy lies
    flatness = librosa.feature.spectral_flatness(y=y_trimmed, n_fft=1024, hop_length=hop_length)[0]    # Tonality vs noiselike quality (geometric vs arithmetic mean)
    contrast = librosa.feature.spectral_contrast(y=y_trimmed, sr=sr, n_fft=1024, hop_length=hop_length) # dB difference between spectral peaks and valleys
    chroma = librosa.feature.chroma_stft(y=y_trimmed, sr=sr, n_fft=1024, hop_length=hop_length)        # 12 semitone energy distribution

    # Aggregate each spectral feature across time frames (Mean and Standard Deviation)
    for name, arr, digits in [
        ("audio_zcr", zcr, 5), ("audio_spec_centroid", sc, 2),
        ("audio_spec_bandwidth", sb, 2), ("audio_spec_rolloff85", rolloff, 2),
        ("audio_spec_flatness", flatness, 6), ("audio_spec_contrast", contrast, 4),
        ("audio_chroma", chroma, 4),
    ]:
        features[f"{name}_mean"] = round(float(np.mean(arr)), digits)
        features[f"{name}_std"] = round(float(np.std(arr)), digits)
        if name == "audio_zcr":
            features[f"{name}_max"] = round(float(np.max(arr)), digits) # Maximum articulation burst speed

    # [STEP 8] Mel-Frequency Cepstral Coefficients (MFCCs) & Temporal Derivatives
    # Extracts 13 static MFCCs modeling human auditory perception of vocal tract filtering
    n_mfcc = 13
    mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=n_mfcc, n_fft=1024, hop_length=hop_length)
    frame_count = mfcc.shape[1]
    
    # DEFENSE POINT: Dynamic delta width prevents crash on short words (<0.5 sec)
    delta_width = min(9, frame_count if frame_count % 2 else frame_count - 1)
    delta_available = delta_width >= 3
    if delta_available:
        # Delta MFCC: Velocity (1st derivative = rate of timbre transition)
        delta_mfcc = librosa.feature.delta(mfcc, width=delta_width, mode="interp")
        # Delta-Delta MFCC: Acceleration (2nd derivative = vocal tract physical acceleration)
        delta2_mfcc = librosa.feature.delta(mfcc, width=delta_width, order=2, mode="interp")
    else:
        delta_mfcc = np.full_like(mfcc, np.nan)
        delta2_mfcc = np.full_like(mfcc, np.nan)
        
    features["audio_delta_available"] = bool(delta_available)
    features["audio_delta_width"] = int(delta_width) if delta_available else pd.NA

    # Summarize all 13 MFCCs, 13 Deltas, and 13 Delta2s across frames (Mean and Std = 78 total features)
    for i in range(n_mfcc):
        for prefix, values in [("audio_mfcc", mfcc), ("audio_delta_mfcc", delta_mfcc), ("audio_delta2_mfcc", delta2_mfcc)]:
            finite = np.isfinite(values[i])
            features[f"{prefix}_{i + 1}_mean"] = _round_or_nan(np.mean(values[i][finite]), 4) if finite.any() else np.nan
            features[f"{prefix}_{i + 1}_std"] = _round_or_nan(np.std(values[i][finite]), 4) if finite.any() else np.nan

    return features


def extract_all_audio_features(raw_audio_dir: str, output_csv: str | None = None) -> pd.DataFrame:
    """
    Batch feature extraction orchestrator:
    Iterates across all 40 raw .WAV files on disk, processes each file,
    collects extracted feature row vectors, and exports the final dataset to CSV.
    """
    raw_dir = Path(raw_audio_dir).resolve()
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {raw_dir}")
        
    # Gather and sort all .wav audio files alphabetically (deterministic ordering)
    files = sorted(p for p in raw_dir.iterdir() if p.suffix.lower() == ".wav")
    print(f"Found {len(files)} audio recordings in {raw_dir}")
    
    records, failures = [], []
    
    # Progress bar iteration across all recordings using tqdm
    for filepath in tqdm(files, desc="Extracting Audio Features"):
        try:
            # Extract 108 acoustic predictors + metadata for this recording
            records.append(extract_audio_features_from_file(str(filepath)))
        except Exception as exc:
            # Audit log any corrupted audio files without silently dropping them
            failures.append({"filename": filepath.name, "error": str(exc)})
            
    # Assemble master DataFrame (40 rows x 146 columns)
    df = pd.DataFrame(records)
    
    # Save to disk if an output CSV destination path was specified
    if output_csv:
        output = Path(output_csv)
        output.parent.mkdir(parents=True, exist_ok=True)  # Create datasets/audio directory if needed
        df.to_csv(output, index=False)                    # Save master acoustic dataset
        # Save extraction error log (empty if 100% successful)
        pd.DataFrame(failures, columns=["filename", "error"]).to_csv(output.parent / "audio_extraction_failures.csv", index=False)
        print(f"Audio feature dataset saved to {output} (Shape: {df.shape}; failures={len(failures)})")
        
    # Throws error if any recording failed extraction (maintains data integrity)
    if failures:
        raise RuntimeError(f"Audio extraction failed for {len(failures)} files; see audio_extraction_failures.csv")
        
    return df


# Script entry point: runs full extraction when invoked directly from command line
if __name__ == "__main__":
    # Extracts all features from DATA/2AudioRecordings/Raw .wav and saves to datasets/audio/audio_feature_dataset.csv
    extract_all_audio_features("DATA/2AudioRecordings/Raw .wav", "datasets/audio/audio_feature_dataset.csv")
