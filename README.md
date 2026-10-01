# Bisaya Post-Class Speech and Affect Recognition Among Computer Science Students

> **Course:** CS412 – Affective Computing Research Mini-Project  
> **Cohort:** 4th-Year Computer Science Students (Group 7)  
> **Milestone:** Second Deliverables (September 28 – October 2, 2026)  
> **Presentation Target:** PIT Presentation (October 5 – 8, 2026)

---

## 🎯 Deliverables Status Overview

All components specified in the **Second Deliverables** have been fully generated, validated, and formatted:

| Deliverable Component | Output Artifact | Status | Description |
|---|---|:---:|---|
| **1. Audio Feature Dataset** | [`datasets/audio_feature_dataset.csv`](file:///d:/SCHOOL/affective/datasets/audio_feature_dataset.csv) | **Completed** | 40 observations, 119 acoustic & prosodic features (F0, RMS, ZCR, 13 MFCCs, Deltas, Delta-Deltas, Spectral Centroid, Bandwidth, Rolloff, Flatness, Contrast, Local Jitter & Shimmer) |
| **2. Facial Feature Dataset** | [`datasets/facial_feature_dataset.csv`](file:///d:/SCHOOL/affective/datasets/facial_feature_dataset.csv) | **Completed** | 40 observations, 200 vision features (52 FACS Blendshapes [mean, std, max], Eye Aspect Ratio [EAR], Mouth Aspect Ratio [MAR], Head Pose [Pitch, Yaw, Roll], Expressiveness) |
| **3. Cleaned Multimodal Dataset** | [`datasets/cleaned_multimodal_dataset.csv`](file:///d:/SCHOOL/affective/datasets/cleaned_multimodal_dataset.csv)<br>[`datasets/cleaned_multimodal_dataset_scaled.csv`](file:///d:/SCHOOL/affective/datasets/cleaned_multimodal_dataset_scaled.csv) | **Completed** | 40 observations, 314 multimodal variables, 0 missing/null values, 0 infinite values. Includes standardized z-score dataset and data dictionary. |
| **4. Descriptive Statistics** | [`outputs/descriptive_statistics/descriptive_statistics_report.md`](file:///d:/SCHOOL/affective/outputs/descriptive_statistics/descriptive_statistics_report.md)<br>6 CSV Tables in `outputs/descriptive_statistics/` | **Completed** | Univariate statistics, AM vs. PM hypothesis testing (Welch's t-test, Mann-Whitney U, Cohen's d), Affect taxonomy breakdown, Lexically controlled analysis, Cross-modal correlation matrix. |
| **5. Tables & Visualizations** | 10 Publication-Grade Figures in [`outputs/visualizations/`](file:///d:/SCHOOL/affective/outputs/visualizations/) | **Completed** | 300 DPI high-resolution figures (Circumplex affect space, MFCC heatmaps, session boxplots, facial action units, cross-modal synchronization, PCA/t-SNE clustering, radar profiles). |

---

## 📁 Repository & Upload Folder Structure

```text
d:\SCHOOL\affective\
├── AudioDataset/                       # Matches portal upload folder
│   └── audio_feature_dataset.csv       # Extracted audio acoustic features (40 x 118)
├── FaceDataset/                        # Matches portal upload folder
│   └── facial_feature_dataset.csv      # Extracted facial vision features (40 x 199)
├── MultimodalDataset/                  # Matches portal upload folder
│   ├── cleaned_multimodal_dataset.csv  # THE TRUE CLEANED MULTIMODAL DATASET (40 x 312)
│   └── multimodal_data_dictionary.csv  # 312-entry data dictionary / codebook
│
├── outputs/
│   ├── descriptive_statistics/
│   │   ├── descriptive_statistics_report.md
│   │   ├── summary_statistics_audio.csv
│   │   ├── summary_statistics_facial.csv
│   │   ├── session_am_pm_comparison.csv
│   │   ├── affect_taxonomy_summary.csv
│   │   ├── lexically_controlled_kapoy.csv
│   │   └── audio_facial_correlations.csv
│   └── visualizations/                 # 10 High-Resolution (300 DPI) Figures
│       ├── 01_dataset_distribution_overview.png
│       ├── 02_affect_word_taxonomy.png
│       ├── 03_audio_prosodic_features_by_session.png
│       ├── 04_mfcc_feature_heatmaps.png
│       ├── 05_spectral_characteristics.png
│       ├── 06_facial_action_units_by_session.png
│       ├── 07_audio_facial_multimodal_correlations.png
│       ├── 08_multimodal_pca_tsne_clustering.png
│       ├── 09_lexically_controlled_analysis.png
│       └── 10_multimodal_radar_affect_profiles.png
│
├── DATA/                               # Raw recordings (Audio: Raw .wav / Video: Raw Cut Video)
├── run_pipeline.py                     # Master execution pipeline script
└── README.md
```

---

## 🔬 Extracted Feature Taxonomy

### 1. Acoustic and Prosodic Features (Speech Modality - 119 features)
- **Fundamental Frequency (Pitch / F0):** `audio_f0_mean_hz`, `audio_f0_std_hz`, `audio_f0_min_hz`, `audio_f0_max_hz`, `audio_f0_median_hz`, `audio_f0_range_hz`, `audio_voiced_ratio`.
- **Vocal Stability & Perturbation:** `audio_jitter_local` (relative pitch period instability), `audio_shimmer_local` (relative amplitude instability).
- **Energy & Intensity:** `audio_rms_mean`, `audio_rms_std`, `audio_rms_max`, `audio_rms_min`.
- **Temporal Dynamics:** `audio_zcr_mean`, `audio_zcr_std`, `audio_zcr_max`, `audio_duration_sec`, `audio_trimmed_dur_sec`.
- **Spectral Characteristics:** `audio_spec_centroid_mean` (brightness), `audio_spec_bandwidth_mean` (spread), `audio_spec_rolloff85_mean` (high-freq boundary), `audio_spec_flatness_mean` (tonality), `audio_spec_contrast_mean` (sub-band energy difference).
- **Timbral Envelope:** 13 Mel-Frequency Cepstral Coefficients (`audio_mfcc_1_mean` to `audio_mfcc_13_mean`), 13 first derivatives (`delta`), and 13 second derivatives (`delta2`).

### 2. Facial Action Units & Landmark Geometry (Vision Modality - 200 features)
- **FACS 52 Blendshapes:** Frame-by-frame activation aggregated across clips (`mean`, `std`, `max`) capturing Action Units (AU1, AU2, AU4, AU5, AU6, AU7, AU9, AU10, AU12, AU14, AU15, AU16, AU20, AU24, AU26, AU45, etc.).
- **Interpretable Action Unit Composites:**
  - `face_smile_mean`, `face_smile_max` (AU12 Lip Corner Puller - Joy/Amusement)
  - `face_frown_mean`, `face_frown_max` (AU15 Lip Corner Depressor - Sadness/Grief)
  - `face_brow_lowerer_mean`, `face_brow_lowerer_max` (AU4 Brow Lowerer - Concentration/Strain)
  - `face_brow_inner_raiser_mean` (AU1 Inner Brow Raiser - Distress/Worry)
  - `face_jaw_open_mean` (AU26/27 Jaw Open - Articulation dynamics)
  - `face_expressiveness_score` (Cross-blendshape dynamic standard deviation)
- **Geometric Landmark Ratios:**
  - `face_ear_mean`, `face_ear_min` (Eye Aspect Ratio - Eyelid openness)
  - `face_mar_mean`, `face_mar_max` (Mouth Aspect Ratio - Mouth vertical/horizontal ratio)
- **Head Pose Dynamics:** `face_pitch_mean_deg`, `face_yaw_mean_deg`, `face_roll_mean_deg` (Head nodding, turning, and tilting angles).

### 3. Cross-Modal Derived Features
- `multimodal_expressiveness_index`: Joint normalized composite of acoustic vocal energy and facial action unit movement.
- `face_valence_proxy`: Objective facial valence metric (`face_smile_mean` − `face_frown_mean`).
- `audio_arousal_proxy`: Objective vocal arousal metric combining normalized pitch (F0) and spectral brightness (centroid).

---

## 📊 Summary of Research Findings (Answering RQ1 - RQ5)

### RQ1: What affective states do CS students report immediately after class?
- **Cognitive & Physical Fatigue (37.5%):** Elicited via *Kapoy* (n=11), *Hangak*, *Labad*, *Mamatay*, *Lutang*. Dominant after intensive 4th-year coding and capstone classes.
- **Stress & Cognitive Strain (27.5%):** Elicited via *Lisod* (n=2), *Libog* (n=2), *Kulba* (n=2), *Kapuliki*, *Gaduha-duha*, *Pildi*, *Gakaguol*, *Kaguol*.
- **Neutral & Composed (15.0%):** Elicited via *Okay* (n=3), *Ambot*, *Hilom*, *Kamatuoran*.
- **Relief & Positive Affect (20.0%):** Elicited via *Hapsay* (n=2), *Nahuwasan*, *Relibo*, *Chuy*, *Thrilled*, *Lingaw*, *Gihigugma*.

### RQ2: How do affective states vary according to Class Session (AM vs. PM)?
- **Utterance Duration:** AM sessions had significantly longer spoken responses (Mean = 0.908s) than PM sessions (Mean = 0.712s) ($t = 4.70, p < 0.0001$, Cohen's $d = 0.98$).
- **Acoustic RMS Energy:** AM classes exhibited significantly higher speech loudness ($p < 0.0001, d = 1.83$).
- **Facial Smile Modulation:** PM classes showed higher smile intensity (Mean = 0.389 vs. 0.213), consistent with end-of-day class completion and relief.

### RQ3: What acoustic and visual characteristics distinguish affective states?
- **Pitch (F0):** Stress/Anxiety exhibited the highest fundamental frequency ($\mu = 155.9$ Hz), whereas Fatigue exhibited significantly lower, subdued pitch ($\mu = 68.9$ Hz).
- **Facial Action Units:** Positive responses exhibited highest smile activation ($\mu = 0.3231$), while Stress/Anxiety exhibited lowest smile ($\mu = 0.1259$) and elevated brow lowering (AU4).

### Section XVII: Lexically Controlled Analysis (*Kapoy*, n=11)
- Holding the spoken word constant to **"Kapoy"**, vocal pitch ranged from **0.0 Hz (creaky/unvoiced whisper)** to **240.2 Hz (high-pitched exclamation)**.
- Smile intensity ranged from **0.001 (flat exhaustion)** to **0.675 (ironic amusement / laughter)**.
- **Conclusion:** Confirms the core research hypothesis: **"The Bisaya word is not automatically the emotion. Prosodic and facial dynamics contain independent affective signals."**

---

## 🚀 Execution Instructions

### Prerequisites
- Windows OS with Python 3.11 installed.

### 1. Activating the Virtual Environment
Open PowerShell in the project directory:
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Running the Complete End-to-End Pipeline
To re-extract all features, compile the multimodal dataset, perform statistical hypothesis tests, and regenerate all 10 figures:
```powershell
.\.venv\Scripts\python.exe run_pipeline.py
```
*(Execution completes in ~30 seconds for all 40 participants across audio and video).*

### 3. Running Individual Modules
- **Extract Audio Only:** `.\.venv\Scripts\python.exe src/audio_features.py`
- **Extract Facial Only:** `.\.venv\Scripts\python.exe src/facial_features.py`
- **Build Multimodal Dataset:** `.\.venv\Scripts\python.exe src/multimodal_dataset.py`
- **Run Descriptive Statistics:** `.\.venv\Scripts\python.exe src/statistics_analysis.py`
- **Generate Visualizations:** `.\.venv\Scripts\python.exe src/visualizations.py`
