# Bisaya Post-Class Speech and Affect Recognition Among Computer Science Students

> **Course:** CS412 – Affective Computing Research Mini-Project  
> **Cohort:** 4th-Year Computer Science Students (Group 7)  
> **Milestone:** Second Deliverables (September 28 – October 2, 2026)  
> **Presentation Target:** PIT Presentation (October 5 – 8, 2026)

---

## 🎯 Deliverables & Project Scope Overview

> [!NOTE]
> **Instructor Guidance (October 2026):**  
> *"Base lang mo sa acoustic and prosodic features. Ayaw na pila ang sa facial input"*  
> In accordance with course instructor Ma'am Love Jhoye's guidance, the primary scope for the Midterm PIT modeling and presentation focuses strictly on **Acoustic and Prosodic Speech Emotion Recognition (Audio Modality)**.
> All facial action unit and multimodal cross-referencing features have been systematically preserved and categorized into dedicated subdirectories (`outputs/facial/` and `outputs/multimodal/`) for complete academic rigor, but all primary analytical tables and 10 publication figures for the PIT presentation are located in [`outputs/audio/`](file:///d:/SCHOOL/affective/outputs/audio/).

| Deliverable Component | Output Directory | Status | Description |
|---|---|:---:|---|
| **1. Audio Modality (PRIMARY FOCUS)** | [`outputs/audio/`](file:///d:/SCHOOL/affective/outputs/audio/) | **Completed** | 10 high-resolution (300 DPI) figures, 5 summary CSV tables, and descriptive markdown report covering acoustic & prosodic features, pitch dynamics, and session tests. |
| **2. Facial Modality (Vision)** | [`outputs/facial/`](file:///d:/SCHOOL/affective/outputs/facial/) | **Completed** | 5 high-resolution (300 DPI) figures and 1 summary CSV table covering MediaPipe FACS blendshapes, EAR/MAR, and head pose dynamics. |
| **3. Multimodal Analysis** | [`outputs/multimodal/`](file:///d:/SCHOOL/affective/outputs/multimodal/) | **Completed** | 4 high-resolution (300 DPI) figures and 4 summary CSV tables covering cross-modal correlations, joint embeddings, and lexically controlled *Kapoy* profiles. |
| **4. Clean Datasets** | [`datasets/`](file:///d:/SCHOOL/affective/datasets/) | **Completed** | Clean datasets categorized into `datasets/audio/`, `datasets/facial/`, and `datasets/multimodal/` (including 344-entry data dictionary). |

---

## 📁 Repository & Output Folder Structure

```text
d:\SCHOOL\affective\
├── datasets/
│   ├── audio/
│   │   ├── audio_feature_dataset.csv             # Extracted acoustic features (40 x 146)
│   │   └── audio_extraction_failures.csv         # Extraction failure log (0 failures)
│   ├── facial/
│   │   ├── facial_feature_dataset.csv            # Extracted facial vision features (40 x 225)
│   │   └── facial_extraction_failures.csv        # Extraction failure log (0 failures)
│   └── multimodal/
│       ├── cleaned_multimodal_dataset.csv        # True cleaned multimodal dataset (40 x 344)
│       ├── cleaned_multimodal_dataset_scaled.csv # Standardized z-score normalized dataset (EDA)
│       ├── descriptive_scaling_parameters.csv   # Predictor normalization parameters
│       └── multimodal_data_dictionary.csv        # 344-entry comprehensive data dictionary
│
├── outputs/
│   ├── audio/                                    # ⭐ PRIMARY FOR MIDTERM PIT (Figures + CSVs)
│   │   ├── 01_audio_sample_distribution.png      # Sample, activity, word frequencies, duration
│   │   ├── 02_empirical_affect_circumplex_audio.png # Russell's 2D valence-arousal space
│   │   ├── 03_prosodic_features_by_session.png   # F0, RMS, ZCR, duration boxplots with Welch t & FDR q
│   │   ├── 04_mfcc_timbral_profiles.png          # 13 MFCC heatmaps across valence & words
│   │   ├── 05_spectral_biomarkers_by_affect.png  # Centroid, bandwidth, rolloff, contrast
│   │   ├── 06_pitch_f0_dynamics_and_voicing.png  # F0 mean, voiced ratios, pitch range
│   │   ├── 07_lexically_controlled_kapoy_acoustics.png # Kapoy (n=11) acoustic variance
│   │   ├── 08_audio_predictor_pca_tsne_clustering.png  # Acoustic-only unsupervised embeddings
│   │   ├── 09_acoustic_feature_correlation_matrix.png  # Inter-feature acoustic correlation matrix
│   │   ├── 10_educational_context_acoustic_shifts.png  # Exam/quiz/lecture acoustic shifts
│   │   ├── summary_statistics_audio.csv          # Missing-aware audio descriptive statistics
│   │   ├── session_am_pm_comparison.csv          # AM vs. PM Welch t-test & FDR q-values
│   │   ├── class_activity_summary.csv            # Exam/quiz/lecture distribution
│   │   ├── spoken_word_affect_summary.csv        # Per-word valence and arousal means
│   │   ├── year_level_affect_summary.csv         # 4th-year affect statistics
│   │   └── audio_descriptive_report.md           # Summary report for audio modality
│   │
│   ├── facial/                                   # Vision Only (Figures + CSV)
│   │   ├── 01_facial_action_units_by_valence.png # AU12 smile, AU15 frown, AU4 brow lowerer
│   │   ├── 02_geometric_ratios_ear_mar.png       # Eye & Mouth Aspect Ratios across valence
│   │   ├── 03_head_pose_dynamics.png             # Head pitch, yaw, roll across sessions
│   │   ├── 04_facial_expression_variability.png  # Cross-AU variability & disgust proxy
│   │   ├── 05_facial_predictor_pca_tsne.png      # Facial predictor PCA & t-SNE
│   │   └── summary_statistics_facial.csv         # Facial descriptive statistics
│   │
│   └── multimodal/                               # Cross-Modal Analysis (Figures + CSVs)
│       ├── 01_audio_facial_correlation_heatmap.png # Cross-modal Pearson correlations with FDR threshold
│       ├── 02_multimodal_joint_pca_tsne.png      # Joint 295 acoustic + facial predictor embeddings
│       ├── 03_multimodal_lexically_controlled_kapoy.png # Acoustic vs. visual Action Units for Kapoy
│       ├── 04_educational_context_affect_profiles.png # Activity × Session confounding matrix
│       ├── audio_facial_correlations.csv         # Inter-modality correlation table
│       ├── ground_truth_affect_correlations.csv  # Correlations with valence & arousal
│       ├── affect_taxonomy_summary.csv           # Binned taxonomy statistics
│       └── lexically_controlled_kapoy.csv        # Kapoy cross-modal observations
│
├── DATA/                                         # Source survey & recordings
├── run_pipeline.py                               # Master execution pipeline
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
