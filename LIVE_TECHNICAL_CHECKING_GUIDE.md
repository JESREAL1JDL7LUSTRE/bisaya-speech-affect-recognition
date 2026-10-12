# 10-Minute Live Technical Checking Guide
**Course:** CS412 – Affective Computing Research Mini-Project  
**Cohort:** 4th-Year Computer Science Students (Group 7)  
**Deliverable:** Live Laptop Technical Demonstration Guide (PIT Presentation Deliverable 2)

---

## ⏱️ Presentation & Demo Time Allocation (10 Minutes Total)

| Section | Target Duration | Primary Objective | Key Artifacts Demonstrated |
| :--- | :---: | :--- | :--- |
| **A. Raw Data & Dataset Verification** | ~3 mins | Prove 40 valid participants, 1-to-1 file correspondence, and ground-truth affect labels. | Metadata CSV, Raw `.wav` directory, `verify_dataset.py` |
| **B. Preprocessing & Feature Extraction** | ~3 mins | Demonstrate raw `.wav` loading, silence trimming, and live extraction of 108 acoustic features. | `src/audio_features.py`, `extract_single_feature.py` |
| **C. Machine Learning & Results** | ~4 mins | Show leak-free CV pipeline, evaluation metric tables, confusion matrices, and model comparison. | `ser_machine_learning_experiments.ipynb`, `evaluate_models.py` |

---

## 💻 Pre-Demo Checklist (Do this 5 minutes before your turn!)

1. **Open VS Code** with the project folder.
2. **Open a PowerShell Terminal** in VS Code and activate the virtual environment:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
3. **Open the following files in tabs** so you can click between them instantly:
   * Tab 1: [`DATA/RESEARCH MINI-PROJECT v2 - G7 - 4th year.csv`](file:///e:/SCHOOL/3rd%20year/elec1/bisaya-speech-affect-recognition/DATA/RESEARCH%20MINI-PROJECT%20v2%20-%20G7%20-%204th%20year.csv)
   * Tab 2: [`verify_dataset.py`](file:///e:/SCHOOL/3rd%20year/elec1/bisaya-speech-affect-recognition/verify_dataset.py)
   * Tab 3: [`src/audio_features.py`](file:///e:/SCHOOL/3rd%20year/elec1/bisaya-speech-affect-recognition/src/audio_features.py)
   * Tab 4: [`ser_machine_learning_experiments.ipynb`](file:///d:/SCHOOL/affective/ser_machine_learning_experiments.ipynb)
   * Tab 5: [`outputs/audio/11_ser_model_confusion_matrices.png`](file:///d:/SCHOOL/affective/outputs/audio/11_ser_model_confusion_matrices.png)
   * Tab 6: [`outputs/audio/12_ser_model_performance_comparison.png`](file:///d:/SCHOOL/affective/outputs/audio/12_ser_model_performance_comparison.png)
4. **Have File Explorer open** to `DATA\2AudioRecordings\Raw .wav` showing the 40 audio files.

---

## 📋 Step-by-Step Demonstration Script

### Part A: Raw Data and Dataset Verification (~3 Minutes)

#### Step A1: Run the Automated Verification Script
In your terminal, execute:
```powershell
.\.venv\Scripts\python.exe verify_dataset.py
```
*(Runs in ~1 second and prints a clean, tabulated verification summary).*

#### Step A2: What to Show on Screen
1. **Show the Survey Metadata CSV (`DATA/RESEARCH MINI-PROJECT v2 - G7 - 4th year.csv`):**
   * Point out the 40 rows: codes `Y4-001` through `Y4-040`.
   * Point out that `Consent Obtained` is recorded as `Yes` for all 40 participants.
2. **Show the Audio Recordings Directory (`DATA/2AudioRecordings/Raw .wav`):**
   * Show that there are exactly 40 `.WAV` files.
   * Highlight the standardized naming convention: `ParticipantCode_YearLevel_Session_Word.wav` (e.g., `Y4-011_Y4_AM_Kapoy.WAV`).
3. **Show File Correspondence in the Terminal Output:**
   * Point to `[3] 1-to-1 File Correspondence Check: [OK] PASSED (100% MATCH)`.
   * Explain that every survey row links to its exact audio recording without missing files or duplicates.
4. **Show Ground Truth Target Affect Labels:**
   * **Valence 3-Class:** Negative ($n=13$, 32.5%), Neutral ($n=13$, 32.5%), Positive ($n=14$, 35.0%).
   * **Arousal 3-Class:** Low ($n=13$, 32.5%), Moderate ($n=16$, 40.0%), High ($n=11$, 27.5%).
5. **State the Secondary Facial Dataset Status:**
   * Show `datasets/facial/facial_feature_dataset.csv` (40 rows × 225 features).
   * State clearly: *"In accordance with Ma'am Love Jhoye's guidance, our primary SER modeling is acoustic-only, but we preserved the 225 facial features for academic completeness."*

> 🎙️ **Talking Points for Part A:**  
> *"Good morning panel. For Part A, we verify our raw dataset consisting of 40 fourth-year Computer Science student responses. As shown on screen, all 40 participants provided informed consent under research codes Y4-001 to Y4-040. There is an exact one-to-one match between our survey records and the 40 raw WAV recordings on disk. Our target affect labels are strictly derived from independent self-reports, mapped into balanced 3-class Valence and Arousal groups."*

---

### Part B: Preprocessing and Feature Extraction (~3 Minutes)

#### Step B1: Run the Single-Recording Extraction Demonstration
In your terminal, execute:
```powershell
.\.venv\Scripts\python.exe extract_single_feature.py
```
*(Optionally test another file: `.\.venv\Scripts\python.exe extract_single_feature.py "DATA/2AudioRecordings/Raw .wav/Y4-001_Y4_PM_Thrilled.WAV"`)*

#### Step B2: What to Show on Screen
1. **Show the Preprocessing Parameters in the Output:**
   * **Target Sampling Rate:** Resampled to 16,000 Hz mono.
   * **Silence Trimming:** `librosa.effects.trim(top_db=25)` to isolate active vocalization without clipping phonemes.
   * **Frame & Hop Length:** STFT frame length of 2048 / 1024, hop length of 256 samples ($16\text{ ms}$).
2. **Show the Extracted Feature Vector Categories:**
   * **Prosody / Pitch:** $F_0$ mean (e.g., $154.08\text{ Hz}$), $F_0$ range, voiced ratio.
   * **Energy / Intensity:** RMS mean (e.g., $0.07889$), RMS std.
   * **Spectral Dynamics:** Zero-Crossing Rate (ZCR), Spectral Centroid ($1555.13\text{ Hz}$), Spectral Bandwidth, Rolloff 85%, Spectral Contrast.
   * **Timbre:** MFCC 1–13 means & stds, Delta MFCCs, Delta-Delta MFCCs.
3. **Show Code Implementation in `src/audio_features.py`:**
   * Briefly switch to [`src/audio_features.py`](file:///d:/SCHOOL/affective/src/audio_features.py) at lines 199–305 to show the librosa extraction functions.
   * Highlight line 231–242: Explain that unvoiced recordings are **preserved as `NA`**, leaving 28 valid pitch values rather than imputing $0\text{ Hz}$ which would distort pitch statistics.

> 🎙️ **Talking Points for Part B:**  
> *"For Part B, we demonstrate the processing of raw audio into a structured 108-feature acoustic predictor vector. Here, we process participant Y4-011 who spoke 'Kapoy'. Librosa loads the file at 16 kHz mono and trims silence using a 25 dB threshold. Using probabilistic YIN (pYIN), we extract fundamental frequency, energy via RMS, articulation speed via Zero-Crossing Rate, spectral centroid and contrast, and 13 MFCCs with first and second derivatives. Crucially, unvoiced frames are maintained as NA rather than zero to maintain mathematical integrity."*

---

### Part C: Machine Learning and Results (~4 Minutes)

#### Step C1: Open the Jupyter Notebook OR Run the Demo Script

##### Option 1 (Recommended for Visual Presentation):
Open [`ser_machine_learning_experiments.ipynb`](file:///d:/SCHOOL/affective/ser_machine_learning_experiments.ipynb) directly in VS Code:
1. Select the `.venv` Python kernel in the top-right corner (`Python 3.11.x (.venv)`).
2. Click **Run All** (takes ~10 seconds).

##### Option 2 (Instant CLI Execution):
In your terminal, execute:
```powershell
.\.venv\Scripts\python.exe evaluate_models.py
```

#### Step C2: What to Show on Screen
1. **Show the Leakage-Free Pipeline Architecture:**
   * Point out the `sklearn.pipeline.Pipeline`:
     ```python
     Pipeline([
         ('imputer', SimpleImputer(strategy='median')),
         ('scaler', StandardScaler()),
         ('clf', model)
     ])
     ```
   * Explain: Imputation of missing pitch and standard scaling are fitted **strictly inside each training fold**, preventing validation data from leaking into training statistics.
   * Confirm that self-report ratings and educational context were strictly excluded from feature inputs $X$.
2. **Show the Model Evaluation Comparison Table:**
   * **Valence 3-Class (Chance = 33.3%):**
     * Majority Baseline: $25.0\% \pm 0.0\%$ (F1: $0.133$)
     * Logistic Regression: $30.0\% \pm 18.7\%$ (F1: $0.290$)
     * Linear SVM: $22.5\% \pm 14.6\%$ (F1: $0.215$)
     * RBF SVM: $27.5\% \pm 9.4\%$ (F1: $0.218$)
     * **Random Forest (Winner):** $\mathbf{32.5\% \pm 6.1\%}$ (Precision: $0.397$, Macro F1: $\mathbf{0.306}$)
   * **Arousal 3-Class (Majority Baseline = 40.0%):**
     * **Linear SVM (Winner):** $\mathbf{47.5\% \pm 9.4\%}$ (LOOCV: $\mathbf{55.0\%}$, Precision: $0.528$, Macro F1: $\mathbf{0.437}$)
     * Random Forest: $45.0\% \pm 6.1\%$ (Macro F1: $0.403$)
     * Logistic Regression: $45.0\% \pm 10.0\%$ (Macro F1: $0.402$)
3. **Show the Confusion Matrices & Model Comparison Figures:**
   * Open Figure 11: [`outputs/audio/11_ser_model_confusion_matrices.png`](file:///d:/SCHOOL/affective/outputs/audio/11_ser_model_confusion_matrices.png).
     * Show the 2×2 grid: Valence (Random Forest, Logistic Regression) and Arousal (Linear SVM, Random Forest).
     * Explain: Annotated with counts, colored by row-normalized recall.
   * Open Figure 12: [`outputs/audio/12_ser_model_performance_comparison.png`](file:///d:/SCHOOL/affective/outputs/audio/12_ser_model_performance_comparison.png).
     * Show grouped bars for Accuracy & Macro-F1 across all 5 models with error bars ($\pm 1\text{ SD}$).
     * Point to the dashed chance level line ($33.3\%$) and dotted majority baseline ($35.0\%$ Valence, $40.0\%$ Arousal).
4. **Explain the Majority Baseline & Variance Across Folds:**
   * *Baseline Note:* Explain why majority baseline shows $25.0\%$ in 5-fold CV: "known artifact of stratified CV where the most common training class is slightly under-represented in the test split; full dataset majority is $35.0\%$, and chance is $\approx 33.3\%$."
   * *Fold Variance:* Note the standard deviations ($\pm 6.1\%$ to $\pm 18.7\%$), proving that single splits are volatile for $N=40$.

> 🎙️ **Talking Points for Part C:**  
> *"In Part C, we benchmark our acoustic classifiers using Stratified 5-Fold Cross-Validation. We enforce strict data hygiene: scaling and pitch imputation are fitted inside the CV loop to prevent data leakage. For 3-class Valence, Random Forest achieves 32.5% accuracy and an F1 of 0.306, performing close to chance level (33.3%). This proves empirically that speech acoustics alone have difficulty distinguishing pleasant from unpleasant states in short natural utterances. However, for 3-class Arousal, Linear SVM achieves 47.5% in 5-fold CV and 55.0% in Leave-One-Out CV, significantly beating the 40.0% majority baseline. Energy and pitch provide a reliable acoustic signature of physiological activation."*

---

## ⚡ Quick Terminal Command Summary

Copy-paste these commands directly into your terminal during the demo:

```powershell
# 1. Activate Virtual Environment
.\.venv\Scripts\Activate.ps1

# 2. Part A: Verify 40 participants, files, and labels (1 sec)
.\.venv\Scripts\python.exe verify_dataset.py

# 3. Part B: Preprocess & extract 108 features from one raw WAV (1 sec)
.\.venv\Scripts\python.exe extract_single_feature.py

# 4. Part C: Run 5-fold CV evaluation & plot confusion matrices (8 secs)
.\.venv\Scripts\python.exe evaluate_models.py
```

---

## 🎯 Anticipated Panel Questions & Defenses

### Q1: "Why did you exclude facial features from your machine learning models?"
* **Answer:** *"Per Ma'am Love Jhoye's guidance in October, our midterm milestone specifically isolates acoustic and prosodic speech emotion recognition. We successfully extracted 225 MediaPipe FACS blendshape features and preserved them in `datasets/facial/`, but keeping the SER baseline acoustic-only ensures we can cleanly evaluate how much emotional information exists purely in the Bisaya speech signal."*

### Q2: "Why is your Valence accuracy only around 32.5% (close to chance)?"
* **Answer:** *"This is an authentic, well-documented finding in Speech Emotion Recognition literature: valence (pleasantness) is notoriously difficult to classify from acoustic acoustics alone, especially in brief spontaneous utterances (mean duration 0.88s). Our PCA and t-SNE projections confirmed heavy acoustic overlap. To classify valence accurately, multimodal fusion with facial expressions (smiles vs. frowns) and semantic text embeddings is necessary."*

### Q3: "Why did Arousal perform significantly better (47.5% - 55%)?"
* **Answer:** *"Arousal measures physiological activation and energy. When students feel high arousal after a difficult exam or quiz, vocal cord tension and lung pressure increase, directly elevating fundamental frequency (F0) and RMS loudness. These physical biomarkers translate directly into our acoustic features, allowing Linear SVM to beat the baseline by 15% in LOOCV."*

### Q4: "How did you prevent data leakage during feature preprocessing?"
* **Answer:** *"We used scikit-learn Pipelines. Standard scaling (mean/variance) and median imputation for unvoiced pitch were computed strictly using the 32 training instances in each fold, and then transformed onto the 8 validation instances. Furthermore, participant ratings and educational context were completely quarantined from predictor matrices."*

### Q5: "Can you tell emotion from the word 'Kapoy' alone?"
* **Answer:** *"Our lexically controlled analysis of the 11 'Kapoy' recordings proves that the word is not the emotion. Within 'Kapoy' alone, pitch ranged from 52 Hz to 296 Hz, and self-reported ratings ranged from Valence 1 to 5. Speech prosody and tone carry independent affective information that goes beyond lexical meaning."*
