# Preprocessing feature extraction and exploratory analysis audit

Reviewed October 2, 2026 against `/Users/macbookpro/Downloads/RESEARCH MINI-PROJECT v2.docx` and the second-deliverable screenshot.

The deliverable files exist, but the repository is not fully correct or ready to describe as validated. The main problems concern the definition of affect, the provenance of filled feeling responses, pitch handling, statistical interpretation, and preparation for later machine learning. Existing source code and deliverable datasets were left unchanged by this review. The survey CSV was independently edited during reverification; that change is described below.

## Reverification with the recordings

Reverified after the 40 WAVs and 40 MOVs became available on October 2, 2026. Both extraction modules were executed for all recordings in a temporary Python 3.11 environment. The merge/scaling/dictionary stages were also executed with fresh features, and all eight statistics tables were regenerated from the submitted unscaled dataset. Generated datasets and statistics went to `/tmp/bisaya-recomputed`; the submitted deliverables were not overwritten.

**Updated verdict: the extraction largely reproduces the saved features, and the recording pairs are consistent. The substantive preprocessing, outcome-definition and interpretation issues below remain.** Reproducible computation is not equivalent to a valid affect label or supported research conclusion.

| Recording or regeneration check | Result |
|---|---|
| Survey/audio/video inventory | All 40 survey audio filenames exist; audio and video filename stems match one-to-one |
| WAV format | All 40 are stereo, 44,100 Hz, 16-bit PCM; durations 0.44118–1.55574 seconds |
| Stored audio durations | All match the supplied WAVs within the CSV rounding precision |
| Duplicate audio content | No duplicate decoded PCM hashes |
| Sample clipping | No samples at the full-scale clipping threshold; this does not rule out prior distortion or noise |
| Video decoding | All 1,039 reported frames decode across all 40 clips; every clip is 1920 × 1080 at 30 fps |
| WAV versus embedded video audio | All 40 pairs decode; aligned waveform correlations range 0.998667–0.999994, with best lag 0 ms at 16 kHz in a ±100 ms search |
| Audio extraction | All 40 succeed, producing 40 × 125; every pitch, RMS energy, duration, sampling-rate and metadata value matches |
| Facial extraction | All 40 succeed, producing 40 × 206; every clip again has 100% face detection |
| Multimodal rebuild | Succeeds with 40 × 319, no null or infinite values |
| Statistical tables | All eight reproduce their numeric values, including Mann–Whitney p-values and Spearman results; Kapoy rows were matched by participant to account for a different display order among tied zero-pitch rows |

Strict comparison of the rounded CSV cells finds 157 audio differences and 794 facial differences, rather than bit-identical regeneration. The largest audio difference is 1.55 Hz in spectral-centroid standard deviation; differences also affect spectral contrast, zero crossings and a small number of MFCC/chroma values. For facial measures, the largest head-pose difference is 0.01 degree; other differences are at most 0.0006. These small differences are consistent with environment-dependent numerical behavior, but their exact cause is not established because the original dependency versions were not recorded. They are not evidence of wrong participant pairing. Fresh merge and scaling outputs inherit these differences; the original scaler arithmetic remains correct for its original inputs.

The verification used librosa 0.11.0, MediaPipe 0.10.21, NumPy 1.26.4, pandas 2.2.3, SciPy 1.15.2, scikit-learn 1.6.1 and OpenCV 4.11.0. The model SHA-256 is `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`. Package versions and all differing cells are saved in `outputs/recording_reverification/`.

### Additional recording findings

- The original 1024-sample pYIN configuration reproduces the same 17 missing-pitch cases. A diagnostic 2048-sample window returns pitch estimates for five of them: Y4-013, Y4-027, Y4-028, Y4-035 and Y4-038. These alternate estimates are candidates for review, not verified true F0; simply increasing the window is not established as the final solution. This reinforces the need for missing-pitch flags instead of physical 0-Hz interpretations.
- Y4-022, Y4-024, Y4-025 and Y4-032 have last-50-ms RMS above half their peak 50-ms-window RMS. The researchers confirmed that these clips were intentionally trimmed at the boundary to avoid background music. This explains the automated boundary flag and is not, by itself, grounds for exclusion. Manually confirm that each spoken word and final syllable remain complete and record whether music overlaps any portion of the speech. If music overlaps speech, add a quality flag because it may influence RMS, ZCR, spectral, MFCC and pitch features. Do not attempt selective noise removal unless the method is justified, documented and applied consistently.
- Midpoint frames from every video were visually inspected. The primary face is visible, but backgrounds, lighting and camera distance vary; several clips include background people. Detection success does not establish uniform acquisition conditions or the validity of inferred emotion.
- Waveform matching supports audio/video correspondence. It does not independently verify exact lip timing, correctness of each spoken word, unacted collection, independent self-report procedure, speaker identity or consent documentation. The supplied videos are already cut; uncut acquisition context remains unavailable.

Evidence files: `raw_audio_quality.csv`, `video_quality_and_alignment.csv`, `pitch_window_diagnostic.csv`, the four `*_feature_differences.csv` files, `verification_environment.txt` and `verification_summary.json` in `outputs/recording_reverification/`.

### Survey changed during verification

At the beginning of the audit, and in the Git-tracked original survey, 16 feeling cells were blank. During this run those same cells were filled with their respective spoken words, and the unused Collector Initials/Notes columns were removed. The current survey therefore has no blank feeling cells. These text values now agree with the existing extracted metadata, which previously used the code's word fallback. The reviewer did not make this survey edit.

Cell completeness is now resolved, but the provenance question remains: the file alone cannot establish whether the additions were recovered independent self-reports or replacements inferred from speech. Retain a correction record identifying the source of the additions. If the entries came from independently collected participant responses, update the audit status accordingly; if they were copied from spoken words, keep the original text responses missing and rely on the observed valence/arousal ratings. Quantitative ratings and all other required survey values were unchanged by the edit.

## Review scope and requirements

Read the supplied DOCX text, all six Python source/pipeline files, README, the repository's document transcription, every dataset/statistics CSV, the model asset inventory, and all ten visualization images (overview inspection, with the correlation heatmap inspected separately). The DOCX is research context, not an instruction to submit, upload, or implement its later modeling stage.

The research document requires at least 40 valid observations for Group 7, independent self-reported affect as the reference outcome, consistent audio preprocessing, one observation per recording, contextual rather than primary acoustic use of word/year/activity/session, educational-context EDA, and a same-word analysis. The screenshot additionally requests facial features and a cleaned multimodal dataset. Facial analysis therefore belongs to this milestone, while the eventual primary speech experiment should retain an audio-only baseline. The document's later ML and model-evaluation requirements are not missing second-deliverable artifacts.

## Checks that passed

| Check | Result |
|---|---|
| Survey observations | 40 unique participant codes; consent column says Yes for all 40 |
| Audio dataset | 40 rows, 125 columns |
| Facial dataset | 40 rows, 206 columns |
| Raw and scaled multimodal datasets | Both 40 rows, 319 columns |
| Participant correspondence | Survey, audio, facial and merged participant sets match; no current duplicates |
| Stored feature preservation | Shared source columns agree between audio/facial and the merged dataset |
| Survey quantitative ratings | All stored valence/arousal scores match the survey and fall within 1–5 |
| Numeric hygiene | All 304 numeric multimodal columns are finite; generated datasets have no blank cells |
| Dictionary coverage | 319 entries for 319 dataset columns |
| Descriptive summaries | All eight statistics for the 29 audio and 20 facial summary rows recompute within rounding tolerance |
| Session comparisons | Means, Welch t statistics/p-values, Cohen's d and Mann–Whitney U statistics recompute within rounding tolerance |
| Correlations | All 90 cross-modal Pearson coefficients and 38 ground-truth Pearson coefficients recompute within rounding tolerance |
| Standardization arithmetic | Every numeric scaled column matches whole-dataset population z-scores within CSV rounding tolerance |
| Visualization artifacts | All ten PNGs decode and contain approximately 300-DPI metadata |
| Python syntax | All source files and run_pipeline.py parse successfully |

These checks confirm internal arithmetic and file integrity. They do not certify participant independence, ethics documentation or research validity. The initial review independently recomputed summaries, Welch tests and Pearson coefficients; the recording reverification additionally regenerated the complete statistics tables with SciPy, including Mann–Whitney p-values and Spearman results.

## Findings requiring correction

### 1. High — Word-derived categories are presented as participant affect

`src/audio_features.py:15–47, 114–139`, `src/facial_features.py:169`, `src/statistics_analysis.py:155–168`, and multiple plots in `src/visualizations.py` derive `affect_category` from `spoken_word`. Consequently, the report's 37.5% Fatigue / 27.5% Stress / 20% Positive / 15% Neutral breakdown describes a researcher-assigned word taxonomy, not an independent affect distribution. No validation of that taxonomy is supplied.

Concrete counterexamples: Y4-002 says Lutang and has valence 4 / arousal 3, but is assigned Fatigue; Y4-015 says Kapoy and has valence 5 / arousal 5, but is also assigned Fatigue. These are precisely the distinctions the research protocol asks the project to preserve.

Correction: identify the existing taxonomy explicitly as lexical categories if retaining it. Base affect summaries and outcomes on observed self-reports and ratings; document any rating-based class definitions. Summarize valence counts as 1:5, 2:8, 3:13, 4:7, 5:7, and arousal counts as 1:3, 2:10, 3:16, 4:4, 5:7.

### 2. High — Word fallback hides missing feelings and the recent fills need provenance

`src/audio_features.py:78`, `src/audio_features.py:127`, and `src/multimodal_dataset.py:62–63` substitute spoken words for missing feelings. The original Git-tracked survey contained 16 blank Self-Reported Feeling cells; the extracted datasets contained none. During reverification the source cells were filled with the same words, as explained above. This resolves present-day blank cells but does not, by itself, establish complete independently observed text responses. Valence and arousal remain available for these rows, so the 16 observations should not automatically be discarded.

The metadata loader also defaults unparseable ratings to 3 and silently overwrites duplicate participant codes. These are latent hazards; current ratings are valid and codes unique.

Correction: document the provenance of the recent additions. Preserve genuinely missing feelings and add a missingness flag. Keep valid ratings. Reject or quarantine missing/invalid ratings and duplicate identifiers rather than manufacturing neutral observations. Provide a quality report distinguishing observed, missing, corrected and excluded values.

### 3. High — Pitch detection failure is treated as measured 0 Hz

`src/audio_features.py:169–205` sets every pitch statistic to zero when pYIN returns no voiced F0. This affects 17/40 recordings (42.5%), including 7/11 Kapoy observations. The resulting overall pitch mean is 100.589 Hz; the mean among the 23 observations with detected pitch is 174.937 Hz. Both describe different quantities; zero-inclusive values should not be interpreted as ordinary fundamental frequency.

There are 9 zero-pitch rows in Fatigue, 5 in Positive, 1 in Stress/Anxiety, and 2 in Neutral. Apparent category differences can therefore reflect unequal detection failure. A failed detector does not establish a whisper, creaky voice, or emotional state.

Correction: represent unavailable pitch with missing values plus a detection flag, inspect affected recordings, and summarize pitch conditional on valid detection. If imputing predictors later, fit imputation within training folds. Report coverage as well as feature values.

### 4. High — Scaling and embedding inputs are unsuitable for an independent predictive experiment

`src/multimodal_dataset.py:103–106` scales every numeric column, including the reference valence and arousal ratings, video frame counts and quality fields. It also fits on all observations. Min/max normalization for the derived indices at lines 67–81 likewise uses the whole dataset.

Whole-dataset scaling is mathematically correct for descriptive visualization, but reusing those transformations before splitting a predictive experiment leaks held-out information. The scaler is not persisted. In `src/visualizations.py:315–318`, PCA/t-SNE inputs include the outcome ratings themselves; this cannot support a claim that acoustic/visual predictors alone reveal affect clusters. The plotted representation uses 302 numeric inputs, while its title claims 314.

Correction: define explicit predictor, outcome, context and quality column roles; preserve raw ratings. Fit scaling, imputation, feature selection and any learned composite normalization only on training folds. Build predictor-only embeddings if interpreting structure in the extracted signals. See [scikit-learn's leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).

### 5. High — Report interpretations contradict the tables

`src/statistics_analysis.py:410–441` hardcodes conclusions that are not supported by the calculations:

| Report claim | Actual current table |
|---|---|
| Strong spectral-centroid/jaw-opening coupling | Pearson r = −0.0016, p = 0.99226 |
| Strong spectral-centroid/MAR coupling | r = 0.2710, p = 0.09077 |
| Positive pitch-range/expressiveness relationship | r = −0.0566, p = 0.72880 |
| Elevated positive-state pitch | Positive lexical category has the lowest zero-inclusive mean, 61.4 Hz |
| Same-word variation proves independent affect information | Variation alone does not establish a relationship to reference affect or control speaker differences |

The ground-truth correlation CSV contains no nominal p < .05 feature association with either valence or arousal among its tested features. This does not prove absence of signal, especially with N=40, but does not support the existing proof claims either. That table is generated but is not incorporated into the main report.

Correction: generate interpretations from actual results, report uncertainty and limitations, and replace unsupported causal/emotional descriptions with observed measurements. Detection rate 100% is not proof of overall recording quality; jitter/shimmer proxies are not proof of clear capture.

### 6. Medium — AM/PM differences are confounded and do not directly answer affect-by-session

There are 34 AM and 6 PM observations. Every PM observation comes from Foreign Language–Japanese / Lecture–Review. Every AM observation comes from Graph Theory / Prelim Exam (14) or Numerical Analysis / Quiz (20). Subject, activity and session cannot be separated in this sample.

Session acoustic comparisons are valid descriptive calculations, but interpretations about time of day, relief or fatigue are not established. Pitch and smile session differences are nonsignificant in the stored tests. Mean reference valence is 2.9706 for AM and 3.6667 for PM; mean reference arousal is 3.0000 and 3.3333, respectively. These are more directly relevant to the requested affect-by-session EDA, subject to the same confounding.

Correction: present counts/distributions of observed ratings by session, activity and word; disclose small cells and confounding. All participants are fourth-year, so year-level variation cannot be analyzed within this group's data.

### 7. Medium — Multiple tests and significance selection overstate evidence

`src/statistics_analysis.py:145, 233` declares significance if the smaller of two p-values is below .05. The repository tests 19 session features with two tests each, 90 cross-modal pairs with two methods each, and 38 feature/outcome associations. No multiplicity adjustment is provided. None of the 90 cross-modal Pearson tests survives Benjamini–Hochberg FDR correction at .05 using the stored p-values.

Correction: choose/report a primary test, label exploratory results, define testing families and include an adjustment or a clear unadjusted-testing limitation. Do not print rounded p = 0.0000 as if the true probability were zero. For example, duration Welch p is approximately 3.87e−5; RMS Welch p is approximately 2.69e−11.

### 8. Medium — Audio perturbation names and short-recording handling need repair

`src/audio_features.py:184–188` differences framewise F0 periods after removing unvoiced frames. This may also connect separated voiced intervals. Lines 226–230 difference framewise RMS. These are frame-variation proxies, not conventional cycle-to-cycle local jitter and shimmer; see the [Praat jitter definition](https://www.fon.hum.uva.nl/praat/manual/Voice_2__Jitter.html) and [shimmer definition](https://www.fon.hum.uva.nl/praat/manual/Voice_3__Shimmer.html).

The MFCC delta call uses width 9 and interpolation mode without checking available frame count. Short inputs with fewer than nine frames can fail; the 160-sample trim fallback does not address this. The batch catches the exception and skips the observation. This is a latent edge case, not a demonstrated failure in the current 40 CSV rows. See [librosa delta documentation](https://librosa.org/doc/0.11.0/generated/librosa.feature.delta.html).

Correction: rename proxies or use validated cycle-based measurements; retain voiced interval boundaries. Validate speech duration and frame counts, handle short inputs explicitly, and log rejected recordings. All current original/trimmed durations are equal: this is not itself an error, but cannot demonstrate successful speech isolation without inspecting audio.

### 9. Medium — Facial geometry and quality controls are incomplete

`src/facial_features.py:28–77` computes Euclidean distances directly on normalized x/y/z coordinates. x and y use image width and height, while z has its own relative scaling. These are not consistent physical axes; EAR/MAR can depend on aspect ratio and pose. Use an explicitly documented pixel-coordinate 2D formulation or validated metric geometry. See [MediaPipe's coordinate description](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md).

Missing facial signals can become zeros; there is no minimum detection/valid-frame gate. The current data reports 100% detection, so failure handling is a robustness concern rather than an observed dropout. Framewise image-mode detection is valid, but there is no video tracking/timestamp treatment or recorded audio/video synchronization evidence. Whole-clip correlations describe participant-level associations, not temporal synchronization.

Blendshape scores and composites should be described as model-derived proxies, not validated FACS coding, measured muscle activation or objective affect. The dictionary also assigns generic categorical/string units to numeric ratings and inaccurate broad units to some features.

### 10. Medium — Merge and extraction validation do not establish 40 valid observations

`src/multimodal_dataset.py:48–59` checks participant-set equality and then merges only on participant code. It does not validate one-to-one cardinality, recording identity, or metadata disagreement before dropping facial metadata. Duplicate participant rows could produce a Cartesian expansion; matched omissions in both modalities could pass the set check. Current rows are unique and shared metadata agrees.

The pipeline does not reconcile every survey observation against successful extraction, enforce the minimum sample size, validate consent/context/rating rules, or produce a structured reject log. Missing survey files can trigger fallback metadata, hardcoded fourth-year labels and fabricated neutral scores.

Correction: validate schema and identifiers, require one-to-one observation joins, compare shared metadata, reconcile source inventories against survey records, and produce a structured quality report. Use explicit checks rather than relying only on Python assertions, which can be disabled.

### 11. Medium — Collection deviations and documentation discrepancies need disclosure

25/40 stored responses are outside the document's seven offered words. The document also says any response is acceptable, so this is a collection-procedure point to document rather than an established violation or reason to delete observations. Clarify how the offered list and free responses were handled. The independent response procedure cannot be established from code or the already-cut recordings. Survey activities are Lecture / Review, Prelim Exam and Quiz rather than a consistently mapped protocol vocabulary; preserve original labels and document a mapping if standardizing.

Y4-021 has survey word Duha-duha while the feature dataset/filename uses Gaduha-duha. Preserve and document the correction or reconcile against the recording. `load_survey_metadata` does not load Year Level and both extractors hardcode fourth-year fallback values; this happens to match the current cohort but is unsafe for class consolidation.

The report says Lisod occurs zero times because it looks up a truncated top-five list; the dataset has two. README's Kapoy maxima (240.2 Hz, smile 0.675) disagree with the current data (231.11 Hz, smile 0.4594). README calls all 125 audio columns features, but these include 16 metadata columns; 109 are numeric audio fields, including sampling rate/durations. Facial 206 columns include 16 metadata fields, four video fields and 186 face fields. The multimodal dataset has 304 numeric columns and 18 constant numeric columns, rather than 314 validated predictors. Windows absolute links and virtual-environment instructions do not provide a reproducible local setup here.

## Reproducibility limits

The initial review lacked raw media and extraction dependencies. The user subsequently supplied both recording directories, which remain excluded by .gitignore. Dependencies were installed into a temporary Python 3.11 environment for this reverification; the project still has no dependency manifest documenting the environment that produced its submitted artifacts.

Audio/facial extraction, multimodal construction and statistics were rerun with temporary outputs. Existing figures were inspected in the initial review and were not regenerated during reverification. The master run_pipeline.py command was not executed as a single operation. Waveform/full-frame checks and midpoint visual inspection now support file integrity and pairing, but comprehensive listening, true pitch validation, precise lip synchronization, independently validated face geometry, speaker identity, collection procedures and consent documentation remain unverified. Survey consent values are metadata evidence, not verification of consent documents.

## Recommended correction order

1. Document the provenance of the recently filled self-report cells, preserve genuine missingness, validate metadata and reconcile the word discrepancy; create a quality report.
2. Replace word-derived affect interpretations with analyses of reference ratings and observed feelings.
3. Mark undetected pitch as unavailable; inspect recordings and rebuild affected summaries.
4. Define predictor/outcome/context/quality roles and separate descriptive scaling from fold-fitted modeling transformations.
5. Regenerate tables, figures and prose with uncertainty, confounding and testing limitations; correct artifact counts and stale claims.
6. Add dependency/setup instructions and a raw-data availability procedure so extraction can be reproduced without committing identifiable recordings unnecessarily.
