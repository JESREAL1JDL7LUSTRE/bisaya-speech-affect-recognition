# Descriptive statistics and exploratory analysis

## Scope and reference outcome

This analysis contains 40 fourth-year participant observations. Affect is represented by independently recorded valence and arousal ratings. Spoken-word categories are lexical context and are not treated as participant emotion.

Valence counts (scores 1–5): {1: 5, 2: 8, 3: 13, 4: 7, 5: 7}. Arousal counts (scores 1–5): {1: 3, 2: 10, 3: 16, 4: 4, 5: 7}.
Pitch was detected for 28/40 recordings; pitch summaries exclude unavailable observations rather than treating them as 0 Hz.

## Educational context

| session   | Subjects                         | Activities        |   N |
|:----------|:---------------------------------|:------------------|----:|
| AM        | Graph Theory, Numerical Analysis | Prelim Exam, Quiz |  34 |
| PM        | Foreign Language - Japanese      | Lecture / Review  |   6 |

Session, subject and activity are completely confounded in this group: PM contains Foreign Language / Lecture-Review, while AM contains Graph Theory / Prelim Exam and Numerical Analysis / Quiz. Session comparisons are descriptive and cannot isolate a time-of-day effect. Year-level differences cannot be evaluated because every observation is fourth year.

## Session comparisons

Welch tests were treated as the primary session tests and adjusted as one family using Benjamini–Hochberg FDR. 6 of 33 comparisons have q < .05. Mann–Whitney results are secondary sensitivity checks.

| Feature                         |   AM_N |   PM_N |     AM_Mean |     PM_Mean |   Welch_p |   Welch_q_BH |   Cohens_d |
|:--------------------------------|-------:|-------:|------------:|------------:|----------:|-------------:|-----------:|
| valence_score                   |     34 |      6 |    2.97059  |    3.66667  |  0.037203 |     0.153463 |  -0.552557 |
| arousal_score                   |     34 |      6 |    3        |    3.33333  |  0.283248 |     0.424872 |  -0.281366 |
| audio_duration_sec              |     34 |      6 |    0.907644 |    0.712117 |  3.9e-05  |     0.000426 |   0.976147 |
| audio_trimmed_dur_sec           |     34 |      6 |    0.907644 |    0.712117 |  3.9e-05  |     0.000426 |   0.976147 |
| audio_f0_mean_hz                |     23 |      5 |  180.037    |  143.97     |  0.024223 |     0.114193 |   0.732321 |
| audio_f0_std_hz                 |     23 |      5 |   21.6504   |   15.83     |  0.393531 |     0.519461 |   0.288408 |
| audio_f0_range_hz               |     23 |      5 |   60.6474   |   45.364    |  0.355161 |     0.489785 |   0.289639 |
| audio_voiced_ratio              |     34 |      6 |    0.456044 |    0.640033 |  0.240272 |     0.412264 |  -0.507947 |
| audio_period_variation_proxy    |     23 |      5 |    0.007385 |    0.009958 |  0.247265 |     0.412264 |  -0.719426 |
| audio_rms_mean                  |     34 |      6 |    0.075131 |    0.0272   |  0        |     0        |   1.83261  |
| audio_rms_std                   |     34 |      6 |    0.033953 |    0.018165 |  0.000189 |     0.001561 |   1.22098  |
| audio_rms_frame_variation_proxy |     34 |      6 |    0.111999 |    0.148353 |  0.09479  |     0.24062  |  -1.15655  |
| audio_zcr_mean                  |     34 |      6 |    0.11991  |    0.095127 |  0.008983 |     0.049406 |   1.31463  |
| audio_spec_centroid_mean        |     34 |      6 | 1586.47     | 1490.42     |  0.071483 |     0.235892 |   0.613957 |
| audio_spec_bandwidth_mean       |     34 |      6 | 1514.22     | 1620.59     |  0.000342 |     0.002257 |  -0.946783 |
| audio_spec_rolloff85_mean       |     34 |      6 | 3162.82     | 3207.41     |  0.657081 |     0.747713 |  -0.118143 |
| audio_spec_flatness_mean        |     34 |      6 |    0.019603 |    0.020122 |  0.859162 |     0.859162 |  -0.051272 |
| face_smile_mean                 |     34 |      6 |    0.217585 |    0.397367 |  0.247902 |     0.412264 |  -0.762093 |
| face_smile_max                  |     34 |      6 |    0.432641 |    0.61525  |  0.239331 |     0.412264 |  -0.583736 |
| face_frown_mean                 |     34 |      6 |    0.002185 |    0.001867 |  0.770862 |     0.847949 |   0.070898 |
| face_frown_max                  |     34 |      6 |    0.016585 |    0.009117 |  0.42904  |     0.54455  |   0.168847 |
| face_brow_lowerer_mean          |     34 |      6 |    0.051294 |    0.112917 |  0.26235  |     0.412264 |  -0.876968 |
| face_brow_inner_raiser_mean     |     34 |      6 |    0.108662 |    0.067367 |  0.46324  |     0.566182 |   0.353832 |
| face_jaw_open_mean              |     34 |      6 |    0.061168 |    0.0546   |  0.799944 |     0.851553 |   0.091519 |
| face_eye_squint_mean            |     34 |      6 |    0.369688 |    0.462633 |  0.080928 |     0.238675 |  -0.796154 |
| face_disgust_proxy_mean         |     34 |      6 |    0.096994 |    0.136783 |  0.250414 |     0.412264 |  -0.423512 |
| face_expressiveness_proxy       |     34 |      6 |    0.039797 |    0.043417 |  0.356207 |     0.489785 |  -0.292854 |
| face_ear_mean                   |     34 |      6 |    0.330759 |    0.309833 |  0.086791 |     0.238675 |   0.693276 |
| face_ear_min                    |     34 |      6 |    0.292647 |    0.256667 |  0.062011 |     0.227372 |   0.662427 |
| face_mar_mean                   |     34 |      6 |    0.161579 |    0.1855   |  0.539257 |     0.635552 |  -0.371695 |
| face_mar_max                    |     34 |      6 |    0.326529 |    0.279717 |  0.23771  |     0.412264 |   0.515326 |
| face_pitch_mean_deg             |     34 |      6 |   -3.65941  |    1.17667  |  0.132258 |     0.31175  |  -0.886272 |
| face_yaw_mean_deg               |     34 |      6 |   -2.04794  |   -2.75667  |  0.826758 |     0.852594 |   0.110706 |

## Rating-derived affect groups

Valence groups use scores 1–2 = Negative, 3 = Neutral, and 4–5 = Positive. Arousal groups use 1–2 = Low, 3 = Moderate, and 4–5 = High. These are transparent analytical bins, not clinical emotion diagnoses.

| valence_group   | arousal_group   | affect_category   |   Count |   Mean_Valence |   Mean_Arousal |   Pitch_Available |   Mean_F0_Hz |   Mean_RMS |   Mean_Smile_Proxy |   Percentage |
|:----------------|:----------------|:------------------|--------:|---------------:|---------------:|------------------:|-------------:|-----------:|-------------------:|-------------:|
| Negative        | Low             | Negative-Low      |       7 |          1.571 |          1.571 |                 5 |      196.292 |      0.088 |              0.277 |         17.5 |
| Negative        | Moderate        | Negative-Moderate |       6 |          1.667 |          3     |                 5 |      199.198 |      0.072 |              0.236 |         15   |
| Neutral         | High            | Neutral-High      |       4 |          3     |          4.5   |                 3 |      162.297 |      0.068 |              0.074 |         10   |
| Neutral         | Low             | Neutral-Low       |       5 |          3     |          2     |                 4 |      159.865 |      0.071 |              0.24  |         12.5 |
| Neutral         | Moderate        | Neutral-Moderate  |       4 |          3     |          3     |                 3 |      171.64  |      0.062 |              0.148 |         10   |
| Positive        | High            | Positive-High     |       7 |          4.714 |          4.714 |                 3 |      155.173 |      0.052 |              0.3   |         17.5 |
| Positive        | Low             | Positive-Low      |       1 |          5     |          2     |                 1 |      160.42  |      0.141 |              0.518 |          2.5 |
| Positive        | Moderate        | Positive-Moderate |       6 |          4.167 |          3     |                 4 |      154.012 |      0.048 |              0.287 |         15   |

## Same-word analysis

There are 11 Kapoy recordings. Their variation can describe within-word heterogeneity, but variation alone does not prove that features predict affect or control for speaker differences.

| participant_code   |   valence_score |   arousal_score | audio_pitch_detected   |   audio_f0_mean_hz |   audio_rms_mean |   face_smile_mean |
|:-------------------|----------------:|----------------:|:-----------------------|-------------------:|-----------------:|------------------:|
| Y4-036             |               1 |               1 | False                  |             nan    |          0.11504 |            0.4625 |
| Y4-037             |               1 |               1 | True                   |             178.11 |          0.10202 |            0.0048 |
| Y4-028             |               1 |               2 | True                   |             156.66 |          0.11739 |            0.2787 |
| Y4-025             |               2 |               2 | False                  |             nan    |          0.06956 |            0.0793 |
| Y4-030             |               2 |               2 | True                   |             176.77 |          0.04083 |            0.1741 |
| Y4-027             |               2 |               3 | True                   |             296.69 |          0.12357 |            0.3027 |
| Y4-011             |               3 |               2 | True                   |             154.08 |          0.07889 |            0.4538 |
| Y4-013             |               3 |               2 | True                   |              52.16 |          0.05473 |            0.3573 |
| Y4-023             |               3 |               2 | False                  |             nan    |          0.05488 |            0.0752 |
| Y4-034             |               3 |               3 | True                   |             180.02 |          0.08475 |            0.4011 |
| Y4-015             |               5 |               5 | False                  |             nan    |          0.04569 |            0.1608 |

## Exploratory associations

Among 90 audio-facial Pearson tests, 0 survive BH FDR at .05. Among 38 feature/outcome tests, 0 survive outcome-specific BH FDR at .05. Lack of an adjusted association does not prove absence of signal in this small pilot sample.

The correlation analysis is participant-level. It should not be described as temporal synchronization because frame-level audio/video dynamics were not correlated.

## Limitations

The sample is small and unbalanced across sessions. Context variables are confounded. Missing pitch is substantial. Sixteen feeling-text entries are marked as unverified word fills pending provenance confirmation. Four recordings were intentionally trimmed to avoid background music; speech completeness and overlap status remain to be confirmed. MediaPipe blendshapes and derived composites are model-based expression proxies, not validated FACS coding or objective emotion measurements.
