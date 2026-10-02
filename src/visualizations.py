"""
Modular Visualization Engine for Bisaya Affect Recognition.
Generates publication-quality, 300-DPI exploratory figures categorized into:
- Audio-only figures: Acoustic and prosodic biomarkers, pitch dynamics, spectral profiles, MFCCs, session differences.
- Facial-only figures: MediaPipe FACS Action Unit proxies, EAR/MAR ratios, head pose dynamics, expressiveness.
- Multimodal figures: Cross-modal correlation matrices, joint predictor embeddings, context confounding profiles.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from src.multimodal_dataset import predictor_columns
from src.statistics_analysis import benjamini_hochberg

# Plot aesthetic configuration
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

VALENCE_PALETTE = {"Negative": "#D95F59", "Neutral": "#6C7A89", "Positive": "#2A9D8F"}
AROUSAL_PALETTE = {"Low": "#457B9D", "Moderate": "#F4A261", "High": "#E76F51"}
SESSION_PALETTE = {"AM": "#E76F51", "PM": "#457B9D"}
ACTIVITY_PALETTE = {"Prelim Exam": "#E76F51", "Quiz": "#F4A261", "Lecture / Review": "#2A9D8F"}


def _save_to_dirs(fig, output_dirs: list[Path], filename: str):
    """Saves figure to all specified output directories at 300 DPI."""
    fig.tight_layout()
    for d in output_dirs:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / filename, bbox_inches="tight", dpi=300)
    plt.close(fig)


# =========================================================================
# 1. AUDIO-ONLY VISUALIZATIONS (Primary Deliverable Focus)
# =========================================================================

def generate_audio_visualizations(df: pd.DataFrame, output_dirs: list[Path]):
    """Generates 10 comprehensive audio acoustic and prosodic figures."""
    print("  -> Generating Audio-Only Figures...")

    # Figure 1: Sample Composition & Durations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # (a) Sessions
    sessions = df.session.value_counts()
    axes[0, 0].pie(
        sessions, labels=[f"{k} (n={v})" for k, v in sessions.items()],
        autopct="%1.1f%%", colors=[SESSION_PALETTE.get(k, "#999") for k in sessions.index],
        startangle=120, textprops={'fontsize': 10}
    )
    axes[0, 0].set_title("(a) Class Session Composition", fontweight="bold")
    
    # (b) Activities
    acts = df.class_activity.value_counts()
    axes[0, 1].bar(acts.index, acts.values, color=[ACTIVITY_PALETTE.get(a, "#577590") for a in acts.index], edgecolor="black")
    axes[0, 1].set_title("(b) Class Activity Breakdown", fontweight="bold")
    axes[0, 1].set_ylabel("Participant Count")
    for i, v in enumerate(acts.values):
        axes[0, 1].text(i, v + 0.5, f"n={v}", ha="center", fontsize=9)
    
    # (c) Frequent Words
    words = df.spoken_word.value_counts().head(10).sort_values()
    axes[1, 0].barh(words.index, words.values, color="#577590", edgecolor="black")
    axes[1, 0].set_title("(c) Frequent Spoken Bisaya Words (Top 10)", fontweight="bold")
    axes[1, 0].set_xlabel("Observations")
    for i, v in enumerate(words.values):
        axes[1, 0].text(v + 0.2, i, f"{v}", va="center", fontsize=9)
    
    # (d) Duration distribution
    sns.histplot(df.audio_duration_sec, kde=True, color="#2A9D8F", ax=axes[1, 1], bins=10)
    mean_dur = df.audio_duration_sec.mean()
    axes[1, 1].axvline(mean_dur, color="#D95F59", ls="--", lw=2, label=f"Mean: {mean_dur:.2f}s")
    axes[1, 1].set_title("(d) Utterance Duration Distribution", fontweight="bold")
    axes[1, 1].set_xlabel("Duration (seconds)")
    axes[1, 1].legend()

    fig.suptitle("Bisaya Speech Dataset Composition & Utterance Durations (N=40)\nSpoken words represent natural lexical context; not assumed emotion labels", fontsize=13, fontweight="bold")
    _save_to_dirs(fig, output_dirs, "01_audio_sample_distribution.png")

    # Figure 2: Empirical Affect Circumplex Space
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axhline(3, color="gray", ls="--", lw=1.2, alpha=0.7)
    ax.axvline(3, color="gray", ls="--", lw=1.2, alpha=0.7)
    
    # Quadrant annotations
    ax.text(1.2, 4.8, "High Arousal / Negative\n(Stress / Anxiety / Frustration)", fontsize=9, color="#D95F59", fontweight="bold", alpha=0.85)
    ax.text(3.8, 4.8, "High Arousal / Positive\n(Excitement / Joy / Thrilled)", fontsize=9, color="#2A9D8F", fontweight="bold", alpha=0.85)
    ax.text(1.2, 1.3, "Low Arousal / Negative\n(Fatigue / Exhaustion / Sadness)", fontsize=9, color="#6C7A89", fontweight="bold", alpha=0.85)
    ax.text(3.8, 1.3, "Low Arousal / Positive\n(Relief / Calm / Relaxation)", fontsize=9, color="#457B9D", fontweight="bold", alpha=0.85)

    grouped = df.groupby("spoken_word").agg(
        Valence=("valence_score", "mean"),
        Arousal=("arousal_score", "mean"),
        N=("participant_code", "size"),
        Mean_F0=("audio_f0_mean_hz", "mean")
    ).reset_index()

    for _, row in grouped.iterrows():
        # Assign color based on valence quadrant
        c = "#D95F59" if row.Valence < 2.8 else "#2A9D8F" if row.Valence > 3.2 else "#6C7A89"
        size = 100 + 50 * row.N
        ax.scatter(row.Valence, row.Arousal, s=size, color=c, alpha=0.8, edgecolor="black", zorder=4)
        ax.annotate(f"{row.spoken_word} (n={row.N})", (row.Valence, row.Arousal), xytext=(5, 5), textcoords="offset points", fontsize=8.5, fontweight="medium")

    ax.set(xlim=(0.8, 5.2), ylim=(0.8, 5.2), xlabel="Mean Self-Reported Valence (1 = Unpleasant, 5 = Pleasant)", ylabel="Mean Self-Reported Arousal (1 = Low Energy, 5 = High Energy)")
    ax.set_title("Empirical Affective Circumplex of Bisaya Spoken Responses\nPositioned strictly by participant self-reports (bubble size = word frequency)", fontsize=12, fontweight="bold")
    _save_to_dirs(fig, output_dirs, "02_empirical_affect_circumplex_audio.png")

    # Figure 3: Prosodic Features by Session with FDR
    prosody = [
        ("audio_f0_mean_hz", "Pitch F0 Mean (Hz)", "F0 (Hz)"),
        ("audio_rms_mean", "RMS Energy (Intensity)", "RMS Amplitude"),
        ("audio_zcr_mean", "Zero-Crossing Rate", "Rate"),
        ("audio_duration_sec", "Speech Duration (s)", "Seconds"),
    ]
    p_values = []
    for col, _, _ in prosody:
        am = df.loc[df.session.eq("AM"), col].dropna()
        pm = df.loc[df.session.eq("PM"), col].dropna()
        p_values.append(stats.ttest_ind(am, pm, equal_var=False).pvalue)
    q_values = benjamini_hochberg(p_values)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, (col, title, ylab), q in zip(axes.flat, prosody, q_values):
        sns.boxplot(data=df, x="session", y=col, hue="session", palette=SESSION_PALETTE, legend=False, ax=ax, width=0.45)
        sns.stripplot(data=df, x="session", y=col, color="black", alpha=0.55, size=6, ax=ax)
        sig = " (q < .05)*" if q < 0.05 else " (n.s.)"
        ax.set_title(f"{title}\nWelch BH q = {q:.3e}{sig}", fontweight="bold", fontsize=11)
        ax.set(xlabel="Class Session", ylabel=ylab)
    fig.suptitle("Acoustic & Prosodic Features by Class Session (AM vs. PM)\nCaution: Session is completely confounded with subject and activity in this cohort", fontsize=13, fontweight="bold")
    _save_to_dirs(fig, output_dirs, "03_prosodic_features_by_session.png")

    # Figure 4: MFCC Timbral Profiles
    mfcc_cols = [f"audio_mfcc_{i}_mean" for i in range(1, 14)]
    fig, axes = plt.subplots(2, 1, figsize=(13, 10))
    
    # (a) By Valence Group
    matrix_v = df.groupby("valence_group")[mfcc_cols].mean().loc[["Negative", "Neutral", "Positive"]]
    matrix_v.columns = [f"MFCC {i}" for i in range(1, 14)]
    sns.heatmap(matrix_v, cmap="coolwarm", center=0, annot=True, fmt=".1f", ax=axes[0], cbar_kws={'label': 'Coefficient Value'})
    axes[0].set_title("(a) Mean MFCC Profiles by Self-Reported Valence Group", fontweight="bold")
    axes[0].set_ylabel("Valence Group")
    
    # (b) By Top Spoken Words
    top_words = df.spoken_word.value_counts().head(6).index
    matrix_w = df[df.spoken_word.isin(top_words)].groupby("spoken_word")[mfcc_cols].mean().loc[top_words]
    matrix_w.columns = [f"MFCC {i}" for i in range(1, 14)]
    sns.heatmap(matrix_w, cmap="coolwarm", center=0, annot=True, fmt=".1f", ax=axes[1], cbar_kws={'label': 'Coefficient Value'})
    axes[1].set_title("(b) Mean MFCC Profiles for Frequent Spoken Words", fontweight="bold")
    axes[1].set_ylabel("Spoken Word")
    axes[1].set_xlabel("Mel-Frequency Cepstral Coefficients (1–13)")

    fig.suptitle("Timbral & Spectral Envelope Representation Across Affect and Words", fontsize=13, fontweight="bold")
    _save_to_dirs(fig, output_dirs, "04_mfcc_timbral_profiles.png")

    # Figure 5: Spectral Biomarkers by Valence and Arousal
    spectral = [
        ("audio_spec_centroid_mean", "Spectral Centroid (Hz) [Brightness]"),
        ("audio_spec_bandwidth_mean", "Spectral Bandwidth (Hz) [Spread]"),
        ("audio_spec_rolloff85_mean", "Spectral Rolloff 85% (Hz) [High-Freq]"),
        ("audio_spec_contrast_mean", "Spectral Contrast [Peak-to-Valley]"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (col, title) in zip(axes.flat, spectral):
        sns.boxplot(data=df, x="valence_group", y=col, order=["Negative", "Neutral", "Positive"], hue="valence_group", palette=VALENCE_PALETTE, legend=False, ax=ax, width=0.45)
        sns.stripplot(data=df, x="valence_group", y=col, order=["Negative", "Neutral", "Positive"], color="black", alpha=0.5, size=6, ax=ax)
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set(xlabel="Self-Reported Valence Group", ylabel="Hz / Magnitude")
    fig.suptitle("Spectral Biomarkers Grouped by Self-Reported Valence", fontsize=13, fontweight="bold")
    _save_to_dirs(fig, output_dirs, "05_spectral_biomarkers_by_affect.png")

    # Figure 6: Pitch Dynamics and Voicing
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    # (a) F0 Mean by Valence
    f0_valid = df.dropna(subset=["audio_f0_mean_hz"])
    sns.boxplot(data=f0_valid, x="valence_group", y="audio_f0_mean_hz", order=["Negative", "Neutral", "Positive"], hue="valence_group", palette=VALENCE_PALETTE, legend=False, ax=axes[0], width=0.45)
    sns.stripplot(data=f0_valid, x="valence_group", y="audio_f0_mean_hz", order=["Negative", "Neutral", "Positive"], color="black", alpha=0.6, ax=axes[0])
    axes[0].set_title(f"(a) Pitch F0 Mean (N={len(f0_valid)} valid)", fontweight="bold")
    axes[0].set(xlabel="Valence Group", ylabel="Fundamental Frequency (Hz)")
    
    # (b) Voiced Ratio by Arousal
    sns.boxplot(data=df, x="arousal_group", y="audio_voiced_ratio", order=["Low", "Moderate", "High"], hue="arousal_group", palette=AROUSAL_PALETTE, legend=False, ax=axes[1], width=0.45)
    sns.stripplot(data=df, x="arousal_group", y="audio_voiced_ratio", order=["Low", "Moderate", "High"], color="black", alpha=0.6, ax=axes[1])
    axes[1].set_title("(b) Voiced Frame Ratio by Arousal", fontweight="bold")
    axes[1].set(xlabel="Arousal Group", ylabel="Voiced Fraction (0–1)")

    # (c) F0 Range
    sns.boxplot(data=f0_valid, x="valence_group", y="audio_f0_range_hz", order=["Negative", "Neutral", "Positive"], hue="valence_group", palette=VALENCE_PALETTE, legend=False, ax=axes[2], width=0.45)
    sns.stripplot(data=f0_valid, x="valence_group", y="audio_f0_range_hz", order=["Negative", "Neutral", "Positive"], color="black", alpha=0.6, ax=axes[2])
    axes[2].set_title(f"(c) Pitch Range (F0 max - F0 min) [Hz]", fontweight="bold")
    axes[2].set(xlabel="Valence Group", ylabel="Pitch Range (Hz)")

    fig.suptitle("Vocal Pitch Dynamics & Voicing Ratios Across Self-Reported Affect\nNote: Missing pitch is preserved as NA (not 0 Hz) across 12 unvoiced recordings", fontsize=13, fontweight="bold")
    _save_to_dirs(fig, output_dirs, "06_pitch_f0_dynamics_and_voicing.png")

    # Figure 7: Lexically Controlled Analysis (Kapoy, N=11)
    kapoy = df[df.spoken_word.eq("Kapoy")].sort_values(["valence_score", "arousal_score", "participant_code"]).copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    colors = kapoy.valence_group.map(VALENCE_PALETTE)
    
    # Subplot 1: F0
    axes[0].barh(kapoy.participant_code, kapoy["audio_f0_mean_hz"].fillna(0), color=colors, edgecolor="black")
    for y, missing in enumerate(kapoy["audio_f0_mean_hz"].isna()):
        if missing:
            axes[0].text(5, y, "NA (Unvoiced)", va="center", fontsize=8, color="darkred", fontweight="bold")
    axes[0].set_title("(a) Pitch F0 (Hz)", fontweight="bold")
    axes[0].set_xlabel("F0 (Hz)")
    axes[0].set_ylabel("Participant Code")

    # Subplot 2: RMS
    axes[1].barh(kapoy.participant_code, kapoy["audio_rms_mean"], color=colors, edgecolor="black")
    axes[1].set_title("(b) Acoustic RMS Energy (Intensity)", fontweight="bold")
    axes[1].set_xlabel("RMS Amplitude")
    axes[1].set_yticks([])

    # Subplot 3: Duration
    axes[2].barh(kapoy.participant_code, kapoy["audio_duration_sec"], color=colors, edgecolor="black")
    axes[2].set_title("(c) Utterance Duration (s)", fontweight="bold")
    axes[2].set_xlabel("Seconds")
    axes[2].set_yticks([])

    # Legend handles
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=VALENCE_PALETTE[k], edgecolor='black', label=f"Valence: {k}") for k in ["Negative", "Neutral", "Positive"]]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05), fontsize=10)

    fig.suptitle("Lexically Controlled Analysis: Acoustic Heterogeneity Within 'Kapoy' (N=11)\nDemonstrating that acoustic characteristics vary across self-reported states even with identical lexical content", fontsize=12, fontweight="bold")
    _save_to_dirs(fig, output_dirs, "07_lexically_controlled_kapoy_acoustics.png")

    # Figure 8: Audio Predictor-Only Unsupervised Clustering (PCA & t-SNE)
    audio_features = [c for c in predictor_columns(df, modality="audio") if df[c].notna().any() and df[c].nunique(dropna=True) > 1]
    X_audio = SimpleImputer(strategy="median").fit_transform(df[audio_features])
    X_audio_scaled = StandardScaler().fit_transform(X_audio)
    
    pca_audio = PCA(n_components=2, random_state=42).fit(X_audio_scaled)
    x_pca = pca_audio.transform(X_audio_scaled)
    tsne_audio = TSNE(n_components=2, perplexity=min(10, len(df) - 1), random_state=42, max_iter=1000, init="pca", learning_rate="auto")
    x_tsne = tsne_audio.fit_transform(X_audio_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for group in ["Negative", "Neutral", "Positive"]:
        mask = df.valence_group.eq(group).to_numpy()
        axes[0].scatter(x_pca[mask, 0], x_pca[mask, 1], label=group, color=VALENCE_PALETTE[group], s=70, edgecolor="black", alpha=0.85)
        axes[1].scatter(x_tsne[mask, 0], x_tsne[mask, 1], label=group, color=VALENCE_PALETTE[group], s=70, edgecolor="black", alpha=0.85)
        
    var_exp = pca_audio.explained_variance_ratio_
    axes[0].set_title(f"Acoustic PCA (PC1: {var_exp[0]*100:.1f}%, PC2: {var_exp[1]*100:.1f}%)", fontweight="bold")
    axes[0].set(xlabel="Principal Component 1", ylabel="Principal Component 2")
    axes[0].legend(title="Valence Group")
    
    axes[1].set_title("Acoustic t-SNE Manifold Projection", fontweight="bold")
    axes[1].set(xlabel="t-SNE Dimension 1", ylabel="t-SNE Dimension 2")
    axes[1].legend(title="Valence Group")

    fig.suptitle(f"Acoustic Predictor-Only Embeddings ({len(audio_features)} Acoustic Features)\nReference ratings strictly excluded from projection inputs to prevent leakage", fontsize=13, fontweight="bold")
    _save_to_dirs(fig, output_dirs, "08_audio_predictor_pca_tsne_clustering.png")

    # Figure 9: Acoustic Inter-Feature Correlation Matrix
    key_audio = [
        "audio_f0_mean_hz", "audio_f0_range_hz", "audio_rms_mean", "audio_rms_std",
        "audio_zcr_mean", "audio_spec_centroid_mean", "audio_spec_bandwidth_mean",
        "audio_spec_rolloff85_mean", "audio_spec_contrast_mean", "audio_duration_sec",
        "audio_period_variation_proxy"
    ]
    labels_clean = [
        "F0 Mean", "F0 Range", "RMS Mean", "RMS Std", "ZCR",
        "Spec Centroid", "Spec Bandwidth", "Rolloff 85%", "Spec Contrast",
        "Duration", "Period Variation"
    ]
    df_sub = df[key_audio].copy()
    df_sub.columns = labels_clean
    corr_mat = df_sub.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_mat, cmap="vlag", center=0, vmin=-1.0, vmax=1.0, annot=True, fmt=".2f", ax=ax, cbar_kws={'label': 'Pearson Correlation (r)'})
    ax.set_title("Acoustic Feature Inter-Correlation Matrix\n(Participant-level associations across speech biomarkers)", fontsize=12, fontweight="bold")
    _save_to_dirs(fig, output_dirs, "09_acoustic_feature_correlation_matrix.png")

    # Figure 10: Educational Context Acoustic Shifts
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    # RMS by Activity
    sns.barplot(data=df, x="class_activity", y="audio_rms_mean", hue="class_activity", palette=ACTIVITY_PALETTE, legend=False, ax=axes[0], edgecolor="black", capsize=0.1)
    axes[0].set_title("(a) Acoustic RMS Energy by Class Activity", fontweight="bold")
    axes[0].set(xlabel="Class Activity", ylabel="Mean RMS Energy")
    axes[0].tick_params(axis='x', rotation=15)

    # Duration by Activity
    sns.barplot(data=df, x="class_activity", y="audio_duration_sec", hue="class_activity", palette=ACTIVITY_PALETTE, legend=False, ax=axes[1], edgecolor="black", capsize=0.1)
    axes[1].set_title("(b) Utterance Duration by Class Activity", fontweight="bold")
    axes[1].set(xlabel="Class Activity", ylabel="Mean Duration (s)")
    axes[1].tick_params(axis='x', rotation=15)

    # Heatmap of ratings across session x activity
    piv = df.pivot_table(index="class_activity", columns="session", values="valence_score", aggfunc="mean")
    sns.heatmap(piv, annot=True, cmap="YlGnBu", vmin=1, vmax=5, ax=axes[2], cbar_kws={'label': 'Mean Valence (1-5)'})
    axes[2].set_title("(c) Mean Valence: Activity x Session Confounding", fontweight="bold")
    axes[2].set(xlabel="Session", ylabel="Class Activity")

    fig.suptitle("Acoustic Shift and Confounding Across Educational Contexts", fontsize=13, fontweight="bold")
    _save_to_dirs(fig, output_dirs, "10_educational_context_acoustic_shifts.png")
    print("  [OK] 10 Audio-Only Figures Generated.")


# =========================================================================
# 2. FACIAL-ONLY VISUALIZATIONS
# =========================================================================

def generate_facial_visualizations(df: pd.DataFrame, output_dirs: list[Path]):
    """Generates 5 dedicated facial Action Unit and landmark figures."""
    print("  -> Generating Facial-Only Figures...")

    # Figure 1: Action Units by Valence
    facial_aus = [
        ("face_smile_mean", "Smile Proxy (AU12 Lip Corner Puller)"),
        ("face_frown_mean", "Frown Proxy (AU15 Depressor)"),
        ("face_brow_lowerer_mean", "Brow Lowerer Proxy (AU4 Strain)"),
        ("face_brow_inner_raiser_mean", "Brow Inner Raiser (AU1 Distress)"),
        ("face_jaw_open_mean", "Jaw Open Proxy (AU26/27 Articulation)"),
        ("face_eye_squint_mean", "Eye Squint Proxy (AU7)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (col, title) in zip(axes.flat, facial_aus):
        sns.boxplot(data=df, x="valence_group", y=col, order=["Negative", "Neutral", "Positive"], hue="valence_group", palette=VALENCE_PALETTE, legend=False, ax=ax, width=0.45)
        sns.stripplot(data=df, x="valence_group", y=col, order=["Negative", "Neutral", "Positive"], color="black", alpha=0.45, size=6, ax=ax)
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set(xlabel="Self-Reported Valence Group", ylabel="Mean Blendshape Score [0-1]")
    fig.suptitle("MediaPipe FACS Blendshape Proxies Grouped by Self-Reported Valence\n(Derived proxies, not objective FACS ground truth)", fontsize=13, fontweight="bold")
    _save_to_dirs(fig, output_dirs, "01_facial_action_units_by_valence.png")

    # Figure 2: Geometric Landmark Ratios (EAR & MAR)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.boxplot(data=df, x="valence_group", y="face_ear_mean", order=["Negative", "Neutral", "Positive"], hue="valence_group", palette=VALENCE_PALETTE, legend=False, ax=axes[0], width=0.45)
    sns.stripplot(data=df, x="valence_group", y="face_ear_mean", order=["Negative", "Neutral", "Positive"], color="black", alpha=0.5, ax=axes[0])
    axes[0].set_title("(a) Eye Aspect Ratio (EAR) across Valence Groups", fontweight="bold")
    axes[0].set(xlabel="Valence Group", ylabel="EAR (2D Pixel Ratio)")

    sns.boxplot(data=df, x="valence_group", y="face_mar_mean", order=["Negative", "Neutral", "Positive"], hue="valence_group", palette=VALENCE_PALETTE, legend=False, ax=axes[1], width=0.45)
    sns.stripplot(data=df, x="valence_group", y="face_mar_mean", order=["Negative", "Neutral", "Positive"], color="black", alpha=0.5, ax=axes[1])
    axes[1].set_title("(b) Mouth Aspect Ratio (MAR) across Valence Groups", fontweight="bold")
    axes[1].set(xlabel="Valence Group", ylabel="MAR (2D Pixel Ratio)")
    fig.suptitle("Geometric 2D Pixel Landmark Ratios by Affective State", fontsize=13, fontweight="bold")
    _save_to_dirs(fig, output_dirs, "02_geometric_ratios_ear_mar.png")

    # Figure 3: Head Pose Dynamics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    poses = [("face_pitch_mean_deg", "Pitch (Nodding) [deg]"), ("face_yaw_mean_deg", "Yaw (Turning) [deg]"), ("face_roll_mean_deg", "Roll (Tilting) [deg]")]
    for ax, (col, title) in zip(axes, poses):
        sns.boxplot(data=df, x="session", y=col, hue="session", palette=SESSION_PALETTE, legend=False, ax=ax, width=0.45)
        sns.stripplot(data=df, x="session", y=col, color="black", alpha=0.5, ax=ax)
        ax.set_title(title, fontweight="bold")
        ax.set(xlabel="Session", ylabel="Degrees")
    fig.suptitle("Head Pose Rotation Dynamics by Class Session (AM vs. PM)", fontsize=13, fontweight="bold")
    _save_to_dirs(fig, output_dirs, "03_head_pose_dynamics.png")

    # Figure 4: Expression Variability & Disgust Proxy
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.boxplot(data=df, x="arousal_group", y="face_expressiveness_proxy", order=["Low", "Moderate", "High"], hue="arousal_group", palette=AROUSAL_PALETTE, legend=False, ax=axes[0], width=0.45)
    sns.stripplot(data=df, x="arousal_group", y="face_expressiveness_proxy", order=["Low", "Moderate", "High"], color="black", alpha=0.5, ax=axes[0])
    axes[0].set_title("(a) Blendshape Dynamic Variability by Arousal", fontweight="bold")
    axes[0].set(xlabel="Arousal Group", ylabel="Standard Deviation across Blendshapes")

    sns.boxplot(data=df, x="valence_group", y="face_disgust_proxy_mean", order=["Negative", "Neutral", "Positive"], hue="valence_group", palette=VALENCE_PALETTE, legend=False, ax=axes[1], width=0.45)
    sns.stripplot(data=df, x="valence_group", y="face_disgust_proxy_mean", order=["Negative", "Neutral", "Positive"], color="black", alpha=0.5, ax=axes[1])
    axes[1].set_title("(b) Disgust Proxy (AU9/10 Nose/Lip) by Valence", fontweight="bold")
    axes[1].set(xlabel="Valence Group", ylabel="Composite Score")
    fig.suptitle("Facial Expressiveness & Affect Proxies", fontsize=13, fontweight="bold")
    _save_to_dirs(fig, output_dirs, "04_facial_expression_variability.png")

    # Figure 5: Facial Predictor PCA & t-SNE
    facial_features = [c for c in predictor_columns(df, modality="facial") if df[c].notna().any() and df[c].nunique(dropna=True) > 1]
    X_face = SimpleImputer(strategy="median").fit_transform(df[facial_features])
    X_face_scaled = StandardScaler().fit_transform(X_face)
    pca_face = PCA(n_components=2, random_state=42).fit(X_face_scaled)
    x_face_pca = pca_face.transform(X_face_scaled)
    tsne_face = TSNE(n_components=2, perplexity=min(10, len(df) - 1), random_state=42, max_iter=1000, init="pca", learning_rate="auto")
    x_face_tsne = tsne_face.fit_transform(X_face_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for group in ["Negative", "Neutral", "Positive"]:
        mask = df.valence_group.eq(group).to_numpy()
        axes[0].scatter(x_face_pca[mask, 0], x_face_pca[mask, 1], label=group, color=VALENCE_PALETTE[group], s=70, edgecolor="black", alpha=0.85)
        axes[1].scatter(x_face_tsne[mask, 0], x_face_tsne[mask, 1], label=group, color=VALENCE_PALETTE[group], s=70, edgecolor="black", alpha=0.85)
        
    v_exp = pca_face.explained_variance_ratio_
    axes[0].set_title(f"Facial PCA (PC1: {v_exp[0]*100:.1f}%, PC2: {v_exp[1]*100:.1f}%)", fontweight="bold")
    axes[0].set(xlabel="Principal Component 1", ylabel="Principal Component 2")
    axes[0].legend(title="Valence Group")
    axes[1].set_title("Facial t-SNE Manifold Projection", fontweight="bold")
    axes[1].set(xlabel="t-SNE Dimension 1", ylabel="t-SNE Dimension 2")
    axes[1].legend(title="Valence Group")
    fig.suptitle(f"Facial Predictor-Only Embeddings ({len(facial_features)} Blendshape/Landmark Features)\nReference ratings strictly excluded from projection inputs", fontsize=13, fontweight="bold")
    _save_to_dirs(fig, output_dirs, "05_facial_predictor_pca_tsne.png")
    print("  [OK] 5 Facial-Only Figures Generated.")


# =========================================================================
# 3. MULTIMODAL VISUALIZATIONS (Cross-Modal Exploration)
# =========================================================================

def generate_multimodal_visualizations(df: pd.DataFrame, output_dirs: list[Path]):
    """Generates 4 multimodal cross-modal correlation and joint embedding figures."""
    print("  -> Generating Multimodal Figures...")

    # Figure 1: Cross-Modal Correlation Heatmap
    audio_cols = ["audio_f0_mean_hz", "audio_f0_range_hz", "audio_rms_mean", "audio_zcr_mean", "audio_spec_centroid_mean", "audio_spec_rolloff85_mean", "audio_duration_sec"]
    face_cols = ["face_smile_mean", "face_frown_mean", "face_brow_lowerer_mean", "face_ear_mean", "face_mar_mean", "face_expressiveness_proxy", "face_pitch_mean_deg"]
    
    corr_df = pd.DataFrame(index=[c.replace("audio_", "") for c in audio_cols], columns=[c.replace("face_", "") for c in face_cols], dtype=float)
    for a in audio_cols:
        for f in face_cols:
            if a in df and f in df:
                corr_df.loc[a.replace("audio_", ""), f.replace("face_", "")] = df[[a, f]].dropna().corr().iloc[0, 1]

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr_df, cmap="vlag", center=0, vmin=-0.6, vmax=0.6, annot=True, fmt=".2f", ax=ax, cbar_kws={'label': 'Pearson Correlation (r)'})
    ax.set_title("Cross-Modal Audio-Facial Participant-Level Correlations\n(Participant-level alignment; does not imply frame-level temporal synchronization)", fontsize=12, fontweight="bold")
    ax.set(xlabel="Facial Expression Proxy", ylabel="Acoustic Biomarker")
    _save_to_dirs(fig, output_dirs, "01_audio_facial_correlation_heatmap.png")

    # Figure 2: Joint Multimodal PCA & t-SNE
    all_features = [c for c in predictor_columns(df, modality="multimodal") if df[c].notna().any() and df[c].nunique(dropna=True) > 1]
    X_multi = SimpleImputer(strategy="median").fit_transform(df[all_features])
    X_multi_scaled = StandardScaler().fit_transform(X_multi)
    
    pca_multi = PCA(n_components=2, random_state=42).fit(X_multi_scaled)
    x_m_pca = pca_multi.transform(X_multi_scaled)
    tsne_multi = TSNE(n_components=2, perplexity=min(10, len(df) - 1), random_state=42, max_iter=1000, init="pca", learning_rate="auto")
    x_m_tsne = tsne_multi.fit_transform(X_multi_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for group in ["Negative", "Neutral", "Positive"]:
        mask = df.valence_group.eq(group).to_numpy()
        axes[0].scatter(x_m_pca[mask, 0], x_m_pca[mask, 1], label=group, color=VALENCE_PALETTE[group], s=70, edgecolor="black", alpha=0.85)
        axes[1].scatter(x_m_tsne[mask, 0], x_m_tsne[mask, 1], label=group, color=VALENCE_PALETTE[group], s=70, edgecolor="black", alpha=0.85)

    m_exp = pca_multi.explained_variance_ratio_
    axes[0].set_title(f"Joint Multimodal PCA (PC1: {m_exp[0]*100:.1f}%, PC2: {m_exp[1]*100:.1f}%)", fontweight="bold")
    axes[0].set(xlabel="Principal Component 1", ylabel="Principal Component 2")
    axes[0].legend(title="Valence Group")
    axes[1].set_title("Joint Multimodal t-SNE Projection", fontweight="bold")
    axes[1].set(xlabel="Dimension 1", ylabel="Dimension 2")
    axes[1].legend(title="Valence Group")
    fig.suptitle(f"Joint Multimodal Predictor Embeddings ({len(all_features)} Acoustic + Facial Features)\nReference outcomes strictly excluded from projection inputs", fontsize=13, fontweight="bold")
    _save_to_dirs(fig, output_dirs, "02_multimodal_joint_pca_tsne.png")

    # Figure 3: Multimodal Kapoy Profile
    kapoy = df[df.spoken_word.eq("Kapoy")].sort_values(["valence_score", "arousal_score", "participant_code"]).copy()
    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    colors = kapoy.valence_group.map(VALENCE_PALETTE)
    
    axes[0].barh(kapoy.participant_code, kapoy["audio_rms_mean"], color=colors, edgecolor="black")
    axes[0].set_title("(a) Acoustic RMS Energy", fontweight="bold")
    axes[0].set_xlabel("RMS Amplitude")
    axes[0].set_ylabel("Participant Code")

    axes[1].barh(kapoy.participant_code, kapoy["face_smile_mean"], color=colors, edgecolor="black")
    axes[1].set_title("(b) Smile Proxy (AU12)", fontweight="bold")
    axes[1].set_xlabel("Blendshape Score")
    axes[1].set_yticks([])

    axes[2].barh(kapoy.participant_code, kapoy["face_brow_lowerer_mean"], color=colors, edgecolor="black")
    axes[2].set_title("(c) Brow Lowerer (AU4)", fontweight="bold")
    axes[2].set_xlabel("Blendshape Score")
    axes[2].set_yticks([])

    axes[3].barh(kapoy.participant_code, kapoy["face_mar_mean"], color=colors, edgecolor="black")
    axes[3].set_title("(d) Mouth Aspect Ratio (MAR)", fontweight="bold")
    axes[3].set_xlabel("Ratio")
    axes[3].set_yticks([])

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=VALENCE_PALETTE[k], edgecolor='black', label=f"Valence: {k}") for k in ["Negative", "Neutral", "Positive"]]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05), fontsize=10)
    fig.suptitle("Cross-Modal Lexically Controlled Profile: 'Kapoy' (N=11)\nAcoustic Intensity vs. Facial Action Unit Proxies", fontsize=13, fontweight="bold")
    _save_to_dirs(fig, output_dirs, "03_multimodal_lexically_controlled_kapoy.png")

    # Figure 4: Educational Context Matrix
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, outcome, title in [(axes[0], "valence_score", "Mean Valence"), (axes[1], "arousal_score", "Mean Arousal")]:
        table = df.pivot_table(index="class_activity", columns="session", values=outcome, aggfunc="mean")
        sns.heatmap(table, annot=True, vmin=1, vmax=5, cmap="viridis", ax=ax, cbar_kws={"label": "Rating (1–5)"})
        ax.set(title=title, xlabel="Session", ylabel="Class Activity")
    fig.suptitle("Educational Context Confounding: Activity x Session\nEmpty cells show structural absence of PM exams/quizzes in sample", fontsize=13, fontweight="bold")
    _save_to_dirs(fig, output_dirs, "04_educational_context_affect_profiles.png")
    print("  [OK] 4 Multimodal Figures Generated.")


# =========================================================================
# MASTER GENERATOR
# =========================================================================

def generate_all_visualizations(
    dataset_csv: str = "datasets/multimodal/cleaned_multimodal_dataset.csv",
    output_dir: str = "outputs/visualizations",
):
    """
    Generates all figures into:
    1. outputs/audio/ AND outputs/visualizations/audio/ (10 Audio Figures)
    2. outputs/facial/ AND outputs/visualizations/facial/ (5 Facial Figures)
    3. outputs/multimodal/ AND outputs/visualizations/multimodal/ (4 Multimodal Figures)
    """
    df = pd.read_csv(dataset_csv)
    base_vis = Path(output_dir)
    base_outputs = Path("outputs")

    audio_dirs = [base_outputs / "audio", base_vis / "audio"]
    facial_dirs = [base_outputs / "facial", base_vis / "facial"]
    multimodal_dirs = [base_outputs / "multimodal", base_vis / "multimodal"]

    print("=================================================================")
    print(" GENERATING DISAGGREGATED FIGURES (AUDIO, FACIAL, MULTIMODAL)")
    print("=================================================================")
    generate_audio_visualizations(df, audio_dirs)
    generate_facial_visualizations(df, facial_dirs)
    generate_multimodal_visualizations(df, multimodal_dirs)
    print("=================================================================")
    print(f"All figures generated successfully across outputs/ and {output_dir}!")


if __name__ == "__main__":
    generate_all_visualizations()
