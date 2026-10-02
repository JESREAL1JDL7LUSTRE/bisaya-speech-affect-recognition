"""Exploratory figures grounded in self-reported valence and arousal."""

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


sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.family": "DejaVu Sans"})
VALENCE_PALETTE = {"Negative": "#D95F59", "Neutral": "#6C7A89", "Positive": "#2A9D8F"}
SESSION_PALETTE = {"AM": "#E76F51", "PM": "#457B9D"}


def _save(fig, output: Path, filename: str):
    fig.tight_layout()
    fig.savefig(output / filename, bbox_inches="tight")
    plt.close(fig)


def generate_all_visualizations(
    dataset_csv: str = "datasets/multimodal/cleaned_multimodal_dataset.csv",
    output_dir: str = "outputs/visualizations",
):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(dataset_csv)

    # 1. Sample composition
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sessions = df.session.value_counts()
    axes[0].pie(sessions, labels=[f"{k} (n={v})" for k, v in sessions.items()], autopct="%1.1f%%", colors=[SESSION_PALETTE.get(k, "#999") for k in sessions.index], startangle=120)
    axes[0].set_title("Class-session composition")
    words = df.spoken_word.value_counts().head(12).sort_values()
    axes[1].barh(words.index, words.values, color="#577590")
    axes[1].set(title="Most frequent spoken words", xlabel="Observations")
    fig.suptitle(f"Dataset composition (N={len(df)})\nSpoken words are lexical context, not emotion labels", fontweight="bold")
    _save(fig, output, "01_dataset_distribution_overview.png")

    # 2. Empirical valence-arousal space
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axhline(3, color="grey", ls="--", lw=1); ax.axvline(3, color="grey", ls="--", lw=1)
    grouped = df.groupby("spoken_word").agg(Valence=("valence_score", "mean"), Arousal=("arousal_score", "mean"), N=("participant_code", "size")).reset_index()
    for _, row in grouped.iterrows():
        ax.scatter(row.Valence, row.Arousal, s=80 + 45 * row.N, color="#4C78A8", alpha=.75, edgecolor="black")
        ax.annotate(f"{row.spoken_word} (n={row.N})", (row.Valence, row.Arousal), xytext=(4, 5), textcoords="offset points", fontsize=8)
    ax.set(xlim=(0.8, 5.2), ylim=(0.8, 5.2), xlabel="Mean self-reported valence (1–5)", ylabel="Mean self-reported arousal (1–5)", title="Empirical ratings by spoken word")
    _save(fig, output, "02_affect_word_taxonomy.png")

    # 3. Descriptive session plots with FDR-adjusted Welch tests
    prosody = [
        ("audio_f0_mean_hz", "Pitch F0 (Hz)"), ("audio_rms_mean", "RMS energy"),
        ("audio_zcr_mean", "Zero-crossing rate"), ("audio_duration_sec", "Duration (s)"),
    ]
    p_values = []
    for column, _ in prosody:
        am = df.loc[df.session.eq("AM"), column].dropna(); pm = df.loc[df.session.eq("PM"), column].dropna()
        p_values.append(stats.ttest_ind(am, pm, equal_var=False).pvalue)
    q_values = benjamini_hochberg(p_values)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (column, label), q in zip(axes.flat, prosody, q_values):
        sns.boxplot(data=df, x="session", y=column, hue="session", palette=SESSION_PALETTE, legend=False, ax=ax)
        sns.stripplot(data=df, x="session", y=column, color="black", alpha=.55, ax=ax)
        ax.set(title=f"{label} (Welch BH q={q:.3g})", xlabel="Session", ylabel=label)
    fig.suptitle("Speech features by session\nSession is confounded with subject and class activity", fontweight="bold")
    _save(fig, output, "03_audio_prosodic_features_by_session.png")

    # 4. MFCC lexical profiles
    mfcc = [f"audio_mfcc_{i}_mean" for i in range(1, 14)]
    top_words = df.spoken_word.value_counts().head(6).index
    matrix = df[df.spoken_word.isin(top_words)].groupby("spoken_word")[mfcc].mean()
    matrix.columns = [f"MFCC {i}" for i in range(1, 14)]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(matrix, cmap="coolwarm", center=0, ax=ax)
    ax.set(title="Mean MFCC profiles for frequent spoken words", xlabel="Coefficient", ylabel="Spoken word")
    _save(fig, output, "04_mfcc_feature_heatmaps.png")

    # 5. Spectral features by reference valence group
    spectral = [("audio_spec_centroid_mean", "Spectral centroid (Hz)"), ("audio_spec_rolloff85_mean", "85% rolloff (Hz)"), ("audio_spec_contrast_mean", "Spectral contrast")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (column, label) in zip(axes, spectral):
        sns.boxplot(data=df, x="valence_group", y=column, order=["Negative", "Neutral", "Positive"], hue="valence_group", palette=VALENCE_PALETTE, legend=False, ax=ax)
        sns.stripplot(data=df, x="valence_group", y=column, order=["Negative", "Neutral", "Positive"], color="black", alpha=.5, ax=ax)
        ax.set(xlabel="Self-reported valence group", ylabel=label, title=label)
    fig.suptitle("Spectral measurements by self-reported valence group", fontweight="bold")
    _save(fig, output, "05_spectral_characteristics.png")

    # 6. Facial model proxies by reference valence group
    facial = [
        ("face_smile_mean", "Smile blendshape proxy"), ("face_frown_mean", "Frown blendshape proxy"),
        ("face_brow_lowerer_mean", "Brow-lowering proxy"), ("face_ear_mean", "Eye aspect ratio"),
        ("face_mar_mean", "Mouth aspect ratio"), ("face_expressiveness_proxy", "Blendshape variability proxy"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (column, label) in zip(axes.flat, facial):
        sns.boxplot(data=df, x="valence_group", y=column, order=["Negative", "Neutral", "Positive"], hue="valence_group", palette=VALENCE_PALETTE, legend=False, ax=ax)
        sns.stripplot(data=df, x="valence_group", y=column, order=["Negative", "Neutral", "Positive"], color="black", alpha=.45, ax=ax)
        ax.set(xlabel="Valence group", ylabel=label, title=label)
    fig.suptitle("MediaPipe-derived expression proxies by self-reported valence", fontweight="bold")
    _save(fig, output, "06_facial_action_units_by_session.png")

    # 7. Participant-level cross-modal correlations
    audio_cols = ["audio_f0_mean_hz", "audio_f0_range_hz", "audio_rms_mean", "audio_zcr_mean", "audio_spec_centroid_mean", "audio_spec_rolloff85_mean", "audio_duration_sec"]
    face_cols = ["face_smile_mean", "face_frown_mean", "face_brow_lowerer_mean", "face_ear_mean", "face_mar_mean", "face_expressiveness_proxy", "face_pitch_mean_deg"]
    corr = pd.DataFrame(index=[c.replace("audio_", "") for c in audio_cols], columns=[c.replace("face_", "") for c in face_cols], dtype=float)
    for a in audio_cols:
        for f in face_cols:
            corr.loc[a.replace("audio_", ""), f.replace("face_", "")] = df[[a, f]].corr().iloc[0, 1]
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, cmap="vlag", center=0, vmin=-.6, vmax=.6, annot=True, fmt=".2f", ax=ax)
    ax.set(title="Participant-level audio–facial Pearson correlations", xlabel="Facial proxy", ylabel="Audio feature")
    _save(fig, output, "07_audio_facial_multimodal_correlations.png")

    # 8. Predictor-only embeddings; outcomes are used only for color after projection.
    features = predictor_columns(df)
    usable = [c for c in features if df[c].notna().any() and df[c].nunique(dropna=True) > 1]
    x = SimpleImputer(strategy="median").fit_transform(df[usable])
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2, random_state=42).fit(x)
    x_pca = pca.transform(x)
    x_tsne = TSNE(n_components=2, perplexity=min(10, len(df) - 1), random_state=42, max_iter=1000, init="pca", learning_rate="auto").fit_transform(x)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for group in ["Negative", "Neutral", "Positive"]:
        mask = df.valence_group.eq(group).to_numpy()
        axes[0].scatter(x_pca[mask, 0], x_pca[mask, 1], label=group, color=VALENCE_PALETTE[group], edgecolor="black", alpha=.8)
        axes[1].scatter(x_tsne[mask, 0], x_tsne[mask, 1], label=group, color=VALENCE_PALETTE[group], edgecolor="black", alpha=.8)
    axes[0].set(title=f"PCA ({pca.explained_variance_ratio_[0]*100:.1f}% + {pca.explained_variance_ratio_[1]*100:.1f}%)", xlabel="PC1", ylabel="PC2")
    axes[1].set(title="t-SNE descriptive projection", xlabel="Dimension 1", ylabel="Dimension 2")
    for ax in axes: ax.legend(title="Self-reported valence")
    fig.suptitle(f"Predictor-only embeddings ({len(usable)} acoustic/facial features)\nRatings were excluded from projection inputs", fontweight="bold")
    _save(fig, output, "08_multimodal_pca_tsne_clustering.png")

    # 9. Same-word descriptive analysis
    kapoy = df[df.spoken_word.eq("Kapoy")].sort_values(["valence_score", "arousal_score", "participant_code"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    colors = kapoy.valence_group.map(VALENCE_PALETTE)
    for ax, column, label in [
        (axes[0], "audio_f0_mean_hz", "Pitch F0 (Hz; missing if undetected)"),
        (axes[1], "audio_rms_mean", "RMS energy"), (axes[2], "face_smile_mean", "Smile blendshape proxy"),
    ]:
        ax.barh(kapoy.participant_code, kapoy[column].fillna(0), color=colors, edgecolor="black")
        if column == "audio_f0_mean_hz":
            for y, missing in enumerate(kapoy[column].isna()):
                if missing: ax.text(2, y, "NA", va="center", fontsize=8)
        ax.set(xlabel=label, ylabel="Participant")
    fig.suptitle("Lexically controlled description: Kapoy\nVariation is descriptive and does not prove affect prediction", fontweight="bold")
    _save(fig, output, "09_lexically_controlled_analysis.png")

    # 10. Direct context × reference-outcome view
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, outcome, title in [(axes[0], "valence_score", "Mean valence"), (axes[1], "arousal_score", "Mean arousal")]:
        table = df.pivot_table(index="class_activity", columns="session", values=outcome, aggfunc="mean")
        sns.heatmap(table, annot=True, vmin=1, vmax=5, cmap="viridis", ax=ax, cbar_kws={"label": "Rating (1–5)"})
        ax.set(title=title, xlabel="Session", ylabel="Class activity")
    fig.suptitle("Educational context and self-reported affect\nEmpty cells show the session/activity confounding", fontweight="bold")
    _save(fig, output, "10_context_affect_profiles.png")

    print(f"Generated 10 corrected exploratory figures in {output}")


if __name__ == "__main__":
    generate_all_visualizations()
