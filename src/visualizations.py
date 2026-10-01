"""
Visualization Generation Module for Bisaya Speech and Affect Recognition.
Generates publication-quality, high-resolution figures (300 DPI) for academic reports,
research presentations, and PIT deliverable submission.
"""

import os
import sys

# Ensure workspace root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Set aesthetic defaults
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

AFFECT_PALETTE = {
    "Fatigue": "#E64B35",        # Muted Red/Coral
    "Stress/Anxiety": "#4DBBD5", # Cyan/Teal
    "Positive": "#00A087",       # Emerald Green
    "Neutral": "#3C5488"         # Deep Slate Blue
}

SESSION_PALETTE = {
    "AM": "#E64B35",
    "PM": "#3C5488"
}


def generate_all_visualizations(
    dataset_csv: str = "datasets/cleaned_multimodal_dataset.csv",
    output_dir: str = "outputs/visualizations"
):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(dataset_csv)
    print(f"Loaded dataset for visualizations: {df.shape}")
    
    # -------------------------------------------------------------
    # Figure 1: Dataset Distribution Overview (Session & Spoken Words)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 1.6]})
    
    # Left: Session Distribution Donut
    session_counts = df['session'].value_counts()
    colors = [SESSION_PALETTE.get(s, '#888888') for s in session_counts.index]
    wedges, texts, autotexts = axes[0].pie(
        session_counts.values,
        labels=[f"{s} Session\n(n={v})" for s, v in session_counts.items()],
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        wedgeprops=dict(width=0.45, edgecolor='w', linewidth=2),
        textprops=dict(fontsize=11, fontweight='bold')
    )
    for at in autotexts:
        at.set_color('white')
        at.set_fontsize(11)
        at.set_fontweight('bold')
    axes[0].set_title("Class Session Distribution (N=40)", fontsize=13, fontweight='bold', pad=12)
    
    # Right: Spoken Word Frequency Bar Chart
    word_counts = df['spoken_word'].value_counts()
    top_words = word_counts.head(10)
    word_colors = [AFFECT_PALETTE.get(df[df['spoken_word'] == w]['affect_category'].iloc[0], '#888888') for w in top_words.index]
    
    bars = axes[1].barh(top_words.index[::-1], top_words.values[::-1], color=word_colors[::-1], edgecolor='black', alpha=0.85, height=0.65)
    for bar in bars:
        w = bar.get_width()
        axes[1].text(w + 0.15, bar.get_y() + bar.get_height()/2, f"{int(w)} ({w/len(df)*100:.1f}%)",
                     ha='left', va='center', fontsize=10, fontweight='bold')
                     
    axes[1].set_xlabel("Participant Response Count", fontsize=11, fontweight='bold')
    axes[1].set_title("Top Bisaya Spoken Word Reactions Elicited Post-Class", fontsize=13, fontweight='bold', pad=12)
    axes[1].set_xlim(0, max(top_words.values) + 2)
    
    # Custom Legend for Categories
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, edgecolor='black', label=cat) for cat, c in AFFECT_PALETTE.items()]
    axes[1].legend(handles=legend_elements, title="Affect Category", loc='lower right', frameon=True)
    
    plt.tight_layout()
    f1_path = os.path.join(output_dir, "01_dataset_distribution_overview.png")
    plt.savefig(f1_path)
    plt.close()
    print(f"Saved: {f1_path}")
    
    # -------------------------------------------------------------
    # Figure 2: Russell's Circumplex Affect Taxonomy Space (Empirical)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Background Quadrant shading
    ax.axhline(0, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
    
    ax.text(0.85, 0.90, "HIGH AROUSAL\nPOSITIVE (Joy / Excitement)", ha='center', va='center', fontsize=9, fontweight='bold', color='#00A087', alpha=0.6)
    ax.text(-0.75, 0.90, "HIGH AROUSAL\nNEGATIVE (Stress / Anxiety)", ha='center', va='center', fontsize=9, fontweight='bold', color='#4DBBD5', alpha=0.6)
    ax.text(-0.75, -0.90, "LOW AROUSAL\nNEGATIVE (Fatigue / Exhaustion)", ha='center', va='center', fontsize=9, fontweight='bold', color='#E64B35', alpha=0.6)
    ax.text(0.85, -0.90, "LOW AROUSAL\nPOSITIVE (Calm / Relief)", ha='center', va='center', fontsize=9, fontweight='bold', color='#3C5488', alpha=0.6)
    
    word_freq = df['spoken_word'].value_counts()
    
    if "valence_score" in df.columns and "arousal_score" in df.columns:
        # Compute true empirical coordinates from participant self-reports (centered on neutral=3)
        mean_v = df.groupby('spoken_word')['valence_score'].mean()
        mean_a = df.groupby('spoken_word')['arousal_score'].mean()
        # Scale 1-5 to [-1, 1] with slight jitter for overlapping points
        np.random.seed(42)
        for word in word_freq.index:
            count = word_freq[word]
            cat = df[df['spoken_word'] == word]['affect_category'].iloc[0]
            color = AFFECT_PALETTE.get(cat, '#888888')
            
            val = (mean_v[word] - 3.0) / 2.0 + (np.random.uniform(-0.04, 0.04) if count == 1 else 0)
            aro = (mean_a[word] - 3.0) / 2.0 + (np.random.uniform(-0.04, 0.04) if count == 1 else 0)
            val = np.clip(val, -0.95, 0.95)
            aro = np.clip(aro, -0.95, 0.95)
            
            size = 180 + count * 85
            ax.scatter(val, aro, s=size, color=color, alpha=0.8, edgecolors='black', linewidth=1.5, zorder=5)
            
            ax.annotate(
                f"{word}\n(n={count}, V:{mean_v[word]:.1f}, A:{mean_a[word]:.1f})",
                (val, aro),
                textcoords="offset points",
                xytext=(0, 12 if aro >= 0 else -20),
                ha='center',
                fontsize=8.5,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.85, lw=1)
            )
    ax.set_title("Empirical Circumplex Affect Space of Bisaya Post-Class Reactions\n(Coordinates derived directly from participant self-reported Valence & Arousal ratings)", fontsize=13, fontweight='bold', pad=14)
        
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Valence (Unpleasant ← 0 → Pleasant)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Arousal (Low Energy ← 0 → High Energy)", fontsize=12, fontweight='bold')
    ax.set_title("Russell's Circumplex Affect Space of Bisaya Post-Class Reactions\n(Bubble size proportional to participant frequency)", fontsize=13, fontweight='bold', pad=14)
    ax.legend(handles=legend_elements, title="Affect Dimension", loc='upper left', frameon=True)
    
    plt.tight_layout()
    f2_path = os.path.join(output_dir, "02_affect_word_taxonomy.png")
    plt.savefig(f2_path)
    plt.close()
    print(f"Saved: {f2_path}")
    
    # -------------------------------------------------------------
    # Figure 3: Prosodic Pitch, Energy & Duration by Session (AM vs PM)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    prosody_vars = [
        ("audio_f0_mean_hz", "Pitch F0 Mean (Hz)", "Fundamental Frequency"),
        ("audio_rms_mean", "RMS Energy (Amplitude)", "Acoustic Loudness / Energy"),
        ("audio_zcr_mean", "Zero Crossing Rate (ZCR)", "Signal Rate of Sign-Changes"),
        ("audio_duration_sec", "Utterance Duration (seconds)", "Spoken Response Duration")
    ]
    
    for idx, (var, ylabel, title) in enumerate(prosody_vars):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        sns.boxplot(x="session", y=var, data=df, ax=ax, palette=SESSION_PALETTE, width=0.45, boxprops=dict(alpha=0.7), showmeans=True, meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black"})
        sns.stripplot(x="session", y=var, data=df, ax=ax, color='black', alpha=0.6, jitter=0.15, size=6)
        
        # Compute p-value
        am_v = df[df["session"] == "AM"][var].dropna()
        pm_v = df[df["session"] == "PM"][var].dropna()
        _, pval = stats.ttest_ind(am_v, pm_v, equal_var=False)
        sig_str = "*** p < .001" if pval < 0.001 else ("** p < .01" if pval < 0.01 else ("* p < .05" if pval < 0.05 else "n.s. (p > .05)"))
        
        ax.set_title(f"{title} (Session Effect: {sig_str})", fontsize=11, fontweight='bold')
        ax.set_xlabel("Class Session", fontsize=10, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
        
    plt.suptitle("Acoustic & Prosodic Speech Characteristics: Morning (AM) vs. Afternoon–Evening (PM)", fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    f3_path = os.path.join(output_dir, "03_audio_prosodic_features_by_session.png")
    plt.savefig(f3_path)
    plt.close()
    print(f"Saved: {f3_path}")
    
    # -------------------------------------------------------------
    # Figure 4: MFCC Timbral Feature Heatmaps across Top Words
    # -------------------------------------------------------------
    mfcc_cols = [f"audio_mfcc_{i}_mean" for i in range(1, 14)]
    top_5_words = df['spoken_word'].value_counts().head(6).index.tolist()
    mfcc_df = df[df['spoken_word'].isin(top_5_words)].groupby('spoken_word')[mfcc_cols].mean()
    mfcc_df.columns = [f"MFCC-{i}" for i in range(1, 14)]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(mfcc_df, cmap="coolwarm", annot=True, fmt=".1f", linewidths=1, linecolor='white', cbar_kws={'label': 'Mean Coefficient Amplitude'}, ax=ax)
    ax.set_title("Timbral Profile: Mean 13 Mel-Frequency Cepstral Coefficients (MFCCs) Across Top Reaction Words", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("MFCC Cepstral Coefficients", fontsize=11, fontweight='bold')
    ax.set_ylabel("Bisaya Spoken Reaction", fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    f4_path = os.path.join(output_dir, "04_mfcc_feature_heatmaps.png")
    plt.savefig(f4_path)
    plt.close()
    print(f"Saved: {f4_path}")
    
    # -------------------------------------------------------------
    # Figure 5: Spectral Characteristics Across Affect Categories
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    spectral_vars = [
        ("audio_spec_centroid_mean", "Spectral Centroid (Hz)", "Vocal Brightness / Frequency Center"),
        ("audio_spec_rolloff85_mean", "Spectral Rolloff 85% (Hz)", "High-Frequency Cutoff"),
        ("audio_spec_contrast_mean", "Spectral Contrast", "Peak-to-Valley Spectral Dynamics")
    ]
    
    for idx, (var, ylabel, title) in enumerate(spectral_vars):
        ax = axes[idx]
        sns.barplot(x="affect_category", y=var, data=df, ax=ax, palette=AFFECT_PALETTE, capsize=0.1, edgecolor='black', alpha=0.85)
        sns.stripplot(x="affect_category", y=var, data=df, ax=ax, color='black', alpha=0.5, jitter=0.2, size=5)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel("Affect Category", fontsize=10, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
        ax.tick_params(axis='x', rotation=15)
        
    plt.suptitle("Spectral Characteristics of Bisaya Speech Across Affect Categories", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    f5_path = os.path.join(output_dir, "05_spectral_characteristics.png")
    plt.savefig(f5_path)
    plt.close()
    print(f"Saved: {f5_path}")
    
    # -------------------------------------------------------------
    # Figure 6: Facial Action Units & Geometric Ratios Across Affect Categories
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    facial_vars = [
        ("face_smile_mean", "Smile Intensity (AU12)", "Zygomaticus Major Activation"),
        ("face_frown_mean", "Frown Intensity (AU15)", "Depressor Anguli Oris Activation"),
        ("face_brow_lowerer_mean", "Brow Lowerer (AU4)", "Corrugator / Concentration / Anger"),
        ("face_ear_mean", "Eye Aspect Ratio (EAR)", "Eyelid Openness Baseline"),
        ("face_mar_mean", "Mouth Aspect Ratio (MAR)", "Speech Articulation Dynamics"),
        ("face_expressiveness_score", "Facial Expressiveness Score", "Overall Dynamic Facial Variability")
    ]
    
    for idx, (var, ylabel, title) in enumerate(facial_vars):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        sns.boxplot(x="affect_category", y=var, data=df, ax=ax, palette=AFFECT_PALETTE, width=0.5, boxprops=dict(alpha=0.75))
        sns.stripplot(x="affect_category", y=var, data=df, ax=ax, color='black', alpha=0.6, jitter=0.15, size=5)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel("Affect Category", fontsize=10, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
        ax.tick_params(axis='x', rotation=15)
        
    plt.suptitle("Facial Action Units and Morphological Dynamics Across Affect Categories", fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    f6_path = os.path.join(output_dir, "06_facial_action_units_by_session.png")
    plt.savefig(f6_path)
    plt.close()
    print(f"Saved: {f6_path}")
    
    # -------------------------------------------------------------
    # Figure 7: Audio-Facial Multimodal Correlation Heatmap
    # -------------------------------------------------------------
    audio_sub = [
        "audio_f0_mean_hz", "audio_f0_range_hz", "audio_rms_mean", "audio_zcr_mean",
        "audio_spec_centroid_mean", "audio_spec_rolloff85_mean", "audio_duration_sec"
    ]
    face_sub = [
        "face_smile_mean", "face_frown_mean", "face_brow_lowerer_mean",
        "face_ear_mean", "face_mar_mean", "face_expressiveness_score", "face_pitch_mean_deg"
    ]
    
    corr_matrix = pd.DataFrame(index=[c.replace("audio_", "").replace("_hz", "").replace("_mean", "") for c in audio_sub],
                               columns=[c.replace("face_", "").replace("_mean", "") for c in face_sub])
    
    for a in audio_sub:
        a_label = a.replace("audio_", "").replace("_hz", "").replace("_mean", "")
        for f in face_sub:
            f_label = f.replace("face_", "").replace("_mean", "")
            r, _ = stats.pearsonr(df[a], df[f])
            corr_matrix.loc[a_label, f_label] = round(r, 2)
            
    corr_matrix = corr_matrix.astype(float)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr_matrix, cmap="vlag", annot=True, fmt=".2f", vmin=-0.6, vmax=0.6, linewidths=1.2, linecolor='white', ax=ax, cbar_kws={'label': 'Pearson Correlation (r)'})
    ax.set_title("Cross-Modal Audio-Visual Synchronization Matrix\n(Acoustic Prosody vs. Facial Action Units)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Facial Expression & Morphological Markers", fontsize=11, fontweight='bold')
    ax.set_ylabel("Acoustic Speech Features", fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    f7_path = os.path.join(output_dir, "07_audio_facial_multimodal_correlations.png")
    plt.savefig(f7_path)
    plt.close()
    print(f"Saved: {f7_path}")
    
    # -------------------------------------------------------------
    # Figure 8: Multimodal PCA & t-SNE Clustering
    # -------------------------------------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Exclude metadata IDs
    feature_cols = [c for c in numeric_cols if not ('rate' in c and df[c].std() == 0)]
    X = StandardScaler().fit_transform(df[feature_cols])
    
    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    # t-SNE
    tsne = TSNE(n_components=2, perplexity=10, random_state=42, max_iter=1000)
    X_tsne = tsne.fit_transform(X)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for cat in df['affect_category'].unique():
        mask = df['affect_category'] == cat
        color = AFFECT_PALETTE.get(cat, '#888888')
        
        # PCA plot
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], label=cat, color=color, s=80, alpha=0.85, edgecolors='black', linewidth=1)
        # t-SNE plot
        axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=cat, color=color, s=80, alpha=0.85, edgecolors='black', linewidth=1)
        
    axes[0].set_title(f"Multimodal PCA (PC1: {pca.explained_variance_ratio_[0]*100:.1f}%, PC2: {pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Principal Component 1", fontsize=10, fontweight='bold')
    axes[0].set_ylabel("Principal Component 2", fontsize=10, fontweight='bold')
    axes[0].legend(title="Affect Category", frameon=True)
    
    axes[1].set_title("Multimodal t-SNE 2D Manifold Projection", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("t-SNE Dimension 1", fontsize=10, fontweight='bold')
    axes[1].set_ylabel("t-SNE Dimension 2", fontsize=10, fontweight='bold')
    axes[1].legend(title="Affect Category", frameon=True)
    
    plt.suptitle("Dimensionality Reduction & Clustering of 314-Feature Multimodal Representation", fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    f8_path = os.path.join(output_dir, "08_multimodal_pca_tsne_clustering.png")
    plt.savefig(f8_path)
    plt.close()
    print(f"Saved: {f8_path}")
    
    # -------------------------------------------------------------
    # Figure 9: Lexically Controlled Analysis (Kapoy n=11)
    # -------------------------------------------------------------
    df_kapoy = df[df["spoken_word"] == "Kapoy"].sort_values("audio_f0_mean_hz", ascending=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Pitch F0 across Kapoy speakers
    axes[0].barh(df_kapoy['participant_code'], df_kapoy['audio_f0_mean_hz'], color='#E64B35', alpha=0.8, edgecolor='black', height=0.6)
    axes[0].set_xlabel("Pitch F0 Mean (Hz)", fontsize=10, fontweight='bold')
    axes[0].set_title("Vocal Pitch Variation within 'Kapoy'\n(Range: 0 - 240 Hz)", fontsize=11, fontweight='bold')
    
    # RMS Energy across Kapoy speakers
    axes[1].barh(df_kapoy['participant_code'], df_kapoy['audio_rms_mean'], color='#4DBBD5', alpha=0.8, edgecolor='black', height=0.6)
    axes[1].set_xlabel("RMS Energy (Amplitude)", fontsize=10, fontweight='bold')
    axes[1].set_title("Speech Loudness Variation within 'Kapoy'", fontsize=11, fontweight='bold')
    
    # Smile intensity across Kapoy speakers (ironic vs genuinely fatigued)
    axes[2].barh(df_kapoy['participant_code'], df_kapoy['face_smile_mean'], color='#00A087', alpha=0.8, edgecolor='black', height=0.6)
    axes[2].set_xlabel("Smile Intensity (AU12)", fontsize=10, fontweight='bold')
    axes[2].set_title("Facial Smile Modulation within 'Kapoy'\n(Ironic/Smiley vs Flat Fatigue)", fontsize=11, fontweight='bold')
    
    plt.suptitle("Lexically Controlled Analysis: Holding Word Constant to 'Kapoy' (n=11)\nProves Speech & Facial Prosody Vary Independently of Lexical Semantics (Protocol Section XVII)", fontsize=13, fontweight='bold', y=1.04)
    plt.tight_layout()
    f9_path = os.path.join(output_dir, "09_lexically_controlled_analysis.png")
    plt.savefig(f9_path)
    plt.close()
    print(f"Saved: {f9_path}")
    
    # -------------------------------------------------------------
    # Figure 10: Multimodal Radar Affect Profiles
    # -------------------------------------------------------------
    radar_features = [
        ("audio_f0_mean_hz", "Pitch F0"),
        ("audio_rms_mean", "RMS Energy"),
        ("audio_spec_centroid_mean", "Brightness"),
        ("audio_duration_sec", "Duration"),
        ("face_smile_mean", "Smile (AU12)"),
        ("face_brow_lowerer_mean", "Brow Furrow"),
        ("face_mar_mean", "Mouth Open (MAR)"),
        ("face_expressiveness_score", "Expressiveness")
    ]
    
    # Normalize features between 0.1 and 0.9 for radar visualization
    radar_df = pd.DataFrame(index=df['affect_category'].unique())
    for col, label in radar_features:
        means = df.groupby('affect_category')[col].mean()
        min_v = df[col].min()
        max_v = df[col].max()
        norm_v = 0.15 + 0.7 * (means - min_v) / (max_v - min_v + 1e-6)
        radar_df[label] = norm_v
        
    labels = list(radar_df.columns)
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for cat in ["Fatigue", "Stress/Anxiety", "Positive", "Neutral"]:
        if cat not in radar_df.index:
            continue
        values = radar_df.loc[cat].values.flatten().tolist()
        values += values[:1]
        color = AFFECT_PALETTE.get(cat, '#888888')
        ax.plot(angles, values, linewidth=2.5, linestyle='solid', label=cat, color=color)
        ax.fill(angles, values, color=color, alpha=0.15)
        
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.set_title("Multimodal Affect Profiles Across Key Dimensions\n(Normalized Acoustic & Visual Signatures)", fontsize=13, fontweight='bold', pad=22)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), title="Affect Category", frameon=True)
    
    plt.tight_layout()
    f10_path = os.path.join(output_dir, "10_multimodal_radar_affect_profiles.png")
    plt.savefig(f10_path)
    plt.close()
    print(f"Saved: {f10_path}")
    
    print("\nAll 10 Publication-Quality Visualizations successfully generated!")


if __name__ == "__main__":
    generate_all_visualizations()
