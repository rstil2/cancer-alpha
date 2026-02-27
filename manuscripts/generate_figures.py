#!/usr/bin/env python3
"""Generate publication-quality figures for the Oncura AIM manuscript."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

# Output directory
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Publication style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

ONCURA_BLUE = '#1565C0'
ONCURA_GOLD = '#F9A825'
ONCURA_TEAL = '#00897B'
ONCURA_RED = '#C62828'
GRAY = '#757575'


def figure1_model_performance():
    """Figure 1: Model Performance Comparison (Table 8)."""
    models = ['LightGBM\n(Champion)', 'XGBoost', 'Random\nForest',
              'Logistic\nRegression', 'Gradient\nBoosting', 'SVM']
    accuracy = [96.5, 96.2, 94.9, 94.8, 92.7, 89.0]
    std_dev = [0.6, 1.0, 1.2, 2.7, 0.8, 1.9]
    colors = [ONCURA_GOLD, ONCURA_BLUE, ONCURA_BLUE,
              ONCURA_TEAL, ONCURA_TEAL, GRAY]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, accuracy, yerr=std_dev, capsize=5,
                  color=colors, edgecolor='white', linewidth=0.8, width=0.65)

    ax.set_ylabel('Balanced Accuracy (%)')
    ax.set_title('Model Performance Comparison on Real TCGA Data (n = 1,200)')
    ax.set_ylim(85, 100)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.axhline(y=89.2, color=ONCURA_RED, linestyle='--', alpha=0.6, linewidth=1)
    ax.text(5.4, 89.5, 'State-of-the-art\n(Yuan et al.)', fontsize=8,
            color=ONCURA_RED, ha='right', va='bottom')

    for bar, acc, sd in zip(bars, accuracy, std_dev):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + sd + 0.3,
                f'{acc}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'Figure_1_Model_Performance.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved: {path}')


def figure2_cancer_type_performance():
    """Figure 2: Cancer Type-Specific Performance (Table 9)."""
    cancers = ['BRCA', 'LUAD', 'LUSC', 'HNSC', 'COAD', 'PRAD', 'LIHC', 'STAD']
    accuracy = [97.8, 96.5, 96.1, 95.7, 95.2, 94.8, 93.4, 91.2]
    precision = [96.2, 95.8, 95.4, 94.9, 94.1, 93.7, 92.8, 90.5]
    recall = [100.0, 97.5, 96.8, 96.5, 96.2, 95.8, 94.1, 92.1]

    x = np.arange(len(cancers))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, accuracy, width, label='Balanced Accuracy',
                   color=ONCURA_BLUE, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x, precision, width, label='Precision',
                   color=ONCURA_TEAL, edgecolor='white', linewidth=0.5)
    bars3 = ax.bar(x + width, recall, width, label='Recall',
                   color=ONCURA_GOLD, edgecolor='white', linewidth=0.5)

    ax.set_ylabel('Performance (%)')
    ax.set_title('Cancer Type-Specific Performance (LightGBM Champion Model)')
    ax.set_xticks(x)
    ax.set_xticklabels(cancers, fontweight='bold')
    ax.set_ylim(87, 102)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.legend(loc='lower left', frameon=True, framealpha=0.9)
    ax.axhline(y=91, color=GRAY, linestyle=':', alpha=0.4, linewidth=0.8)
    ax.text(7.4, 91.2, '91% clinical\nrelevance threshold', fontsize=8,
            color=GRAY, ha='right', va='bottom')

    for bar, acc in zip(bars1, accuracy):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{acc}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'Figure_2_Cancer_Performance.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved: {path}')


def figure3_benchmarking():
    """Figure 3: Benchmarking Against State-of-the-Art (Table 10)."""
    studies = [
        'Oncura\n(This Study)',
        'Yuan et al.\n(2023)',
        'Zhang et al.\n(2021)',
        'Cheerla &\nGevaert (2019)',
        'Li et al.\n(2020)',
        'Poirion et al.\n(2021)',
    ]
    accuracy = [96.5, 89.2, 88.3, 86.1, 84.7, 83.9]
    methods = ['LightGBM\nFramework', 'Transformer', 'DNN',
               'DeepSurv+CNN', 'Random Forest', 'Pan-Cancer\nBERT']
    samples = [1200, 4127, 3586, 5314, 2448, 7742]
    colors = [ONCURA_GOLD] + [ONCURA_BLUE]*5

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(studies[::-1], accuracy[::-1], color=colors[::-1],
                   edgecolor='white', linewidth=0.8, height=0.55)

    ax.set_xlabel('Balanced Accuracy (%)')
    ax.set_title('Oncura vs. State-of-the-Art Cancer Classification Systems')
    ax.set_xlim(80, 100)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))

    for bar, acc, method, n in zip(bars, accuracy[::-1], methods[::-1], samples[::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{acc}%  ({method}, n={n:,})',
                ha='left', va='center', fontsize=9)

    # Highlight the gap
    ax.axvline(x=89.2, color=ONCURA_RED, linestyle='--', alpha=0.4, linewidth=1)
    ax.annotate('', xy=(96.5, 5.15), xytext=(89.2, 5.15),
                arrowprops=dict(arrowstyle='<->', color=ONCURA_RED, lw=1.5))
    ax.text(92.85, 5.45, '+7.3pp improvement', ha='center', va='bottom',
            fontsize=9, fontweight='bold', color=ONCURA_RED)

    ax.set_xlim(80, 105)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'Figure_3_Benchmarking.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved: {path}')


def figure4_feature_importance():
    """Figure 4: Feature Importance and SHAP Analysis."""
    features = [
        'Age at Diagnosis',
        'Gene Expr. Cluster 1\n(Tissue-specific)',
        'Gene Expr. Cluster 7\n(Oncogene patterns)',
        'Gene Expr. Cluster 12\n(Tumor suppressor)',
        'Gene Expr. Cluster 3\n(Metabolic pathways)',
        'Mutation Load\n(TMB)',
        'TP53 Mutation Status',
        'Methylation Pattern 1\n(CpG islands)',
        'Copy Number Cluster 2\n(Chromosomal instability)',
        'Clinical Stage',
    ]
    shap_values = [0.124, 0.089, 0.067, 0.054, 0.048, 0.042, 0.039, 0.036, 0.033, 0.031]

    # Color by modality
    modality_colors = {
        'Clinical': '#E65100',
        'Expression': ONCURA_BLUE,
        'Mutation': ONCURA_RED,
        'Methylation': ONCURA_TEAL,
        'CNA': '#7B1FA2',
    }
    colors = [
        modality_colors['Clinical'],    # Age
        modality_colors['Expression'],  # GE Cluster 1
        modality_colors['Expression'],  # GE Cluster 7
        modality_colors['Expression'],  # GE Cluster 12
        modality_colors['Expression'],  # GE Cluster 3
        modality_colors['Mutation'],    # Mutation Load
        modality_colors['Mutation'],    # TP53
        modality_colors['Methylation'], # Methylation
        modality_colors['CNA'],         # CNA
        modality_colors['Clinical'],    # Stage
    ]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(features[::-1], shap_values[::-1], color=colors[::-1],
                   edgecolor='white', linewidth=0.8, height=0.6)

    ax.set_xlabel('Mean |SHAP Value| (Feature Importance)')
    ax.set_title('Top 10 Most Important Features — Biologically Validated')

    for bar, val in zip(bars, shap_values[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', ha='left', va='center', fontsize=9)

    # Legend for modalities
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=modality_colors['Clinical'], label='Clinical'),
        Patch(facecolor=modality_colors['Expression'], label='Gene Expression (ICGC ARGO)'),
        Patch(facecolor=modality_colors['Mutation'], label='Mutations'),
        Patch(facecolor=modality_colors['Methylation'], label='Methylation'),
        Patch(facecolor=modality_colors['CNA'], label='Copy Number Alterations'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True,
              framealpha=0.9, title='Data Modality', title_fontsize=10)

    ax.set_xlim(0, 0.15)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'Figure_4_Feature_Importance.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved: {path}')


if __name__ == '__main__':
    figure1_model_performance()
    figure2_cancer_type_performance()
    figure3_benchmarking()
    figure4_feature_importance()
    print('\nAll figures generated successfully.')
