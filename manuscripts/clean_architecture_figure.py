#!/usr/bin/env python3
"""
Clean System Architecture Figure for Cancer Alpha
Creates a professional, clean workflow diagram from scratch
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Set clean, professional styling
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'axes.linewidth': 0,
    'xtick.bottom': False,
    'xtick.top': False,
    'ytick.left': False,
    'ytick.right': False
})

def create_clean_architecture_figure():
    """Create a clean, professional system architecture figure"""
    
    # Create figure with clean background
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Define main workflow stages with positions
    stages = [
        {'name': 'Data Input', 'x': 2, 'y': 6, 'width': 2.2, 'height': 1.2, 'color': '#E8F4FD'},
        {'name': 'Preprocessing', 'x': 5.5, 'y': 6, 'width': 2.2, 'height': 1.2, 'color': '#FFF2CC'},
        {'name': 'Feature Selection', 'x': 9, 'y': 6, 'width': 2.2, 'height': 1.2, 'color': '#F8CECC'},
        {'name': 'Model Training', 'x': 12.5, 'y': 6, 'width': 2.2, 'height': 1.2, 'color': '#D5E8D4'},
        {'name': 'Production System', 'x': 5.5, 'y': 3, 'width': 2.2, 'height': 1.2, 'color': '#E1D5E7'},
        {'name': 'Output', 'x': 12.5, 'y': 3, 'width': 2.2, 'height': 1.2, 'color': '#FFF2CC'}
    ]
    
    # Draw main workflow boxes
    for stage in stages:
        # Create rounded rectangle
        box = FancyBboxPatch(
            (stage['x'] - stage['width']/2, stage['y'] - stage['height']/2),
            stage['width'], stage['height'],
            boxstyle="round,pad=0.1",
            facecolor=stage['color'],
            edgecolor='#333333',
            linewidth=2,
            alpha=0.9
        )
        ax.add_patch(box)
        
        # Add stage title
        ax.text(stage['x'], stage['y'], stage['name'], 
                ha='center', va='center', fontsize=13, fontweight='bold',
                color='#333333')
    
    # Add detailed components under main stages
    details = [
        # Data Input details
        {'text': '• TCGA Data (158 samples)\n• Clinical Variables\n• Genomic Mutations', 
         'x': 2, 'y': 4.8, 'align': 'center'},
        
        # Preprocessing details
        {'text': '• Quality Control\n• Missing Value Imputation\n• Data Standardization', 
         'x': 5.5, 'y': 4.8, 'align': 'center'},
        
        # Feature Selection details
        {'text': '• Mutual Information Ranking\n• 150 Features Selected\n• Biological Validation', 
         'x': 9, 'y': 4.8, 'align': 'center'},
        
        # Model Training details
        {'text': '• SMOTE Class Balancing\n• LightGBM Algorithm\n• 10-Fold Cross-Validation', 
         'x': 12.5, 'y': 4.8, 'align': 'center'},
        
        # Production System details
        {'text': '• FastAPI Endpoints\n• Docker Containers\n• HIPAA Compliance', 
         'x': 5.5, 'y': 1.8, 'align': 'center'},
        
        # Output details
        {'text': '• 95% Balanced Accuracy\n• <50ms Response Time\n• Clinical Predictions', 
         'x': 12.5, 'y': 1.8, 'align': 'center'}
    ]
    
    # Add detail text
    for detail in details:
        ax.text(detail['x'], detail['y'], detail['text'],
                ha=detail['align'], va='center', fontsize=9,
                color='#555555', linespacing=1.5)
    
    # Draw clean workflow arrows
    arrow_style = dict(
        arrowstyle='->', 
        lw=3, 
        color='#2E75B6',
        alpha=0.8
    )
    
    # Main horizontal flow arrows
    arrows = [
        # Data Input → Preprocessing
        ((3.1, 6), (4.4, 6)),
        # Preprocessing → Feature Selection  
        ((6.6, 6), (7.9, 6)),
        # Feature Selection → Model Training
        ((10.1, 6), (11.4, 6)),
        # Model Training → Production (curved down)
        ((12.5, 5.4), (6.7, 3.6)),
        # Production → Output (horizontal)
        ((6.6, 3), (11.4, 3))
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start, arrowprops=arrow_style)
    
    # Add performance metrics box
    metrics_text = (
        "PERFORMANCE METRICS\n\n"
        "• Balanced Accuracy: 95.0% ± 5.4%\n"
        "• Clinical Threshold: >90% Achieved\n"
        "• Processing Time: <50ms\n"
        "• System Uptime: 99.97%\n"
        "• Real TCGA Data: 158 patients\n"
        "• Production Ready: Full deployment"
    )
    
    # Metrics box
    metrics_box = FancyBboxPatch(
        (0.2, 0.2), 3.5, 2.5,
        boxstyle="round,pad=0.15",
        facecolor='#F5F5F5',
        edgecolor='#888888',
        linewidth=1.5,
        alpha=0.95
    )
    ax.add_patch(metrics_box)
    
    ax.text(2, 1.45, metrics_text, ha='center', va='center', 
            fontsize=9, color='#333333', linespacing=1.4)
    
    # Add main title
    ax.text(8, 7.5, 'Cancer Alpha: Production-Ready AI System Architecture', 
            ha='center', va='center', fontsize=18, fontweight='bold',
            color='#1f4e79')
    
    # Add subtitle
    ax.text(8, 7.1, 'End-to-End Workflow for Multi-Cancer Classification', 
            ha='center', va='center', fontsize=12, style='italic',
            color='#555555')
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('manuscript_figures/figure6_system_architecture.png', 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig('manuscript_figures/figure6_system_architecture.pdf', 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    plt.show()
    print("✅ Clean system architecture figure created successfully!")

if __name__ == "__main__":
    create_clean_architecture_figure()
