"""
Figure 1: Cancer Alpha System Architecture
Generate publication-quality system architecture diagram
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def create_system_architecture_figure():
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Define colors
    colors = {
        'frontend': '#E3F2FD',
        'backend': '#F3E5F5',
        'data': '#E8F5E8',
        'infrastructure': '#FFF3E0',
        'monitoring': '#FCE4EC'
    }
    
    # Define component positions and sizes
    components = {
        'load_balancer': {'pos': (1, 8), 'size': (2, 1), 'color': colors['infrastructure']},
        'ingress': {'pos': (4, 8), 'size': (2, 1), 'color': colors['infrastructure']},
        'web_frontend': {'pos': (1, 6), 'size': (2.5, 1.5), 'color': colors['frontend']},
        'api_backend': {'pos': (4.5, 6), 'size': (2.5, 1.5), 'color': colors['backend']},
        'redis': {'pos': (1, 4), 'size': (2, 1), 'color': colors['data']},
        'postgresql': {'pos': (4, 4), 'size': (2, 1), 'color': colors['data']},
        'ml_models': {'pos': (7, 4), 'size': (2, 1), 'color': colors['backend']},
        'prometheus': {'pos': (1, 2), 'size': (2, 1), 'color': colors['monitoring']},
        'grafana': {'pos': (4, 2), 'size': (2, 1), 'color': colors['monitoring']},
        'kubernetes': {'pos': (7, 2), 'size': (2, 1), 'color': colors['infrastructure']}
    }
    
    # Draw components
    for name, config in components.items():
        x, y = config['pos']
        w, h = config['size']
        
        # Create rounded rectangle
        rect = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.1",
            facecolor=config['color'],
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(rect)
        
        # Add component labels
        labels = {
            'load_balancer': 'Load Balancer\n(nginx)',
            'ingress': 'Ingress\nController',
            'web_frontend': 'Web Frontend\n(React + TypeScript)\nPort: 3000',
            'api_backend': 'API Backend\n(FastAPI)\nPort: 8000',
            'redis': 'Redis\n(Caching)\nPort: 6379',
            'postgresql': 'PostgreSQL\n(Database)\nPort: 5432',
            'ml_models': 'ML Models\n(Ensemble)\nPersistent',
            'prometheus': 'Prometheus\n(Monitoring)\nPort: 9090',
            'grafana': 'Grafana\n(Dashboards)\nPort: 3001',
            'kubernetes': 'Kubernetes\n(Orchestration)\nAuto-scaling'
        }
        
        ax.text(x + w/2, y + h/2, labels[name], 
                ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Draw connections
    connections = [
        ('load_balancer', 'web_frontend'),
        ('ingress', 'api_backend'),
        ('web_frontend', 'api_backend'),
        ('api_backend', 'redis'),
        ('api_backend', 'postgresql'),
        ('api_backend', 'ml_models'),
        ('prometheus', 'grafana'),
        ('prometheus', 'api_backend'),
        ('kubernetes', 'ml_models')
    ]
    
    for start, end in connections:
        start_pos = components[start]['pos']
        start_size = components[start]['size']
        end_pos = components[end]['pos']
        end_size = components[end]['size']
        
        # Calculate connection points
        start_x = start_pos[0] + start_size[0]/2
        start_y = start_pos[1] + start_size[1]/2
        end_x = end_pos[0] + end_size[0]/2
        end_y = end_pos[1] + end_size[1]/2
        
        # Draw arrow
        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='#666666'))
    
    # Add title and labels
    ax.set_title('Cancer Alpha System Architecture', fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    legend_elements = [
        patches.Patch(color=colors['frontend'], label='Frontend Components'),
        patches.Patch(color=colors['backend'], label='Backend Components'),
        patches.Patch(color=colors['data'], label='Data Storage'),
        patches.Patch(color=colors['infrastructure'], label='Infrastructure'),
        patches.Patch(color=colors['monitoring'], label='Monitoring')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(10, 9))
    
    # Set axis properties
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add deployment flow annotation
    ax.text(6, 0.5, 'Production-Ready Deployment with Docker + Kubernetes', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('src/phase5_manuscript_and_submission/figures/figure1_system_architecture.png', 
                dpi=300, bbox_inches='tight')
    plt.savefig('src/phase5_manuscript_and_submission/figures/figure1_system_architecture.pdf', 
                bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    create_system_architecture_figure()
    print("Figure 1: System Architecture created successfully!")
