#!/usr/bin/env python3
"""
Strategic Expansion to 50K+ Samples
===================================

Based on current analysis showing 43,631 samples, we need 6,369 more.
Focus on cancer types with highest expansion potential.

STRICT RULE: Only real TCGA data - zero synthetic data allowed!
"""

import json
import pandas as pd
from pathlib import Path

def analyze_expansion_opportunities():
    """Identify cancer types with highest expansion potential"""
    
    # Load current coverage analysis
    analysis_file = "tcga_coverage_analysis_50k_20250822_095144.json"
    with open(analysis_file, 'r') as f:
        coverage = json.load(f)
    
    print("=" * 80)
    print("STRATEGIC EXPANSION TO 50K+ SAMPLES")
    print("=" * 80)
    print(f"Current samples: {coverage['total_samples']:,}")
    print(f"Target samples: 50,000")
    print(f"Additional needed: {50000 - coverage['total_samples']:,}")
    print()
    
    # Identify high-yield expansion opportunities
    cancer_priorities = []
    
    for cancer_type, data in coverage['cancer_type_coverage'].items():
        if cancer_type == 'unknown':
            continue
            
        total_files = data['total_files']
        
        # Calculate expansion potential based on current coverage
        # Focus on cancer types that could easily add 1000+ samples
        expansion_potential = 0
        
        # High-volume cancer types with room for growth
        if total_files > 3000:  # Already large, could expand further
            expansion_potential = 2000
        elif total_files > 2000:  # Medium size, good expansion potential  
            expansion_potential = 1500
        elif total_files > 1000:  # Smaller, but could still expand
            expansion_potential = 1000
        else:  # Very small, limited expansion
            expansion_potential = 500
            
        # Bonus for cancer types with uneven data type coverage (room to balance)
        data_type_counts = data['data_type_counts']
        max_count = max(data_type_counts.values())
        min_count = min(data_type_counts.values())
        
        if max_count - min_count > 200:  # Uneven coverage = expansion opportunity
            expansion_potential += 500
            
        cancer_priorities.append({
            'cancer_type': cancer_type,
            'current_files': total_files,
            'expansion_potential': expansion_potential,
            'data_imbalance': max_count - min_count,
            'priority_score': expansion_potential + (total_files * 0.1)
        })
    
    # Sort by priority score
    cancer_priorities.sort(key=lambda x: x['priority_score'], reverse=True)
    
    print("TOP EXPANSION TARGETS:")
    print("-" * 80)
    cumulative_samples = 0
    selected_targets = []
    
    for i, cancer in enumerate(cancer_priorities[:10]):
        cumulative_samples += cancer['expansion_potential']
        selected_targets.append(cancer)
        
        print(f"{i+1:2d}. {cancer['cancer_type']:15s} | "
              f"Current: {cancer['current_files']:4d} files | "
              f"Potential: +{cancer['expansion_potential']:4d} samples | "
              f"Cumulative: {cumulative_samples:5d}")
        
        # Stop when we have enough targets to reach 50K+
        if cumulative_samples >= 7000:  # Buffer above needed 6,369
            break
    
    print()
    print(f"Selected targets will add ~{cumulative_samples:,} samples")
    print(f"This will bring total to ~{coverage['total_samples'] + cumulative_samples:,} samples")
    print()
    
    # Generate specific download commands for top targets
    print("RECOMMENDED EXPANSION ACTIONS:")
    print("-" * 80)
    
    top_5_targets = selected_targets[:5]  # Focus on top 5 for efficiency
    
    for cancer in top_5_targets:
        cancer_type = cancer['cancer_type']
        print(f"\n{cancer_type}:")
        print(f"  Current files: {cancer['current_files']:,}")
        print(f"  Expansion target: +{cancer['expansion_potential']:,} samples")
        print(f"  Action: Run massive downloader with focus on {cancer_type}")
        
    return selected_targets[:5]

def generate_expansion_commands(targets):
    """Generate specific commands for expansion"""
    
    print("\nEXPANSION COMMANDS:")
    print("=" * 80)
    
    # Create focused download commands
    for cancer in targets:
        cancer_type = cancer['cancer_type']
        print(f"# Expand {cancer_type}")
        print(f"python scalable_tcga_downloader.py --cancer-types {cancer_type} --max-samples 2000 --all-data-types")
        print()
    
    print("# Process expanded dataset")
    print("python ultra_massive_multi_omics_processor.py")
    print()
    print("# Validate final count")
    print("python quick_coverage_check.py")

if __name__ == "__main__":
    targets = analyze_expansion_opportunities()
    generate_expansion_commands(targets)
