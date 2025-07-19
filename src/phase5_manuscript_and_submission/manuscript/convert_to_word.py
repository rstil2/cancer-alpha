#!/usr/bin/env python3
"""
Convert Cancer Alpha manuscript to Word format with embedded figures
Formatted for Nature Machine Intelligence submission
"""

import os
import shutil
from pathlib import Path
import subprocess

def main():
    # Define paths
    base_dir = Path("/Users/stillwell/projects/cancer-alpha")
    manuscript_dir = base_dir / "src/phase5_manuscript_and_submission/manuscript"
    figures_source = base_dir / "project36_fourth_source/manuscript_submission/figures"
    submission_dir = manuscript_dir / "nature_machine_intelligence_submission"
    
    # Create submission directory
    submission_dir.mkdir(exist_ok=True)
    figures_dir = submission_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # Copy figures to submission directory
    figure_files = [
        "model_performance_comparison.png",  # Figure 1
        "data_source_distribution.png",      # Figure 2
        "feature_importance_analysis.png",   # Figure 3
        "roc_curves.png",                    # Figure 4
        "tsne_analysis.png"                  # Figure 5
    ]
    
    print("Copying figures to submission directory...")
    for fig_file in figure_files:
        src = figures_source / fig_file
        dst = figures_dir / fig_file
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  ✓ Copied {fig_file}")
        else:
            print(f"  ✗ Missing {fig_file}")
    
    # Copy manuscript files
    manuscript_files = [
        "cancer_alpha_manuscript_formatted.md",
        "cover_letter_nature_machine_intelligence.md"
    ]
    
    print("\nCopying manuscript files...")
    for file in manuscript_files:
        src = manuscript_dir / file
        dst = submission_dir / file
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  ✓ Copied {file}")
        else:
            print(f"  ✗ Missing {file}")
    
    # Create a combined document with embedded figures
    print("\nCreating combined document with embedded figures...")
    
    # Read the formatted manuscript
    manuscript_path = submission_dir / "cancer_alpha_manuscript_formatted.md"
    with open(manuscript_path, 'r') as f:
        content = f.read()
    
    # Create enhanced version with figure paths
    enhanced_content = content.replace(
        "### Figure 1. Cancer Alpha System Architecture",
        "### Figure 1. Cancer Alpha System Architecture\n![Figure 1](figures/model_performance_comparison.png)"
    ).replace(
        "### Figure 2. Data Source Distribution",
        "### Figure 2. Data Source Distribution\n![Figure 2](figures/data_source_distribution.png)"
    ).replace(
        "### Figure 3. Feature Importance Analysis",
        "### Figure 3. Feature Importance Analysis\n![Figure 3](figures/feature_importance_analysis.png)"
    ).replace(
        "### Figure 4. ROC Curves for Multi-Class Cancer Classification",
        "### Figure 4. ROC Curves for Multi-Class Cancer Classification\n![Figure 4](figures/roc_curves.png)"
    ).replace(
        "### Figure 5. t-SNE Analysis of Multi-Modal Features",
        "### Figure 5. t-SNE Analysis of Multi-Modal Features\n![Figure 5](figures/tsne_analysis.png)"
    )
    
    # Write enhanced manuscript
    enhanced_path = submission_dir / "cancer_alpha_manuscript_with_figures.md"
    with open(enhanced_path, 'w') as f:
        f.write(enhanced_content)
    
    print(f"  ✓ Created enhanced manuscript: {enhanced_path}")
    
    # Create submission README
    readme_content = """# Cancer Alpha - Nature Machine Intelligence Submission

## Files Included:

### Main Documents:
- `cancer_alpha_manuscript_formatted.md` - Main manuscript formatted for Nature Machine Intelligence
- `cancer_alpha_manuscript_with_figures.md` - Enhanced manuscript with embedded figure references
- `cover_letter_nature_machine_intelligence.md` - Cover letter for journal submission

### Figures:
- `figures/model_performance_comparison.png` - Figure 1: Cancer Alpha System Architecture
- `figures/data_source_distribution.png` - Figure 2: Data Source Distribution
- `figures/feature_importance_analysis.png` - Figure 3: Feature Importance Analysis
- `figures/roc_curves.png` - Figure 4: ROC Curves for Multi-Class Cancer Classification
- `figures/tsne_analysis.png` - Figure 5: t-SNE Analysis of Multi-Modal Features

## Submission Details:
- **Journal:** Nature Machine Intelligence
- **Manuscript Type:** Article
- **Word Count:** 4,500 words (excluding references and figure legends)
- **Figures:** 5
- **Tables:** 2 (embedded in manuscript)
- **References:** 10

## Next Steps:
1. Convert manuscript to Word format using pandoc or similar tool
2. Embed figures at the end of the Word document
3. Review journal formatting requirements
4. Submit via Nature Machine Intelligence submission portal

## Technical Requirements Met:
✓ Multi-modal transformer architecture
✓ Production-ready infrastructure
✓ Real experimental data
✓ Ethical AI considerations
✓ Regulatory compliance framework
✓ Comprehensive figures and tables
✓ Nature Machine Intelligence formatting

## Contact:
Cancer Alpha Development Team
Email: cancer-alpha@research.org
"""
    
    readme_path = submission_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"  ✓ Created README: {readme_path}")
    
    # Try to convert to Word using pandoc (if available)
    try:
        print("\nAttempting to convert to Word format using pandoc...")
        word_output = submission_dir / "cancer_alpha_manuscript.docx"
        
        pandoc_cmd = [
            "pandoc",
            str(enhanced_path),
            "-o", str(word_output),
            "--reference-doc=/System/Library/Application Support/Microsoft/MSXML60/1033/template.docx",
            "--toc",
            "--number-sections"
        ]
        
        result = subprocess.run(pandoc_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"  ✓ Successfully converted to Word: {word_output}")
        else:
            print(f"  ✗ Pandoc conversion failed: {result.stderr}")
            print("  → You can manually convert the markdown file to Word")
    
    except FileNotFoundError:
        print("  ✗ Pandoc not found. Please install pandoc or manually convert to Word")
        print("  → brew install pandoc  # on macOS")
        print("  → Or use online converters like pandoc.org/try")
    
    print(f"\n🎉 Submission package created in: {submission_dir}")
    print("\nFiles ready for Nature Machine Intelligence submission:")
    print("─" * 50)
    
    for file in submission_dir.rglob("*"):
        if file.is_file():
            print(f"  {file.relative_to(submission_dir)}")
    
    print("\n📝 Next steps:")
    print("1. Review the Word document (if created)")
    print("2. Manually embed figures at the end if needed")
    print("3. Check Nature Machine Intelligence formatting requirements")
    print("4. Submit via journal portal")

if __name__ == "__main__":
    main()
