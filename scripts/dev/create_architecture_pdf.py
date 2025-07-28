#!/usr/bin/env python3
"""
Create a simple PDF version of the architecture paper with patent information.
"""

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import red, orange
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

def create_architecture_pdf():
    """Create a PDF version of the architecture paper."""
    
    # Create the PDF document
    doc = SimpleDocTemplate(
        "preprints/multimodal_transformer_architecture_corrected.pdf",
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    patent_style = ParagraphStyle(
        'PatentWarning',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=20,
        alignment=TA_CENTER,
        textColor=red,
        borderColor=orange,
        borderWidth=2,
        borderPadding=10
    )
    
    heading_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Build the story (content)
    story = []
    
    # Title
    story.append(Paragraph("Multi-Modal Transformer Architecture for Genomic Data Integration: A Novel Approach to Cancer Classification", title_style))
    story.append(Spacer(1, 12))
    
    # Patent Warning
    patent_text = """
    <b>⚠️ PATENT PROTECTED TECHNOLOGY ⚠️</b><br/><br/>
    <b>Patent:</b> Provisional Application No. 63/847,316<br/>
    <b>Title:</b> Systems and Methods for Cancer Classification Using Multi-Modal Transformer-Based Architectures<br/>
    <b>Patent Holder:</b> Dr. R. Craig Stillwell<br/>
    <b>Commercial Use:</b> Requires separate patent license<br/><br/>
    <i>For commercial licensing inquiries, contact: craig.stillwell@gmail.com</i>
    """
    story.append(Paragraph(patent_text, patent_style))
    story.append(Spacer(1, 24))
    
    # Abstract
    story.append(Paragraph("Abstract", heading_style))
    abstract_text = """
    Cancer genomics research increasingly relies on multi-modal data integration to capture the complex molecular landscape of tumors. Here, we present a novel multi-modal transformer architecture specifically designed for integrating heterogeneous genomic data types in cancer classification tasks. Our approach addresses key computational challenges in applying attention mechanisms to genomic data through modality-specific encoders, cross-modal attention layers, and synthetic data generation strategies. The architecture demonstrates effective fusion of methylation patterns, fragmentomics profiles, and copy number alteration data through learned attention weights. We validate our approach using synthetic genomic datasets that preserve realistic data characteristics while enabling controlled experimentation. This work contributes to the growing field of AI-driven cancer genomics by providing a scalable framework for multi-modal genomic data analysis that can be adapted across different cancer types and genomic platforms.
    """
    story.append(Paragraph(abstract_text, normal_style))
    story.append(Spacer(1, 12))
    
    # Keywords
    story.append(Paragraph("<b>Keywords:</b> transformer networks, multi-modal learning, cancer genomics, attention mechanisms, methylation analysis, fragmentomics", normal_style))
    story.append(Spacer(1, 24))
    
    # Introduction
    story.append(Paragraph("1. Introduction", heading_style))
    intro_text = """
    The integration of multiple genomic data modalities represents one of the most promising frontiers in computational cancer biology. Traditional machine learning approaches in cancer genomics have largely focused on single-modality analyses, limiting their ability to capture the complex interdependencies between different molecular layers. Recent advances in transformer architectures, originally developed for natural language processing, have shown remarkable success in biological sequence analysis, yet their application to multi-modal genomic data integration remains underexplored.
    """
    story.append(Paragraph(intro_text, normal_style))
    story.append(Spacer(1, 12))
    
    # Methods
    story.append(Paragraph("2. Methods", heading_style))
    methods_text = """
    Our multi-modal transformer architecture consists of three main components: modality-specific encoders, cross-modal attention layers, and classification heads. The overall architecture is implemented using PyTorch Lightning to ensure reproducible training and efficient distributed computing.
    """
    story.append(Paragraph(methods_text, normal_style))
    story.append(Spacer(1, 12))
    
    # Results
    story.append(Paragraph("3. Results", heading_style))
    results_text = """
    Our multi-modal transformer architecture successfully integrates three genomic modalities with effective attention-based fusion. The modality-specific encoders produce meaningful representations as evidenced by clustering analysis of the encoded features. Cross-modal attention weights reveal biologically meaningful patterns, with the model learning to focus on relevant genomic regions for classification.
    """
    story.append(Paragraph(results_text, normal_style))
    story.append(Spacer(1, 12))
    
    # Conclusion
    story.append(Paragraph("4. Conclusion", heading_style))
    conclusion_text = """
    We present a novel multi-modal transformer architecture specifically designed for cancer genomics applications. The architecture effectively integrates methylation, fragmentomics, and copy number alteration data through modality-specific encoders and cross-modal attention mechanisms. Our synthetic data generation framework enables controlled validation and reproducible research in multi-modal genomic analysis. The computational efficiency and modular design of our approach make it suitable for large-scale genomic applications and extensible to additional data modalities.
    """
    story.append(Paragraph(conclusion_text, normal_style))
    story.append(Spacer(1, 24))
    
    # Patent Notice at the end
    story.append(Paragraph("Patent Protection Notice", heading_style))
    patent_notice = """
    This work describes technology protected by provisional patent application No. 63/847,316. Commercial use of the described methods and systems requires separate patent licensing. For licensing inquiries, please contact Dr. R. Craig Stillwell at craig.stillwell@gmail.com.
    """
    story.append(Paragraph(patent_notice, normal_style))
    
    # Build the PDF
    doc.build(story)
    print("✅ Successfully created architecture paper PDF with patent information!")
    return True

if __name__ == "__main__":
    try:
        create_architecture_pdf()
    except ImportError:
        print("❌ ReportLab not found. Installing...")
        import subprocess
        subprocess.run(["pip", "install", "reportlab"])
        create_architecture_pdf()
    except Exception as e:
        print(f"❌ Error creating PDF: {e}")
        print("Please install reportlab: pip install reportlab")
