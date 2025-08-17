#!/usr/bin/env python3
"""
Extract Figure 1 from the architecture paper PDF for use in README.
"""

import fitz  # PyMuPDF
import os
from PIL import Image
import io

def extract_figure1():
    """Extract Figure 1 from the architecture paper PDF."""
    
    pdf_path = "preprints/multimodal_transformer_architecture_corrected.pdf"
    output_path = "docs/figures/architecture_figure1.png"
    
    try:
        # Open the PDF
        doc = fitz.open(pdf_path)
        print(f"📄 Opened PDF with {len(doc)} pages")
        
        # Look through the first few pages for Figure 1
        figure_found = False
        
        for page_num in range(min(5, len(doc))):  # Check first 5 pages
            page = doc.load_page(page_num)
            
            # Get all images on this page
            image_list = page.get_images()
            
            if image_list:
                print(f"📊 Found {len(image_list)} images on page {page_num + 1}")
                
                # Extract each image
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    # Convert to PNG if not already
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        
                        # Save the first substantial image as Figure 1
                        if len(img_data) > 10000:  # Filter out small images
                            with open(output_path, "wb") as f:
                                f.write(img_data)
                            
                            print(f"✅ Extracted Figure 1 to {output_path}")
                            print(f"📏 Image size: {len(img_data)} bytes")
                            
                            # Try to get image dimensions
                            try:
                                img_pil = Image.open(io.BytesIO(img_data))
                                print(f"📐 Dimensions: {img_pil.size[0]}x{img_pil.size[1]} pixels")
                            except:
                                pass
                            
                            figure_found = True
                            pix = None
                            break
                    
                    pix = None
                
                if figure_found:
                    break
        
        doc.close()
        
        if not figure_found:
            print("❌ Could not find Figure 1 in the PDF")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error extracting figure: {e}")
        return False

def add_figure_to_readme():
    """Add Figure 1 to the README.md file."""
    
    readme_path = "README.md"
    figure_path = "docs/figures/architecture_figure1.png"
    
    # Check if figure exists
    if not os.path.exists(figure_path):
        print("❌ Figure not found, cannot add to README")
        return False
    
    # Read current README
    with open(readme_path, 'r') as f:
        content = f.read()
    
    # Find a good place to insert the figure (after the main description)
    # Look for the "What Makes Oncura Special?" section
    insert_marker = "## 🌟 What Makes Oncura Special?"
    
    if insert_marker in content:
        # Insert the figure before this section
        figure_section = f"""
## 🏗️ **Multi-Modal Transformer Architecture**

<div align="center">
<img src="docs/figures/architecture_figure1.png" alt="Multi-Modal Transformer Architecture" width="800"/>

*Figure 1: Novel multi-modal transformer architecture for cancer genomics data integration, showing modality-specific encoders, cross-modal attention mechanisms, and classification layers.*
</div>

---

"""
        
        # Replace the section
        updated_content = content.replace(insert_marker, figure_section + insert_marker)
        
        # Write back to README
        with open(readme_path, 'w') as f:
            f.write(updated_content)
        
        print("✅ Added Figure 1 to README.md")
        return True
    else:
        print("❌ Could not find insertion point in README")
        return False

if __name__ == "__main__":
    try:
        # Extract the figure
        if extract_figure1():
            # Add to README
            add_figure_to_readme()
        
    except ImportError:
        print("❌ PyMuPDF not found. Installing...")
        import subprocess
        subprocess.run(["pip", "install", "PyMuPDF", "Pillow"])
        
        # Try again
        if extract_figure1():
            add_figure_to_readme()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please ensure PyMuPDF is installed: pip install PyMuPDF")
