#!/usr/bin/env python3
"""
Extract Figure 1 from page 3 of the Word document.
"""

from docx import Document
from docx.document import Document as _Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph
import os

def extract_images_from_docx():
    """Extract all images from the Word document."""
    
    docx_path = "preprints/multimodal_transformer_architecture_corrected.docx"
    
    try:
        # Open the Word document
        doc = Document(docx_path)
        
        # Access the document relationships to find images
        rels = doc.part.rels
        
        image_count = 0
        for rel in rels:
            if "image" in rels[rel].target_ref:
                image_count += 1
                # Get the image data
                image_data = rels[rel].target_part.blob
                
                # Get the file extension from the content type
                content_type = rels[rel].target_part.content_type
                if 'png' in content_type:
                    ext = '.png'
                elif 'jpeg' in content_type or 'jpg' in content_type:
                    ext = '.jpg'
                elif 'gif' in content_type:
                    ext = '.gif'
                else:
                    ext = '.png'  # default
                
                # Save the image
                filename = f"docx_image_{image_count}{ext}"
                with open(filename, 'wb') as f:
                    f.write(image_data)
                
                print(f"📷 Extracted image {image_count}: {filename} ({len(image_data)} bytes)")
                
                # If this is likely Figure 1 (first substantial image), copy it
                if image_count == 1 and len(image_data) > 50000:  # First substantial image
                    with open("docs/figures/architecture_figure1.png", 'wb') as f:
                        f.write(image_data)
                    print(f"✅ Set as Figure 1: {filename}")
        
        if image_count == 0:
            print("❌ No images found in the Word document")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error extracting from Word document: {e}")
        return False

def alternative_extraction():
    """Alternative method using python-docx2txt if available."""
    try:
        import docx2txt
        
        # Extract images using docx2txt
        docx_path = "preprints/multimodal_transformer_architecture_corrected.docx"
        
        # Create a temporary directory for extraction
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract text and images
            text = docx2txt.process(docx_path, temp_dir)
            
            # List extracted images
            import glob
            images = glob.glob(os.path.join(temp_dir, "*"))
            
            if images:
                print(f"📷 Found {len(images)} images using docx2txt")
                
                # Copy the first image as Figure 1
                first_image = images[0]
                import shutil
                shutil.copy2(first_image, "docs/figures/architecture_figure1.png")
                print(f"✅ Copied {first_image} as Figure 1")
                
                return True
            else:
                print("❌ No images found using docx2txt")
                return False
                
    except ImportError:
        print("❌ docx2txt not available")
        return False

if __name__ == "__main__":
    try:
        # Try the main extraction method
        if not extract_images_from_docx():
            # Try alternative method
            print("Trying alternative extraction method...")
            alternative_extraction()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Installing required packages...")
        import subprocess
        subprocess.run(["pip", "install", "python-docx", "docx2txt"])
        
        # Try again
        if not extract_images_from_docx():
            alternative_extraction()
