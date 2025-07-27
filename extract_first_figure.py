#!/usr/bin/env python3
"""
Extract the first substantial figure image assuming it's Figure 1 from the PDF.
"""

import fitz  # PyMuPDF
import os
from PIL import Image
import io

def extract_first_substantial_image():
    """Extract the first substantial image, aiming to accurately find Figure 1."""
    
    pdf_path = "preprints/multimodal_transformer_architecture_corrected.pdf"
    output_path = "docs/figures/architecture_figure1.png"
    
    try:
        # Open the PDF
        doc = fitz.open(pdf_path)
        print(f"📄 Opened PDF with {len(doc)} pages")
        
        # Loop over the first few pages to find Figure 1
        for page_num in range(min(3, len(doc))):  # Check early pages
            page = doc.load_page(page_num)
            
            # Get all images on this page
            image_list = page.get_images()
            
            if image_list:
                print(f"📊 Found {len(image_list)} images on page {page_num + 1}")
                
                # Extract images
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        
                        # Focus on larger images to find the main figure
                        if len(img_data) > 100000:  # Considerably large
                            with open(output_path, "wb") as f:
                                f.write(img_data)
                            
                            print(f"✅ Extracted Figure 1 to {output_path}")
                            print(f"📏 Image size: {len(img_data)} bytes")
                            
                            # Try to get image dimensions
                            img_pil = Image.open(io.BytesIO(img_data))
                            print(f"📐 Dimensions: {img_pil.size[0]}x{img_pil.size[1]} pixels")
                            
                            # Assume this is Figure 1 and stop search
                            return True
                    
                    pix = None
        
        doc.close()
        
        print("❌ Figure 1 not found in the early pages")
        return False
        
    except Exception as e:
        print(f"❌ Error extracting figure: {e}")
        return False


if __name__ == "__main__":
    try:
        # Extract Figure 1 from initial pages
        extract_first_substantial_image()
        
    except Exception as e:
        print(f"❌ Error: {e}")
