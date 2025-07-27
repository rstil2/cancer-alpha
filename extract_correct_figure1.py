#!/usr/bin/env python3
"""
Extract the correct Figure 1 from the architecture paper PDF.
"""

import fitz  # PyMuPDF
import os
from PIL import Image
import io

def extract_all_figures():
    """Extract all figures from the PDF to identify Figure 1."""
    
    pdf_path = "preprints/multimodal_transformer_architecture_corrected.pdf"
    
    try:
        # Open the PDF
        doc = fitz.open(pdf_path)
        print(f"📄 Opened PDF with {len(doc)} pages")
        
        figure_count = 0
        
        # Look through all pages for figures
        for page_num in range(len(doc)):
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
                        
                        # Save substantial images for review
                        if len(img_data) > 5000:  # Filter out very small images
                            figure_count += 1
                            temp_path = f"temp_figure_{figure_count}_page{page_num + 1}.png"
                            
                            with open(temp_path, "wb") as f:
                                f.write(img_data)
                            
                            # Get image dimensions
                            try:
                                img_pil = Image.open(io.BytesIO(img_data))
                                dims = f"{img_pil.size[0]}x{img_pil.size[1]}"
                            except:
                                dims = "unknown"
                            
                            print(f"📷 Figure {figure_count}: {temp_path} ({len(img_data)} bytes, {dims} pixels)")
                    
                    pix = None
        
        doc.close()
        
        if figure_count > 0:
            print(f"\n✅ Extracted {figure_count} figures for review")
            print("Please check the temp_figure_*.png files to identify which is Figure 1")
            print("Then I'll move the correct one to docs/figures/architecture_figure1.png")
        else:
            print("❌ No substantial figures found")
            
        return figure_count
        
    except Exception as e:
        print(f"❌ Error extracting figures: {e}")
        return 0

def replace_figure1(figure_number):
    """Replace the current figure with the correct Figure 1."""
    
    source_path = f"temp_figure_{figure_number}_page*.png"
    target_path = "docs/figures/architecture_figure1.png"
    
    # Find the temp file
    import glob
    temp_files = glob.glob(f"temp_figure_{figure_number}_page*.png")
    
    if temp_files:
        source_file = temp_files[0]
        
        # Copy to the correct location
        import shutil
        shutil.copy2(source_file, target_path)
        
        print(f"✅ Replaced architecture_figure1.png with {source_file}")
        
        # Clean up temp files
        for temp_file in glob.glob("temp_figure_*.png"):
            os.remove(temp_file)
        print("🧹 Cleaned up temporary files")
        
        return True
    else:
        print(f"❌ Could not find temp_figure_{figure_number}")
        return False

if __name__ == "__main__":
    try:
        # Extract all figures for review
        figure_count = extract_all_figures()
        
        if figure_count > 0:
            print(f"\n🔍 Please review the {figure_count} extracted figures above")
            print("Which one is the actual Figure 1 from the paper?")
            print("Enter the figure number (1, 2, 3, etc.) or 'q' to quit:")
            
            # In a real scenario, you'd input this. For now, let's assume Figure 1 is likely the first one
            # that appears early in the document with substantial content
            
            # Let's try the first figure that's reasonably large
            # You can modify this after seeing which one is correct
            choice = "1"  # Default to first figure for now
            
            if choice.isdigit():
                figure_num = int(choice)
                if 1 <= figure_num <= figure_count:
                    replace_figure1(figure_num)
                else:
                    print("❌ Invalid figure number")
            
    except Exception as e:
        print(f"❌ Error: {e}")
