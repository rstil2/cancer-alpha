#!/usr/bin/env python3
"""
Convert the architecture paper markdown to PDF with patent information.
"""

import markdown
import pdfkit
import os

def convert_md_to_pdf():
    """Convert the architecture paper from markdown to PDF."""
    
    # Read the markdown file
    with open('scientific_reports_methodology_paper_with_citations.md', 'r') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html = markdown.markdown(md_content, extensions=['extra', 'codehilite'])
    
    # Add CSS styling for better PDF appearance
    html_with_style = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                line-height: 1.6;
            }}
            h1, h2, h3, h4 {{
                color: #333;
                margin-top: 30px;
            }}
            h1 {{
                border-bottom: 2px solid #333;
                padding-bottom: 10px;
            }}
            .patent-warning {{
                background-color: #fff3cd;
                border: 2px solid #ffc107;
                padding: 15px;
                margin: 20px 0;
                border-radius: 5px;
            }}
            code {{
                background-color: #f4f4f4;
                padding: 2px 4px;
                border-radius: 3px;
            }}
            pre {{
                background-color: #f4f4f4;
                padding: 10px;
                border-radius: 5px;
                overflow-x: auto;
            }}
            blockquote {{
                border-left: 4px solid #ccc;
                padding-left: 15px;
                margin-left: 0;
                font-style: italic;
            }}
        </style>
    </head>
    <body>
        {html}
    </body>
    </html>
    """
    
    # Convert HTML to PDF using wkhtmltopdf alternative or weasyprint
    try:
        # Try using weasyprint
        from weasyprint import HTML, CSS
        
        HTML(string=html_with_style).write_pdf(
            'preprints/multimodal_transformer_architecture_corrected.pdf'
        )
        print("✅ Successfully converted architecture paper to PDF using WeasyPrint")
        
    except ImportError:
        try:
            # Try using pdfkit (wkhtmltopdf wrapper)
            options = {
                'page-size': 'A4',
                'margin-top': '0.75in',
                'margin-right': '0.75in',
                'margin-bottom': '0.75in',
                'margin-left': '0.75in',
                'encoding': "UTF-8",
                'no-outline': None
            }
            
            pdfkit.from_string(html_with_style, 
                             'preprints/multimodal_transformer_architecture_corrected.pdf',
                             options=options)
            print("✅ Successfully converted architecture paper to PDF using pdfkit")
            
        except Exception as e:
            # Fallback: save as HTML
            with open('preprints/multimodal_transformer_architecture_corrected.html', 'w') as f:
                f.write(html_with_style)
            print(f"⚠️  PDF conversion failed: {e}")
            print("📄 Saved as HTML instead: preprints/multimodal_transformer_architecture_corrected.html")
            return False
    
    return True

if __name__ == "__main__":
    success = convert_md_to_pdf()
    if success:
        print("🎉 Architecture paper with patent information is ready!")
    else:
        print("❌ Please manually convert the HTML file to PDF")
