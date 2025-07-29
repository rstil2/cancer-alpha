#!/usr/bin/env python3
"""
Demo Package Server for Cancer Alpha
====================================

This script creates a simple HTTP server to serve the demo package download.
Users can download the self-contained demo package from the README links.

Usage:
    python3 serve_demo.py

Then visit: http://localhost:8000/cancer_genomics_ai_demo.zip
"""

import http.server
import socketserver
import os
import sys
import zipfile
import shutil
from pathlib import Path

PORT = 8000
DEMO_PACKAGE_DIR = "DEMO_PACKAGE/cancer_genomics_ai_demo"
ZIP_NAME = "cancer_genomics_ai_demo.zip"

def create_demo_zip():
    """Create the demo package ZIP file"""
    print("📦 Creating demo package ZIP file...")
    
    # Ensure demo directory exists
    if not os.path.exists(DEMO_PACKAGE_DIR):
        print(f"❌ Demo directory not found: {DEMO_PACKAGE_DIR}")
        return False
    
    # Create ZIP file
    with zipfile.ZipFile(ZIP_NAME, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add all files from demo package
        for root, dirs, files in os.walk(DEMO_PACKAGE_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                # Create archive path (remove DEMO_PACKAGE/ prefix)
                archive_path = os.path.relpath(file_path, "DEMO_PACKAGE")
                zipf.write(file_path, archive_path)
                print(f"  📄 Added: {archive_path}")
    
    file_size = os.path.getsize(ZIP_NAME) / (1024 * 1024)  # MB
    print(f"✅ Created {ZIP_NAME} ({file_size:.1f} MB)")
    return True

class DemoHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler for demo downloads"""
    
    def do_GET(self):
        if self.path == f"/{ZIP_NAME}":
            # Serve the demo ZIP file
            try:
                with open(ZIP_NAME, 'rb') as f:
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/zip')
                    self.send_header('Content-Disposition', f'attachment; filename="{ZIP_NAME}"')
                    self.send_header('Content-Length', str(os.path.getsize(ZIP_NAME)))
                    self.end_headers()
                    shutil.copyfileobj(f, self.wfile)
                print(f"📥 Demo package downloaded from {self.client_address[0]}")
            except FileNotFoundError:
                self.send_error(404, "Demo package not found")
        else:
            # For other requests, show a simple info page
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Cancer Alpha Demo Server</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                    .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                    h1 {{ color: #2c3e50; text-align: center; }}
                    .download-btn {{ display: inline-block; background: #4CAF50; color: white; padding: 15px 30px; text-decoration: none; border-radius: 5px; font-size: 18px; margin: 20px 0; }}
                    .download-btn:hover {{ background: #45a049; }}
                    .info {{ background: #e3f2fd; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                    .warning {{ background: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #ffa000; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🧬 Cancer Alpha Demo Server</h1>
                    
                    <div class="info">
                        <h3>📦 Demo Package Available</h3>
                        <p>The Cancer Alpha interactive demo is ready for download. This self-contained package includes:</p>
                        <ul>
                            <li>🤖 Real TCGA Production Models (97.6% & 88.6% accuracy)</li>
                            <li>🔍 SHAP Explainability Interface</li>
                            <li>📊 Interactive Streamlit Web Application</li>
                            <li>🧬 Multi-Modal Genomic Data Analysis</li>
                            <li>🖥️ Cross-Platform Startup Scripts</li>
                        </ul>
                    </div>
                    
                    <div style="text-align: center;">
                        <a href="/{ZIP_NAME}" class="download-btn">⬇️ Download Demo Package</a>
                        <p><strong>File:</strong> {ZIP_NAME} ({os.path.getsize(ZIP_NAME) / (1024*1024):.1f} MB)</p>
                    </div>
                    
                    <div class="warning">
                        <h3>⚠️ Patent Protection Notice</h3>
                        <p>This technology is patent-protected (Provisional Application No. 63/847,316). 
                        This demo is for evaluation purposes only. Commercial use requires separate licensing.</p>
                    </div>
                    
                    <div class="info">
                        <h3>🚀 Quick Start</h3>
                        <ol>
                            <li>Download and extract the ZIP file</li>
                            <li>Run <code>start_demo.sh</code> (Mac/Linux) or <code>start_demo.bat</code> (Windows)</li>
                            <li>Open browser to <code>http://localhost:8501</code></li>
                            <li>Select a production model and test cancer classification!</li>
                        </ol>
                    </div>
                </div>
            </body>
            </html>
            """
            self.wfile.write(html.encode())

def main():
    """Start the demo server"""
    print("🧬 Cancer Alpha Demo Server")
    print("=" * 40)
    
    # Create demo package if it doesn't exist
    if not os.path.exists(ZIP_NAME):
        if not create_demo_zip():
            print("❌ Failed to create demo package")
            sys.exit(1)
    else:
        file_size = os.path.getsize(ZIP_NAME) / (1024 * 1024)
        print(f"📦 Using existing demo package: {ZIP_NAME} ({file_size:.1f} MB)")
    
    # Start server
    try:
        with socketserver.TCPServer(("", PORT), DemoHandler) as httpd:
            print(f"🌐 Demo server running at: http://localhost:{PORT}")
            print(f"📥 Demo download link: http://localhost:{PORT}/{ZIP_NAME}")
            print("📝 Press Ctrl+C to stop the server")
            print()
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Demo server stopped")
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ Port {PORT} is already in use. Try stopping other servers or use a different port.")
        else:
            print(f"❌ Server error: {e}")

if __name__ == "__main__":
    main()
